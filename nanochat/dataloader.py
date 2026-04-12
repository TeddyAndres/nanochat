"""
Distributed dataloaders for pretraining.

BOS-aligned bestfit:
   - Every row starts with BOS token
   - Documents packed using best-fit algorithm to minimize cropping
   - When no document fits remaining space, crops a document to fill exactly
   - 100% utilization (no padding), ~35% tokens cropped at T=2048

Compared to the original tokenizing_distributed_data_loader:
BOS-aligned loses ~35% of tokens to cropping, but ensures that
there are fewer "confusing" tokens in the train/val batches as every token can
now attend back to the BOS token and sees the full context of the document.

Fallback to the original if you have very limited data AND long documents:
https://github.com/karpathy/nanochat/blob/3c3a3d7/nanochat/dataloader.py#L78-L117
"""

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import cast

import torch

from nanochat.common import get_dist_info
from nanochat.sparse_manifest import (
    DUAL_SPARSE_MANIFEST_VERSION,
    SPARSE_GROUPING_MANIFEST_KIND,
    SPARSE_SEQUENCE_BASE_MANIFEST_KIND,
    SequenceManifestShardAccessor,
    compute_next_transition,
    get_sparse_manifest_kind,
    load_sparse_manifest_header,
    resolve_grouping_base_manifest_path,
    stream_sparse_manifest_steps,
    validate_sequence_manifest,
    validate_sparse_manifest,
)
from nanochat.token_cache import (
    ensure_token_cache,
    iter_cached_token_batches,
    iter_document_text_batches,
    load_cached_token_batch_by_state,
    prepare_token_cache_writer,
    resolve_token_cache_dir,
)


def _tokenize_document_batch(tokenizer, text_batch, state, bos_token, tokenizer_threads):
    token_lists = [
        torch.tensor(tokens, dtype=torch.long)
        for tokens in tokenizer.encode(text_batch, prepend=bos_token, num_threads=tokenizer_threads)
    ]
    return token_lists, state

def _iter_tokenized_document_batches(
    tokenizer,
    *,
    split,
    tokenizer_threads,
    tokenizer_batch_size,
    resume_state_dict,
    token_cache_dir,
    token_cache_shard_batches,
    token_cache_workers,
):
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    bos_token = tokenizer.get_bos_token_id()
    if token_cache_dir is not None:
        resolved_cache_dir = resolve_token_cache_dir(token_cache_dir)
        ensure_token_cache(
            resolved_cache_dir,
            split,
            tokenizer=tokenizer,
            tokenizer_threads=tokenizer_threads,
            tokenizer_batch_size=tokenizer_batch_size,
            bos_token_id=bos_token,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
            shard_batch_count=token_cache_shard_batches,
            num_workers=token_cache_workers,
        )
        yield from iter_cached_token_batches(
            resolved_cache_dir,
            split,
            resume_state_dict=resume_state_dict,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
            repeat=True,
        )
        return

    live_text_batches = iter_document_text_batches(
        split,
        resume_state_dict,
        tokenizer_batch_size,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
    )
    prefetch_futures: deque[Future] = deque()

    def enqueue_prefetch(executor: ThreadPoolExecutor) -> None:
        while len(prefetch_futures) < 2:
            try:
                text_batch, state = next(live_text_batches)
            except StopIteration:
                break
            prefetch_futures.append(
                executor.submit(
                    _tokenize_document_batch,
                    tokenizer,
                    text_batch,
                    state,
                    bos_token,
                    tokenizer_threads,
                )
            )

    try:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="tokenize-prefetch") as executor:
            enqueue_prefetch(executor)
            while prefetch_futures:
                token_lists, state = prefetch_futures.popleft().result()
                enqueue_prefetch(executor)
                yield token_lists, state
    finally:
        pass


def tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, B, T, split,
    tokenizer_threads=4, tokenizer_batch_size=128,
    device="cuda", resume_state_dict=None,
    buffer_size=1000,
    return_active_vocab=False,
    return_sequence_recipe=False,
    pin_memory_output=False,
    vocab_size=None,
    token_cache_dir=None,
    token_cache_shard_batches=256,
    token_cache_workers=1,
):
    """
    BOS-aligned dataloader with Best-Fit Cropping.

    Reduces token waste compared to simple greedy cropping by searching a buffer
    for documents that fit well, while maintaining 100% utilization (no padding).

    Algorithm for each row:
    1. From buffered docs, pick the LARGEST doc that fits entirely
    2. Repeat until no doc fits
    3. When nothing fits, crop a doc to fill remaining space exactly

    Key properties:
    - Every row starts with BOS
    - 100% utilization (no padding, every token is trained on)
    - Approximately 35% of all tokens are discarded due to cropping
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    row_capacity = T + 1
    token_batches = _iter_tokenized_document_batches(
        tokenizer,
        split=split,
        tokenizer_threads=tokenizer_threads,
        tokenizer_batch_size=tokenizer_batch_size,
        resume_state_dict=resume_state_dict,
        token_cache_dir=token_cache_dir,
        token_cache_shard_batches=token_cache_shard_batches,
        token_cache_workers=token_cache_workers,
    )
    doc_buffer = []
    pq_idx, rg_idx, epoch = 0, 0, 1

    def refill_buffer():
        nonlocal pq_idx, rg_idx, epoch
        token_lists, state = next(token_batches)
        pq_idx = int(state["pq_idx"])
        rg_idx = int(state["rg_idx"])
        epoch = int(state["epoch"])
        for doc_index_in_batch, tokens in enumerate(token_lists):
            doc_buffer.append({
                "tokens": tokens,
                "source_state": dict(state),
                "doc_index_in_batch": int(doc_index_in_batch),
            })

    # Pre-allocate buffers once: layout is [inputs (B*T) | targets (B*T)]
    # This gives us contiguous views and a single HtoD transfer
    output_device = torch.device(device)
    use_cuda = output_device.type == "cuda"
    use_pinned_output = use_cuda or bool(pin_memory_output)
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long) # for building rows without creating Python lists
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_pinned_output) # staging area (CPU)
    gpu_buffer = torch.empty(
        2 * B * T,
        dtype=torch.long,
        device=device,
        pin_memory=(output_device.type == "cpu" and use_pinned_output),
    ) # on-device buffer or pinned CPU output buffer
    cpu_inputs = cpu_buffer[:B * T].view(B, T) # a few views into these buffers just for convenience
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)
    if return_active_vocab:
        assert vocab_size is not None, "vocab_size is required when return_active_vocab=True"
        global_to_local = torch.full((vocab_size,), -1, dtype=torch.long)

    while True:
        batch_recipe_rows = [] if return_sequence_recipe else None
        for row_idx in range(B):
            pos = 0
            row_recipe_segments = [] if return_sequence_recipe else None
            while pos < row_capacity:
                # Ensure buffer has documents
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # Find largest doc that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc["tokens"])
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    tokens = cast(torch.Tensor, doc["tokens"])
                    doc_len = int(tokens.numel())
                    row_buffer[row_idx, pos:pos + doc_len] = tokens
                    if row_recipe_segments is not None:
                        row_recipe_segments.append({
                            "source_state": dict(cast(dict, doc["source_state"])),
                            "doc_index_in_batch": int(doc["doc_index_in_batch"]),
                            "start_offset": 0,
                            "end_offset": doc_len,
                        })
                    pos += doc_len
                else:
                    # No doc fits - crop shortest in buffer to fill remaining and minimize waste
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(cast(torch.Tensor, doc_buffer[i]["tokens"])))
                    doc = doc_buffer.pop(shortest_idx)
                    tokens = cast(torch.Tensor, doc["tokens"])
                    row_buffer[row_idx, pos:pos + remaining] = tokens[:remaining]
                    if row_recipe_segments is not None:
                        row_recipe_segments.append({
                            "source_state": dict(cast(dict, doc["source_state"])),
                            "doc_index_in_batch": int(doc["doc_index_in_batch"]),
                            "start_offset": 0,
                            "end_offset": int(remaining),
                        })
                    pos += remaining
            if batch_recipe_rows is not None:
                assert row_recipe_segments is not None
                batch_recipe_rows.append({
                    "row_index": int(row_idx),
                    "segments": row_recipe_segments,
                })

        # Copy to pinned CPU buffer, then single HtoD transfer
        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        active_ids = None
        if return_active_vocab:
            active_ids = torch.unique(cpu_buffer, sorted=True)
            global_to_local.fill_(-1)
            global_to_local[active_ids] = torch.arange(active_ids.numel(), dtype=torch.long)
            cpu_inputs.copy_(global_to_local[cpu_inputs])
            cpu_targets.copy_(global_to_local[cpu_targets])
            if use_cuda:
                active_ids = active_ids.pin_memory()

        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}
        sequence_recipe = None
        if batch_recipe_rows is not None:
            sequence_recipe = {
                "device_batch_size": int(B),
                "max_seq_len": int(T),
                "row_capacity": int(row_capacity),
                "rows": batch_recipe_rows,
            }

        # Single HtoD copy into persistent GPU buffer and yield
        if gpu_buffer.data_ptr() != cpu_buffer.data_ptr():
            gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        if return_active_vocab and return_sequence_recipe:
            yield inputs, targets, active_ids, state_dict, sequence_recipe
        elif return_active_vocab:
            yield inputs, targets, active_ids, state_dict
        elif return_sequence_recipe:
            yield inputs, targets, state_dict, sequence_recipe
        else:
            yield inputs, targets, state_dict

def tokenizing_distributed_data_loader_bos_bestfit(*args, **kwargs):
    """Helper that omits state_dict from yields."""
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state_bos_bestfit(*args, **kwargs):
        yield inputs, targets


def tokenizing_distributed_data_loader_with_state_bos_bestfit_dynamic(*args, **kwargs):
    """Dynamic-vocab variant that also returns active vocab ids for each batch."""
    kwargs["return_active_vocab"] = True
    for inputs, targets, active_ids, state_dict in tokenizing_distributed_data_loader_with_state_bos_bestfit(*args, **kwargs):
        yield inputs, targets, active_ids, state_dict


def tokenizing_distributed_data_loader_with_state_bos_bestfit_manifest(
    tokenizer, B, T, split,
    manifest_path,
    tokenizer_threads=4, tokenizer_batch_size=128,
    device="cuda", resume_state_dict=None,
    buffer_size=1000,
    pin_memory_output=False,
    vocab_size=None,
    include_local_batch=False,
    step_override_provider=None,
    use_sequence_base_manifest=False,
    token_cache_dir="",
    token_cache_shard_batches=256,
    token_cache_workers=1,
):
    """Manifest-driven sparse loader with fixed logical U slots.

    The live BOS best-fit batch construction stays unchanged, but vocab remapping and
    overlap transitions are driven by a precomputed sparse-manifest JSON.
    """
    assert vocab_size is not None, "vocab_size is required for manifest sparse loader"
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    manifest = load_sparse_manifest_header(manifest_path)
    manifest_version = int(manifest.get("version", 1))
    manifest_kind = get_sparse_manifest_kind(manifest)
    if manifest_kind != SPARSE_GROUPING_MANIFEST_KIND:
        raise ValueError("Manifest sparse loader requires a grouping sparse manifest")
    validate_sparse_manifest(
        manifest,
        split=split,
        vocab_size=vocab_size,
        device_batch_size=B,
        max_seq_len=T,
        grad_accum_steps=int(manifest.get("grad_accum_steps", 1)),
        ddp_world_size=ddp_world_size,
    )

    u_max = int(manifest["u_max"])
    num_manifest_steps = int(manifest["num_steps"])
    if u_max <= 0:
        raise ValueError("Sparse manifest must define a positive u_max")

    base_resume_state = None if resume_state_dict is None else {
        key: value
        for key, value in resume_state_dict.items()
        if key != "manifest_step"
    }
    manifest_step = 0 if resume_state_dict is None else int(resume_state_dict.get("manifest_step", 0))
    if manifest_step < 0 or manifest_step >= num_manifest_steps:
        raise ValueError(
            f"Sparse manifest step {manifest_step} is out of range for {num_manifest_steps} stored steps"
        )

    use_dual_manifest = (
        bool(use_sequence_base_manifest) and
        manifest_version >= DUAL_SPARSE_MANIFEST_VERSION and
        "base_manifest_path" in manifest
    )
    base_loader = None
    resolve_sequence_unit = None
    resolved_cache_dir = None
    token_batch_cache = {}
    if use_dual_manifest:
        base_manifest_path = resolve_grouping_base_manifest_path(manifest_path, manifest)
        base_manifest = load_sparse_manifest_header(base_manifest_path)
        if get_sparse_manifest_kind(base_manifest) != SPARSE_SEQUENCE_BASE_MANIFEST_KIND:
            raise ValueError("Grouping manifest base_manifest_path must reference a sequence-base manifest")
        validate_sequence_manifest(
            base_manifest,
            split=split,
            vocab_size=vocab_size,
            device_batch_size=B,
            max_seq_len=T,
            ddp_world_size=ddp_world_size,
        )

        shard_entries = base_manifest.get("shards")
        assert isinstance(shard_entries, list) and len(shard_entries) > 0
        resolved_cache_dir = resolve_token_cache_dir(token_cache_dir)
        shard_ranges = []
        for shard_entry in shard_entries:
            start_sequence_id = int(shard_entry.get("start_sequence_id", -1))
            shard_num_units = int(shard_entry.get("num_sequence_units", 0))
            if start_sequence_id < 0 or shard_num_units <= 0:
                raise ValueError("Sequence manifest shard entries must define start_sequence_id and num_sequence_units")
            shard_ranges.append((start_sequence_id, start_sequence_id + shard_num_units, shard_entry))
        loaded_sequence_shard_path = None
        loaded_sequence_shard_accessor = None

        def resolve_sequence_unit(sequence_id: int) -> dict:
            nonlocal loaded_sequence_shard_path, loaded_sequence_shard_accessor
            for start_sequence_id, end_sequence_id, shard_entry in shard_ranges:
                if start_sequence_id <= sequence_id < end_sequence_id:
                    shard_path = base_manifest_path.parent / str(shard_entry["path"])
                    if loaded_sequence_shard_path != shard_path:
                        if loaded_sequence_shard_accessor is not None:
                            loaded_sequence_shard_accessor.close()
                        loaded_sequence_shard_accessor = SequenceManifestShardAccessor(shard_path)
                        loaded_sequence_shard_path = shard_path
                    assert loaded_sequence_shard_accessor is not None
                    sequence_unit = loaded_sequence_shard_accessor.get_sequence_unit(sequence_id)
                    if sequence_unit is None:
                        raise ValueError(f"Sequence manifest shard is missing sequence_id={sequence_id}")
                    return sequence_unit
            raise ValueError(f"Sequence manifest is missing sequence_id={sequence_id}")
    else:
        base_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
            tokenizer,
            B,
            T,
            split,
            tokenizer_threads=tokenizer_threads,
            tokenizer_batch_size=tokenizer_batch_size,
            device="cpu",
            resume_state_dict=base_resume_state,
            buffer_size=buffer_size,
            pin_memory_output=pin_memory_output,
            token_cache_dir=token_cache_dir,
            token_cache_shard_batches=token_cache_shard_batches,
            token_cache_workers=token_cache_workers,
        )

    output_device = torch.device(device)
    use_cuda = output_device.type == "cuda"
    use_pinned_output = use_cuda or bool(pin_memory_output)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_pinned_output)
    gpu_buffer = torch.empty(
        2 * B * T,
        dtype=torch.long,
        device=device,
        pin_memory=(output_device.type == "cpu" and use_pinned_output),
    )
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    slot_to_global = torch.full((u_max,), -1, dtype=torch.long)
    global_to_slot = torch.full((vocab_size,), -1, dtype=torch.long)
    use_manifest_transitions = step_override_provider is None

    def apply_manifest_transition(step_entry):
        next_leaving_ids = torch.tensor(step_entry["next_leaving_ids"], dtype=torch.long)
        next_new_ids = torch.tensor(step_entry["next_new_ids"], dtype=torch.long)
        if next_leaving_ids.numel() > 0:
            leaving_slots = global_to_slot[next_leaving_ids]
            if (leaving_slots < 0).any():
                raise ValueError("Sparse manifest leaving ids are not present in the current slot map")
            slot_to_global[leaving_slots] = -1
            global_to_slot[next_leaving_ids] = -1
        else:
            leaving_slots = torch.empty(0, dtype=torch.long)
        if next_new_ids.numel() == 0:
            return next_new_ids, next_new_ids, next_leaving_ids, leaving_slots
        reusable_slots = leaving_slots.tolist()
        reusable_slots.extend(torch.nonzero(slot_to_global < 0, as_tuple=False).flatten().tolist())
        assigned_slots = []
        seen = set()
        for slot in reusable_slots:
            if slot in seen:
                continue
            seen.add(slot)
            assigned_slots.append(int(slot))
            if len(assigned_slots) == next_new_ids.numel():
                break
        if len(assigned_slots) < next_new_ids.numel():
            raise ValueError(
                f"Sparse manifest overflow: need {next_new_ids.numel()} new slots, only found {len(assigned_slots)} free slots out of U_max={u_max}"
            )
        assigned_slots_tensor = torch.tensor(assigned_slots, dtype=torch.long)
        slot_to_global[assigned_slots_tensor] = next_new_ids
        global_to_slot[next_new_ids] = assigned_slots_tensor
        return next_new_ids, assigned_slots_tensor, next_leaving_ids, leaving_slots

    def reconcile_active_ids(desired_active_ids: torch.Tensor):
        desired_active_ids = desired_active_ids.detach().to(device="cpu", dtype=torch.long)
        current_active_slot_ids = torch.nonzero(slot_to_global >= 0, as_tuple=False).flatten()
        current_active_ids = slot_to_global[current_active_slot_ids] if current_active_slot_ids.numel() > 0 else torch.empty(0, dtype=torch.long)
        _, next_leaving_ids, next_new_ids = compute_next_transition(current_active_ids, desired_active_ids)
        leaving_slots = torch.empty(0, dtype=torch.long)
        if next_leaving_ids.numel() > 0:
            leaving_slots = global_to_slot[next_leaving_ids]
            if (leaving_slots < 0).any():
                raise ValueError("Sparse manifest leaving ids are not present in the current slot map")
            slot_to_global[leaving_slots] = -1
            global_to_slot[next_leaving_ids] = -1
        else:
            leaving_slots = torch.empty(0, dtype=torch.long)
        if next_new_ids.numel() == 0:
            return next_new_ids, torch.empty(0, dtype=torch.long), next_leaving_ids, leaving_slots
        reusable_slots = leaving_slots.tolist()
        reusable_slots.extend(torch.nonzero(slot_to_global < 0, as_tuple=False).flatten().tolist())
        assigned_slots = []
        seen = set()
        for slot in reusable_slots:
            if slot in seen:
                continue
            seen.add(slot)
            assigned_slots.append(int(slot))
            if len(assigned_slots) == next_new_ids.numel():
                break
        if len(assigned_slots) < next_new_ids.numel():
            raise ValueError(
                f"Sparse manifest overflow: need {next_new_ids.numel()} new slots, only found {len(assigned_slots)} free slots out of U_max={u_max}"
            )
        assigned_slots_tensor = torch.tensor(assigned_slots, dtype=torch.long)
        slot_to_global[assigned_slots_tensor] = next_new_ids
        global_to_slot[next_new_ids] = assigned_slots_tensor
        return next_new_ids, assigned_slots_tensor, next_leaving_ids, leaving_slots

    def preview_next_transition(next_active_ids: torch.Tensor):
        next_active_ids = next_active_ids.detach().to(device="cpu", dtype=torch.long)
        current_active_slot_ids = torch.nonzero(slot_to_global >= 0, as_tuple=False).flatten()
        current_active_ids = slot_to_global[current_active_slot_ids] if current_active_slot_ids.numel() > 0 else torch.empty(0, dtype=torch.long)
        _, leaving_ids, new_ids = compute_next_transition(current_active_ids, next_active_ids)
        if leaving_ids.numel() > 0:
            leaving_slots = global_to_slot[leaving_ids]
            if (leaving_slots < 0).any():
                raise ValueError("Sparse manifest leaving ids are not present in the current slot map")
        else:
            leaving_slots = torch.empty(0, dtype=torch.long)
        if new_ids.numel() == 0:
            new_slots = torch.empty(0, dtype=torch.long)
        else:
            free_slots = torch.nonzero(slot_to_global < 0, as_tuple=False).flatten().tolist()
            free_slots = [int(slot) for slot in free_slots if int(slot) not in set(leaving_slots.tolist())]
            reusable_slots = leaving_slots.tolist() + free_slots
            new_slots = torch.tensor(reusable_slots[:new_ids.numel()], dtype=torch.long)
            if new_slots.numel() < new_ids.numel():
                raise ValueError(
                    f"Sparse manifest overflow: need {new_ids.numel()} new slots, only found {new_slots.numel()} free slots out of U_max={u_max}"
                )
        return new_ids, new_slots, leaving_ids, leaving_slots

    def resolve_step_entry(step_index: int, default_step_entry: dict):
        if step_override_provider is None:
            return default_step_entry
        override_entry = step_override_provider(step_index)
        return default_step_entry if override_entry is None else override_entry

    manifest_iter = stream_sparse_manifest_steps(manifest_path)
    try:
        current_step_entry = resolve_step_entry(0, next(manifest_iter))
    except StopIteration as exc:
        raise ValueError("Sparse manifest contains no step entries") from exc

    def get_microsteps(step_entry):
        if manifest_version >= 2:
            microsteps = step_entry.get("microsteps")
            if not isinstance(microsteps, list) or len(microsteps) == 0:
                raise ValueError("Sparse manifest step is missing microsteps")
            return microsteps
        return [step_entry]

    current_microsteps = get_microsteps(current_step_entry)
    current_micro_idx = 0
    current_micro_entry = current_microsteps[current_micro_idx]
    buffered_next_step_entry = None

    first_active_ids = torch.tensor(current_micro_entry["active_ids"], dtype=torch.long)
    if first_active_ids.numel() > u_max:
        raise ValueError(
            f"Sparse manifest first step exceeds U_max: {first_active_ids.numel()} > {u_max}"
        )
    if use_manifest_transitions:
        first_slots = torch.arange(first_active_ids.numel(), dtype=torch.long)
        slot_to_global[first_slots] = first_active_ids
        global_to_slot[first_active_ids] = first_slots
        current_new_ids = first_active_ids.clone()
        current_new_slots = first_slots.clone()
    else:
        current_new_ids = torch.empty(0, dtype=torch.long)
        current_new_slots = torch.empty(0, dtype=torch.long)
    current_leaving_ids = torch.empty(0, dtype=torch.long)
    current_leaving_slots = torch.empty(0, dtype=torch.long)
    for replay_step in range(manifest_step):
        for replay_micro_entry in current_microsteps:
            if use_manifest_transitions:
                current_new_ids, current_new_slots, current_leaving_ids, current_leaving_slots = apply_manifest_transition(replay_micro_entry)
            else:
                replay_active_ids = torch.tensor(replay_micro_entry["active_ids"], dtype=torch.long)
                current_new_ids, current_new_slots, current_leaving_ids, current_leaving_slots = reconcile_active_ids(replay_active_ids)
        try:
            current_step_entry = resolve_step_entry(replay_step + 1, next(manifest_iter))
        except StopIteration as exc:
            raise ValueError(
                f"Sparse manifest ended before requested resume step {manifest_step}"
            ) from exc
        current_microsteps = get_microsteps(current_step_entry)
        current_micro_idx = 0
        current_micro_entry = current_microsteps[current_micro_idx]

    while True:
        if manifest_step >= num_manifest_steps:
            raise StopIteration
        if not use_manifest_transitions:
            desired_active_ids = torch.tensor(current_micro_entry["active_ids"], dtype=torch.long)
            current_new_ids, current_new_slots, _, _ = reconcile_active_ids(desired_active_ids)
        if use_dual_manifest:
            assert resolve_sequence_unit is not None
            assert resolved_cache_dir is not None
            sequence_id = int(current_micro_entry.get("sequence_id", -1))
            if sequence_id < 0:
                raise ValueError("Dual-manifest microsteps must define a non-negative sequence_id")
            sequence_unit = resolve_sequence_unit(sequence_id)
            if "inputs" in sequence_unit and "targets" in sequence_unit:
                base_inputs = torch.as_tensor(sequence_unit["inputs"], dtype=torch.long)
                base_targets = torch.as_tensor(sequence_unit["targets"], dtype=torch.long)
                if tuple(base_inputs.shape) != (B, T) or tuple(base_targets.shape) != (B, T):
                    raise ValueError(
                        f"Sequence manifest batch shape mismatch for sequence_id={sequence_id}: "
                        f"expected {(B, T)}, found inputs={tuple(base_inputs.shape)} targets={tuple(base_targets.shape)}"
                    )
            else:
                sequence_recipe = sequence_unit.get("sequence_recipe")
                if not isinstance(sequence_recipe, dict):
                    raise ValueError("Sequence manifest units must define either inputs/targets or a sequence_recipe")
                row_capacity = int(sequence_recipe.get("row_capacity", T + 1))
                if row_capacity != T + 1:
                    raise ValueError(
                        f"Sequence manifest row_capacity mismatch for sequence_id={sequence_id}: expected {T + 1}, found {row_capacity}"
                    )
                recipe_rows = sequence_recipe.get("rows")
                if not isinstance(recipe_rows, list) or len(recipe_rows) != B:
                    raise ValueError(
                        f"Sequence manifest rows mismatch for sequence_id={sequence_id}: expected {B}, found {0 if recipe_rows is None else len(recipe_rows)}"
                    )
                row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
                for row_idx, row_entry in enumerate(recipe_rows):
                    segments = row_entry.get("segments") if isinstance(row_entry, dict) else None
                    if not isinstance(segments, list) or len(segments) == 0:
                        raise ValueError(f"Sequence manifest row {row_idx} is missing segments for sequence_id={sequence_id}")
                    pos = 0
                    for segment in segments:
                        if not isinstance(segment, dict):
                            raise ValueError("Sequence manifest segment entries must be dicts")
                        source_state = segment.get("source_state")
                        if not isinstance(source_state, dict):
                            raise ValueError("Sequence manifest segments must define source_state")
                        batch_key = (
                            int(source_state.get("pq_idx", -1)),
                            int(source_state.get("rg_idx", -1)),
                            int(source_state.get("text_batch_index", -1)),
                        )
                        token_lists = token_batch_cache.get(batch_key)
                        if token_lists is None:
                            token_lists = load_cached_token_batch_by_state(
                                resolved_cache_dir,
                                split,
                                pq_idx=batch_key[0],
                                rg_idx=batch_key[1],
                                text_batch_index=batch_key[2],
                            )
                            token_batch_cache[batch_key] = token_lists
                        doc_index_in_batch = int(segment.get("doc_index_in_batch", -1))
                        if doc_index_in_batch < 0 or doc_index_in_batch >= len(token_lists):
                            raise ValueError(
                                f"Sequence manifest segment doc_index_in_batch={doc_index_in_batch} is invalid for sequence_id={sequence_id}"
                            )
                        source_tokens = token_lists[doc_index_in_batch]
                        start_offset = int(segment.get("start_offset", 0))
                        end_offset = int(segment.get("end_offset", -1))
                        if start_offset < 0 or end_offset < start_offset or end_offset > source_tokens.numel():
                            raise ValueError(
                                f"Sequence manifest segment offsets are invalid for sequence_id={sequence_id}: [{start_offset}, {end_offset})"
                            )
                        segment_tokens = source_tokens[start_offset:end_offset]
                        next_pos = pos + int(segment_tokens.numel())
                        if next_pos > row_capacity:
                            raise ValueError(
                                f"Sequence manifest row overflow for sequence_id={sequence_id}: row {row_idx} exceeds row_capacity={row_capacity}"
                            )
                        row_buffer[row_idx, pos:next_pos] = segment_tokens
                        pos = next_pos
                    if pos != row_capacity:
                        raise ValueError(
                            f"Sequence manifest row underfill for sequence_id={sequence_id}: row {row_idx} filled {pos} tokens, expected {row_capacity}"
                        )
                base_inputs = row_buffer[:, :-1]
                base_targets = row_buffer[:, 1:]
            base_state_dict = sequence_unit.get("state_dict", {})
            if not isinstance(base_state_dict, dict):
                raise ValueError("Sequence manifest state_dict payload must be a dict")
            cpu_inputs.copy_(base_inputs)
            cpu_targets.copy_(base_targets)
        else:
            assert base_loader is not None
            base_inputs, base_targets, base_state_dict = next(base_loader)
            cpu_inputs.copy_(base_inputs)
            cpu_targets.copy_(base_targets)
        remapped_inputs = global_to_slot[cpu_inputs]
        remapped_targets = global_to_slot[cpu_targets]
        if (remapped_inputs < 0).any() or (remapped_targets < 0).any():
            raise ValueError(
                f"Sparse manifest mismatch at step {manifest_step}: live batch contains tokens missing from manifest active_ids"
            )
        cpu_inputs.copy_(remapped_inputs)
        cpu_targets.copy_(remapped_targets)

        active_mask_cpu = slot_to_global >= 0
        active_slot_ids_cpu = torch.nonzero(active_mask_cpu, as_tuple=False).flatten()
        active_ids_cpu = slot_to_global[active_slot_ids_cpu]

        grad_accum_ids_cpu = active_ids_cpu.clone()
        accum_steps = 1
        micro_step_index = 0
        targets_union_cpu_local = None
        inputs_union_cpu_local = None
        if manifest_version >= 2:
            grad_accum_ids_cpu = torch.tensor(current_step_entry["grad_accum_active_ids"], dtype=torch.long)
            accum_steps = len(current_microsteps)
            micro_step_index = current_micro_idx
            grad_accum_global_to_slot = torch.full((vocab_size,), -1, dtype=torch.long)
            if grad_accum_ids_cpu.numel() > 0:
                grad_accum_global_to_slot[grad_accum_ids_cpu] = torch.arange(grad_accum_ids_cpu.numel(), dtype=torch.long)
            inputs_union_cpu_local = grad_accum_global_to_slot[slot_to_global.index_select(0, cpu_inputs.reshape(-1))].view_as(cpu_inputs)
            if (inputs_union_cpu_local < 0).any():
                raise ValueError(
                    f"Sparse manifest mismatch at step {manifest_step}: grad-accum union is missing input tokens"
                )
            targets_union_cpu_local = grad_accum_global_to_slot[slot_to_global.index_select(0, cpu_targets.reshape(-1))].view_as(cpu_targets)
            if (targets_union_cpu_local < 0).any():
                raise ValueError(
                    f"Sparse manifest mismatch at step {manifest_step}: grad-accum union is missing target tokens"
                )

        is_last_microstep = current_micro_idx == len(current_microsteps) - 1
        if use_manifest_transitions:
            current_leaving_ids = torch.tensor(current_micro_entry["next_leaving_ids"], dtype=torch.long)
            if current_leaving_ids.numel() > 0:
                current_leaving_slots = global_to_slot[current_leaving_ids]
                if (current_leaving_slots < 0).any():
                    raise ValueError("Sparse manifest leaving ids are not present in the current slot map")
            else:
                current_leaving_slots = torch.empty(0, dtype=torch.long)
        else:
            next_micro_entry = None
            if not is_last_microstep:
                next_micro_entry = current_microsteps[current_micro_idx + 1]
            elif manifest_step < num_manifest_steps - 1:
                if buffered_next_step_entry is None:
                    try:
                        buffered_next_step_entry = resolve_step_entry(manifest_step + 1, next(manifest_iter))
                    except StopIteration as exc:
                        raise ValueError(
                            f"Sparse manifest ended early after step {manifest_step}; expected {num_manifest_steps} total steps"
                        ) from exc
                next_microsteps = get_microsteps(buffered_next_step_entry)
                next_micro_entry = next_microsteps[0]

            if next_micro_entry is not None:
                next_active_ids = torch.tensor(next_micro_entry["active_ids"], dtype=torch.long)
                _, _, current_leaving_ids, current_leaving_slots = preview_next_transition(next_active_ids)
            else:
                current_leaving_ids = torch.empty(0, dtype=torch.long)
                current_leaving_slots = torch.empty(0, dtype=torch.long)

        step_meta = {
            "mode": "fixed-u",
            "active_ids_cpu": active_ids_cpu.clone(),
            "active_slot_ids_cpu": active_slot_ids_cpu.clone(),
            "active_mask_cpu": active_mask_cpu.clone(),
            "slot_to_global_cpu": slot_to_global.clone(),
            "stage_ids_cpu": current_new_ids.clone(),
            "stage_slot_ids_cpu": current_new_slots.clone(),
            "writeback_ids_cpu": current_leaving_ids.clone(),
            "writeback_slot_ids_cpu": current_leaving_slots.clone(),
            "is_last_step": manifest_step == num_manifest_steps - 1,
            "grad_accum_ids_cpu": grad_accum_ids_cpu.clone(),
            "grad_accum_steps": accum_steps,
            "grad_accum_micro_step": micro_step_index,
            "is_grad_accum_boundary": micro_step_index == accum_steps - 1,
            "sequence_id": int(current_micro_entry.get("sequence_id", -1)),
        }
        if include_local_batch:
            step_meta["inputs_cpu_local"] = cpu_inputs.clone()
            step_meta["targets_cpu_local"] = cpu_targets.clone()
        if inputs_union_cpu_local is not None:
            step_meta["inputs_union_cpu_local"] = inputs_union_cpu_local.clone()
        if targets_union_cpu_local is not None:
            step_meta["targets_union_cpu_local"] = targets_union_cpu_local.clone()

        state_dict = dict(base_state_dict)
        state_dict["manifest_step"] = manifest_step

        if gpu_buffer.data_ptr() != cpu_buffer.data_ptr():
            gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        yield inputs, targets, step_meta, state_dict

        if manifest_step == num_manifest_steps - 1 and is_last_microstep:
            manifest_step += 1
            continue
        if use_manifest_transitions:
            current_new_ids, current_new_slots, _, _ = apply_manifest_transition(current_micro_entry)
        if not is_last_microstep:
            current_micro_idx += 1
            current_micro_entry = current_microsteps[current_micro_idx]
            continue

        manifest_step += 1
        if buffered_next_step_entry is not None:
            current_step_entry = buffered_next_step_entry
            buffered_next_step_entry = None
        else:
            try:
                current_step_entry = resolve_step_entry(manifest_step, next(manifest_iter))
            except StopIteration as exc:
                raise ValueError(
                    f"Sparse manifest ended early after step {manifest_step - 1}; expected {num_manifest_steps} total steps"
                ) from exc
        current_microsteps = get_microsteps(current_step_entry)
        current_micro_idx = 0
        current_micro_entry = current_microsteps[current_micro_idx]
