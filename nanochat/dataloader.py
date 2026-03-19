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

import torch
import pyarrow.parquet as pq

from nanochat.common import get_dist_info
from nanochat.dataset import list_parquet_files
from nanochat.sparse_manifest import load_sparse_manifest_header, stream_sparse_manifest_steps, validate_sparse_manifest

def _document_batches(split, resume_state_dict, tokenizer_batch_size):
    """
    Infinite iterator over document batches (list of text strings) from parquet files.

    Handles DDP sharding and approximate resume. Each yield is (text_batch, (pq_idx, rg_idx, epoch))
    where text_batch is a list of document strings, indices track position for resumption,
    and epoch counts how many times we've cycled through the dataset (starts at 1).
    """
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    warn_on_legacy = ddp_rank == 0 and split == "train" # rank 0 on train split will warn on legacy
    parquet_paths = list_parquet_files(warn_on_legacy=warn_on_legacy)
    assert len(parquet_paths) != 0, "No dataset parquet files found, did you run dataset.py?"
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]

    resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
    resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
    resume_epoch = resume_state_dict.get("epoch", 1) if resume_state_dict is not None else 1
    first_pass = True
    pq_idx = resume_pq_idx
    epoch = resume_epoch

    while True:  # iterate infinitely (multi-epoch)
        pq_idx = resume_pq_idx if first_pass else 0
        while pq_idx < len(parquet_paths):
            filepath = parquet_paths[pq_idx]
            pf = pq.ParquetFile(filepath)
            # Start from resume point if resuming on same file, otherwise from DDP rank
            if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                base_idx = resume_rg_idx // ddp_world_size
                base_idx += 1  # advance by 1 so we don't repeat data after resuming
                rg_idx = base_idx * ddp_world_size + ddp_rank
                if rg_idx >= pf.num_row_groups:
                    pq_idx += 1
                    continue
                resume_rg_idx = None  # only do this once
            else:
                rg_idx = ddp_rank
            while rg_idx < pf.num_row_groups:
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx, epoch)
                rg_idx += ddp_world_size
            pq_idx += 1
        first_pass = False
        epoch += 1


def tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, B, T, split,
    tokenizer_threads=4, tokenizer_batch_size=128,
    device="cuda", resume_state_dict=None,
    buffer_size=1000,
    return_active_vocab=False,
    vocab_size=None,
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
    batches = _document_batches(split, resume_state_dict, tokenizer_batch_size)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    pq_idx, rg_idx, epoch = 0, 0, 1

    def refill_buffer():
        nonlocal pq_idx, rg_idx, epoch
        doc_batch, (pq_idx, rg_idx, epoch) = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
        for tokens in token_lists:
            doc_buffer.append(tokens)

    # Pre-allocate buffers once: layout is [inputs (B*T) | targets (B*T)]
    # This gives us contiguous views and a single HtoD transfer
    use_cuda = torch.device(device).type == "cuda"
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long) # for building rows without creating Python lists
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda) # staging area (CPU)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device) # on-device buffer
    cpu_inputs = cpu_buffer[:B * T].view(B, T) # a few views into these buffers just for convenience
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)
    if return_active_vocab:
        assert vocab_size is not None, "vocab_size is required when return_active_vocab=True"
        global_to_local = torch.full((vocab_size,), -1, dtype=torch.long)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                # Ensure buffer has documents
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # Find largest doc that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    doc_len = len(doc)
                    row_buffer[row_idx, pos:pos + doc_len] = torch.tensor(doc, dtype=torch.long)
                    pos += doc_len
                else:
                    # No doc fits - crop shortest in buffer to fill remaining and minimize waste
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

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

        # Single HtoD copy into persistent GPU buffer and yield
        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        if return_active_vocab:
            yield inputs, targets, active_ids, state_dict
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
    vocab_size=None,
    include_local_batch=False,
):
    """Manifest-driven sparse loader with fixed logical U slots.

    The live BOS best-fit batch construction stays unchanged, but vocab remapping and
    overlap transitions are driven by a precomputed sparse-manifest JSON.
    """
    assert vocab_size is not None, "vocab_size is required for manifest sparse loader"
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    manifest = load_sparse_manifest_header(manifest_path)
    manifest_version = int(manifest.get("version", 1))
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
    )

    use_cuda = torch.device(device).type == "cuda"
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device)
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    slot_to_global = torch.full((u_max,), -1, dtype=torch.long)
    global_to_slot = torch.full((vocab_size,), -1, dtype=torch.long)

    def apply_next_transition(step_entry):
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
            return next_new_ids, next_new_ids
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
        return next_new_ids, assigned_slots_tensor

    manifest_iter = stream_sparse_manifest_steps(manifest_path)
    try:
        current_step_entry = next(manifest_iter)
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

    first_active_ids = torch.tensor(current_micro_entry["active_ids"], dtype=torch.long)
    if first_active_ids.numel() > u_max:
        raise ValueError(
            f"Sparse manifest first step exceeds U_max: {first_active_ids.numel()} > {u_max}"
        )
    first_slots = torch.arange(first_active_ids.numel(), dtype=torch.long)
    slot_to_global[first_slots] = first_active_ids
    global_to_slot[first_active_ids] = first_slots
    current_new_ids = first_active_ids.clone()
    current_new_slots = first_slots.clone()
    for replay_step in range(manifest_step):
        for replay_micro_entry in current_microsteps:
            current_new_ids, current_new_slots = apply_next_transition(replay_micro_entry)
        try:
            current_step_entry = next(manifest_iter)
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
        current_leaving_ids = torch.tensor(current_micro_entry["next_leaving_ids"], dtype=torch.long)
        if current_leaving_ids.numel() > 0:
            current_leaving_slots = global_to_slot[current_leaving_ids]
            if (current_leaving_slots < 0).any():
                raise ValueError("Sparse manifest leaving ids are not present in the current slot map")
        else:
            current_leaving_slots = torch.empty(0, dtype=torch.long)

        grad_accum_ids_cpu = active_ids_cpu.clone()
        accum_steps = 1
        micro_step_index = 0
        if manifest_version >= 2:
            grad_accum_ids_cpu = torch.tensor(current_step_entry["grad_accum_active_ids"], dtype=torch.long)
            accum_steps = len(current_microsteps)
            micro_step_index = current_micro_idx

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
        }
        if include_local_batch:
            step_meta["inputs_cpu_local"] = cpu_inputs.clone()
            step_meta["targets_cpu_local"] = cpu_targets.clone()

        state_dict = dict(base_state_dict)
        state_dict["manifest_step"] = manifest_step

        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        yield inputs, targets, step_meta, state_dict

        is_last_microstep = current_micro_idx == len(current_microsteps) - 1
        if manifest_step == num_manifest_steps - 1 and is_last_microstep:
            manifest_step += 1
            continue

        current_new_ids, current_new_slots = apply_next_transition(current_micro_entry)
        if not is_last_microstep:
            current_micro_idx += 1
            current_micro_entry = current_microsteps[current_micro_idx]
            continue

        manifest_step += 1
        try:
            current_step_entry = next(manifest_iter)
        except StopIteration as exc:
            raise ValueError(
                f"Sparse manifest ended early after step {manifest_step - 1}; expected {num_manifest_steps} total steps"
            ) from exc
        current_microsteps = get_microsteps(current_step_entry)
        current_micro_idx = 0
        current_micro_entry = current_microsteps[current_micro_idx]
