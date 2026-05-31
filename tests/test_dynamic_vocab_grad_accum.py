import torch

from nanochat.dynamic_vocab import DynamicVocabRuntime
from nanochat.gpt import GPT, GPTConfig


def _build_runtime(base_state_dict, *, disable_overlap_reuse: bool) -> DynamicVocabRuntime:
    config = GPTConfig(
        sequence_len=2,
        vocab_size=16,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=8,
        window_pattern="L",
    )
    model = GPT(config)
    model.load_state_dict(base_state_dict)
    runtime = DynamicVocabRuntime(
        model,
        device="cpu",
        embedding_lr=0.05,
        value_embedding_lr=0.05,
        unembedding_lr=0.05,
        fixed_u_max=4,
        lm_head_u_max=4,
        grad_accum_u_max=4,
        adam_betas=(0.8, 0.95),
        weight_decay=0.0,
    )
    runtime.disable_fixed_overlap_reuse = disable_overlap_reuse
    return runtime


def _step_meta(window_ids: list[int], inputs_local: list[list[int]], targets_local: list[list[int]], micro_step: int, grad_accum_steps: int) -> dict:
    active_ids_cpu = torch.tensor(window_ids, dtype=torch.long)
    active_slot_ids_cpu = torch.arange(len(window_ids), dtype=torch.long)
    return {
        "active_ids_cpu": active_ids_cpu,
        "active_slot_ids_cpu": active_slot_ids_cpu,
        "active_mask_cpu": torch.ones(len(window_ids), dtype=torch.bool),
        "slot_to_global_cpu": active_ids_cpu.clone(),
        "stage_ids_cpu": active_ids_cpu.clone(),
        "stage_slot_ids_cpu": active_slot_ids_cpu.clone(),
        "writeback_ids_cpu": torch.empty(0, dtype=torch.long),
        "writeback_slot_ids_cpu": torch.empty(0, dtype=torch.long),
        "grad_accum_ids_cpu": active_ids_cpu.clone(),
        "grad_accum_steps": grad_accum_steps,
        "grad_accum_micro_step": micro_step,
        "is_grad_accum_boundary": micro_step == 0,
        "is_last_step": False,
        "inputs_union_cpu_local": torch.tensor(inputs_local, dtype=torch.long),
        "targets_union_cpu_local": torch.tensor(targets_local, dtype=torch.long),
    }


def _run_window(runtime: DynamicVocabRuntime, window_ids: list[int], micro_batches: list[tuple[list[list[int]], list[list[int]]]]) -> None:
    for micro_step, (inputs_local, targets_local) in enumerate(micro_batches):
        step_ctx = runtime.prepare_step(
            _step_meta(
                window_ids,
                inputs_local,
                targets_local,
                micro_step=micro_step,
                grad_accum_steps=len(micro_batches),
            )
        )
        assert step_ctx.union_inputs is not None
        assert step_ctx.union_targets is not None
        loss = runtime.model(
            step_ctx.union_inputs,
            step_ctx.union_targets,
            active_vocab=step_ctx.active_vocab,
        )
        loss.backward()
        runtime.accumulate_gradients(step_ctx)
    runtime.apply_accumulated_gradients()
    runtime.model.zero_grad(set_to_none=True)


def test_grad_accum_overlap_reuse_matches_no_reuse_across_window_transition():
    torch.manual_seed(0)
    base_model = GPT(
        GPTConfig(
            sequence_len=2,
            vocab_size=16,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=8,
            window_pattern="L",
        )
    )
    base_model.init_weights()
    base_state_dict = base_model.state_dict()

    runtime_with_reuse = _build_runtime(base_state_dict, disable_overlap_reuse=False)
    runtime_without_reuse = _build_runtime(base_state_dict, disable_overlap_reuse=True)

    window_0 = [0, 1, 2, 3]
    window_1 = [2, 3, 4, 5]
    micro_batches = [
        ([[0, 1]], [[1, 2]]),
        ([[2, 0]], [[3, 1]]),
    ]

    for runtime in (runtime_with_reuse, runtime_without_reuse):
        _run_window(runtime, window_0, micro_batches)
        _run_window(runtime, window_1, micro_batches)

    touched_ids = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)
    for table_name, spec_with_reuse in runtime_with_reuse.table_specs.items():
        spec_without_reuse = runtime_without_reuse.table_specs[table_name]
        torch.testing.assert_close(
            spec_with_reuse["param"].index_select(0, touched_ids),
            spec_without_reuse["param"].index_select(0, touched_ids),
            atol=0,
            rtol=0,
        )
        state_with_reuse = runtime_with_reuse.state[spec_with_reuse["param"]]
        state_without_reuse = runtime_without_reuse.state[spec_without_reuse["param"]]
        torch.testing.assert_close(
            state_with_reuse["exp_avg"].index_select(0, touched_ids),
            state_without_reuse["exp_avg"].index_select(0, touched_ids),
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            state_with_reuse["exp_avg_sq"].index_select(0, touched_ids),
            state_without_reuse["exp_avg_sq"].index_select(0, touched_ids),
            atol=0,
            rtol=0,
        )

    torch.testing.assert_close(runtime_with_reuse.last_seen_step_cpu[touched_ids], runtime_without_reuse.last_seen_step_cpu[touched_ids], atol=0, rtol=0)
    torch.testing.assert_close(runtime_with_reuse.hot_activation_count_cpu[touched_ids], runtime_without_reuse.hot_activation_count_cpu[touched_ids], atol=0, rtol=0)
    torch.testing.assert_close(runtime_with_reuse.token_event_step_count_cpu[touched_ids], runtime_without_reuse.token_event_step_count_cpu[touched_ids], atol=0, rtol=0)


def test_persistent_slot_compute_produces_consistent_transitions():
    """Isolated test for the persistent slot decision logic (Phase 2 foundation).

    Exercises _compute_persistent_slot_state_cpu (and the for_meta wrapper)
    with a controlled initial map state and a realistic arriving/leaving
    grad-accum window. Verifies that the returned decisions are deterministic
    and that the side effects on the live maps are exactly the expected
    mutations for the current (pre-refactor) implementation.

    After the pure-compute refactor this test will be updated to assert on the
    returned PersistentSlotTransition object with zero map mutations from compute.
    """
    torch.manual_seed(123)

    base_model = GPT(
        GPTConfig(
            sequence_len=2,
            vocab_size=16,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=8,
            window_pattern="L",
        )
    )
    base_model.init_weights()
    base_state_dict = base_model.state_dict()

    runtime = _build_runtime(base_state_dict, disable_overlap_reuse=False)
    # Force fixed-U mode with small numbers for easy inspection
    assert runtime.fixed_u_mode
    assert runtime.fixed_input_u_max >= 4

    # Seed a realistic "previous window" state on the maps
    # Previous window had tokens [0,1,2,3] in slots 0,1,2,3
    prev_ids = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    prev_slots = torch.arange(4, dtype=torch.long)

    runtime._fixed_input_global_to_slot_cpu.fill_(-1)
    runtime._fixed_input_global_to_slot_cpu[prev_ids] = prev_slots
    runtime.fixed_input_slot_to_global_cpu.fill_(-1)
    runtime.fixed_input_slot_to_global_cpu[prev_slots] = prev_ids

    runtime.fixed_lm_head_slot_to_global_cpu.fill_(-1)
    runtime.fixed_lm_head_slot_to_global_cpu[prev_slots] = prev_ids

    runtime._grad_accum_wte_local_to_slot_cpu.fill_(-1)
    runtime._fixed_live_state = True
    runtime._grad_accum_union_slot_remap_active = False

    # New window: [2,3,4,5]  -> 2,3 survive; 0,1 leave; 4,5 arrive
    new_grad_accum = torch.tensor([2, 3, 4, 5], dtype=torch.long)

    # Call the low-level compute directly (now pure / side-effect-free)
    result = runtime._compute_persistent_slot_state_cpu(
        grad_accum_ids_cpu=new_grad_accum,
        preserve_resident_grads=False,
        union_input_tables=True,
        stage_ids_cpu=new_grad_accum,
        stage_slot_ids_cpu=torch.arange(4, dtype=torch.long),
    )

    # Basic shape / presence checks (supports dataclass or legacy dict during transition)
    if hasattr(result, "used_persistent_input_slots"):
        assert result.used_persistent_input_slots is True
        assert hasattr(result, "deferred_writeback_ids_cpu")
        assert hasattr(result, "input_stage_ids_cpu")
    else:
        assert result.get("_used_persistent_input_slots") is True
        assert "deferred_writeback_ids_cpu" in result
        assert "input_stage_ids_cpu" in result

    # Apply the transition (this is the main-thread step that mutates the maps).
    # This is the new required pattern after the Phase 2 refactor.
    runtime._apply_persistent_slot_transition(result)

    # After apply, the invariants must hold:
    # - No slot is double-booked
    # - Every active grad_accum id has a valid slot in the inverse map
    post_global_to_slot = runtime._fixed_input_global_to_slot_cpu
    active_slots = post_global_to_slot[new_grad_accum]
    assert (active_slots >= 0).all(), "All new grad-accum ids must have valid slots after transition"
    # The forward map must be consistent for the assigned slots
    for gid, slot in zip(new_grad_accum.tolist(), active_slots.tolist()):
        assert runtime.fixed_input_slot_to_global_cpu[slot] == gid

    # The local_to_slot for the union remap must have been populated for all 4 positions
    local_map = runtime._grad_accum_wte_local_to_slot_cpu[:4]
    assert (local_map >= 0).all()

    # Also exercise the for_meta wrapper (the one submitted to prefetch)
    meta = {
        "active_ids_cpu": new_grad_accum,   # fallback used by for_meta
        "grad_accum_ids_cpu": new_grad_accum,
        "preserve_resident_grads": False,
        "union_input_tables": True,
        "stage_ids_cpu": new_grad_accum,
        "stage_slot_ids_cpu": torch.arange(4, dtype=torch.long),
    }
    result2 = runtime._compute_persistent_slot_state_for_meta(meta)
    if hasattr(result2, "used_persistent_input_slots"):
        assert result2.used_persistent_input_slots is True
    else:
        assert result2.get("_used_persistent_input_slots") is True

    # The two paths should have produced structured data without crashing.
    # (The second call sees post-apply state from the first; we do not re-apply here.)
    has_ms = hasattr(result, "prep_persistent_slot_ms") or (isinstance(result, dict) and "prep_persistent_slot_ms" in result)
    assert has_ms
    has_ms2 = hasattr(result2, "prep_persistent_slot_ms") or (isinstance(result2, dict) and "prep_persistent_slot_ms" in result2)
    assert has_ms2