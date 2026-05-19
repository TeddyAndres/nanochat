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