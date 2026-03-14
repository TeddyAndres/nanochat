"""
Tests for first-pass dynamic vocab runtime.

Run:
python -m pytest tests/test_dynamic_vocab.py -v
"""

import torch
import torch.nn as nn

from nanochat.dynamic_vocab import COLD_LOGIT_BIAS_CLAMP_MAX, DynamicVocabRuntime
from nanochat.gpt import GPT, GPTConfig


def build_tiny_model(vocab_size=8):
    config = GPTConfig(
        sequence_len=4,
        vocab_size=vocab_size,
        n_layer=2,
        n_head=1,
        n_kv_head=1,
        n_embd=32,
        window_pattern="L",
    )
    model = GPT(config)
    model.init_weights()
    return model


def build_fixed_step_meta(
    slot_to_global,
    stage_slots,
    stage_ids,
    writeback_slots,
    writeback_ids,
    is_last_step=False,
    grad_accum_ids=None,
    grad_accum_steps=1,
    grad_accum_micro_step=0,
    is_grad_accum_boundary=True,
):
    slot_to_global = torch.tensor(slot_to_global, dtype=torch.long)
    active_mask = slot_to_global >= 0
    active_slot_ids = torch.nonzero(active_mask, as_tuple=False).flatten()
    active_ids = slot_to_global[active_slot_ids]
    if grad_accum_ids is None:
        grad_accum_ids = active_ids.tolist()
    return {
        "mode": "fixed-u",
        "active_ids_cpu": active_ids,
        "active_slot_ids_cpu": active_slot_ids,
        "active_mask_cpu": active_mask,
        "slot_to_global_cpu": slot_to_global,
        "stage_ids_cpu": torch.tensor(stage_ids, dtype=torch.long),
        "stage_slot_ids_cpu": torch.tensor(stage_slots, dtype=torch.long),
        "writeback_ids_cpu": torch.tensor(writeback_ids, dtype=torch.long),
        "writeback_slot_ids_cpu": torch.tensor(writeback_slots, dtype=torch.long),
        "grad_accum_ids_cpu": torch.tensor(grad_accum_ids, dtype=torch.long),
        "grad_accum_steps": grad_accum_steps,
        "grad_accum_micro_step": grad_accum_micro_step,
        "is_grad_accum_boundary": is_grad_accum_boundary,
        "is_last_step": is_last_step,
    }


def set_zero_sparse_grads(step_ctx):
    step_ctx.active_vocab["wte"].grad = torch.zeros_like(step_ctx.active_vocab["wte"])
    step_ctx.active_vocab["lm_head"].grad = torch.zeros_like(step_ctx.active_vocab["lm_head"])
    for value_embed in step_ctx.active_vocab["value_embeds"].values():
        value_embed.grad = torch.zeros_like(value_embed)


def test_dynamic_vocab_forward_matches_dense_when_u_equals_v():
    torch.manual_seed(0)
    model = build_tiny_model(vocab_size=8)
    runtime = DynamicVocabRuntime(model, device="cpu", embedding_lr=0.01, value_embedding_lr=0.01, unembedding_lr=0.01)
    active_ids = torch.arange(model.config.vocab_size, dtype=torch.long)
    step_ctx = runtime.prepare_step(active_ids)
    idx = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    targets = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    dense_logits = model(idx)
    sparse_logits = model(idx, active_vocab=step_ctx.active_vocab)
    dense_loss = model(idx, targets)
    sparse_loss = model(idx, targets, active_vocab=step_ctx.active_vocab)

    assert torch.allclose(sparse_logits, dense_logits, atol=1e-5, rtol=1e-5)
    assert torch.allclose(sparse_loss, dense_loss, atol=1e-5, rtol=1e-5)


def test_dynamic_vocab_runtime_updates_only_active_rows():
    torch.manual_seed(0)
    model = build_tiny_model(vocab_size=10)
    runtime = DynamicVocabRuntime(model, device="cpu", embedding_lr=0.05, value_embedding_lr=0.04, unembedding_lr=0.03)
    active_ids = torch.tensor([1, 3, 7], dtype=torch.long)
    inactive_id = 0
    wte = model.transformer["wte"]
    assert isinstance(wte, nn.Embedding)
    original_wte_inactive = wte.weight[inactive_id].clone()
    original_wte_active = wte.weight[active_ids].clone()

    step_ctx = runtime.prepare_step(active_ids)
    assert step_ctx.active_vocab is not None
    step_ctx.active_vocab["wte"].grad = torch.ones_like(step_ctx.active_vocab["wte"])
    step_ctx.active_vocab["lm_head"].grad = 2 * torch.ones_like(step_ctx.active_vocab["lm_head"])
    for value_embed in step_ctx.active_vocab["value_embeds"].values():
        value_embed.grad = 3 * torch.ones_like(value_embed)

    runtime.step(step_ctx)

    assert step_ctx.active_vocab is None
    assert step_ctx.optimizer_state is None
    assert not torch.allclose(wte.weight[active_ids], original_wte_active)
    assert torch.allclose(wte.weight[inactive_id], original_wte_inactive)


def test_dynamic_vocab_runtime_state_dict_round_trip():
    torch.manual_seed(0)
    model = build_tiny_model(vocab_size=10)
    runtime = DynamicVocabRuntime(model, device="cpu", embedding_lr=0.05, value_embedding_lr=0.04, unembedding_lr=0.03)
    active_ids = torch.tensor([1, 3, 7], dtype=torch.long)
    step_ctx = runtime.prepare_step(active_ids)
    assert step_ctx.active_vocab is not None
    step_ctx.active_vocab["wte"].grad = torch.ones_like(step_ctx.active_vocab["wte"])
    step_ctx.active_vocab["lm_head"].grad = torch.ones_like(step_ctx.active_vocab["lm_head"])
    for value_embed in step_ctx.active_vocab["value_embeds"].values():
        value_embed.grad = torch.ones_like(value_embed)
    runtime.step(step_ctx)

    saved_state = runtime.state_dict()
    restored = DynamicVocabRuntime(model, device="cpu", embedding_lr=0.05, value_embedding_lr=0.04, unembedding_lr=0.03)
    restored.load_state_dict(saved_state)

    for name, spec in runtime.table_specs.items():
        param = spec["param"]
        restored_param = restored.table_specs[name]["param"]
        runtime_state = runtime.state[param]
        restored_state = restored.state[restored_param]
        assert runtime_state["step"] == restored_state["step"]
        assert torch.allclose(runtime_state["exp_avg"], restored_state["exp_avg"])
        assert torch.allclose(runtime_state["exp_avg_sq"], restored_state["exp_avg_sq"])
    assert runtime.runtime_step == restored.runtime_step
    assert torch.equal(runtime.last_seen_step_cpu, restored.last_seen_step_cpu)


def test_dynamic_vocab_dense_materialization_round_trip():
    torch.manual_seed(0)
    model = build_tiny_model(vocab_size=10)
    runtime = DynamicVocabRuntime(model, device="cpu", embedding_lr=0.05, value_embedding_lr=0.04, unembedding_lr=0.03)
    idx = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)

    wte_before = runtime.table_specs["wte"]["param"].data
    logits_before = None
    with runtime.materialize_dense_params() as dense_model:
        logits_before = dense_model(idx)
        assert runtime.table_specs["wte"]["param"].device.type == "cpu"
        assert logits_before.shape[-1] == model.config.vocab_size

    assert runtime.table_specs["wte"]["param"].data.device.type == "cpu"
    assert runtime.table_specs["wte"]["param"].data.data_ptr() == wte_before.data_ptr()


def test_fixed_u_runtime_delays_common_writeback_until_final_step():
    torch.manual_seed(0)
    model = build_tiny_model(vocab_size=10)
    runtime = DynamicVocabRuntime(
        model,
        device="cpu",
        embedding_lr=0.05,
        value_embedding_lr=0.04,
        unembedding_lr=0.03,
        fixed_u_max=6,
    )
    wte = model.transformer["wte"]
    original_rows = wte.weight[[1, 3, 7]].clone()

    step0 = build_fixed_step_meta(
        slot_to_global=[1, 3, 7, -1, -1, -1],
        stage_slots=[0, 1, 2],
        stage_ids=[1, 3, 7],
        writeback_slots=[0],
        writeback_ids=[1],
        is_last_step=False,
    )
    step0_ctx = runtime.prepare_step(step0)
    active_slots0 = step0_ctx.active_slot_ids_cpu
    assert active_slots0 is not None
    step0_ctx.active_vocab["wte"].grad = torch.zeros_like(step0_ctx.active_vocab["wte"])
    step0_ctx.active_vocab["wte"].grad[active_slots0] = 1
    step0_ctx.active_vocab["lm_head"].grad = torch.zeros_like(step0_ctx.active_vocab["lm_head"])
    step0_ctx.active_vocab["lm_head"].grad[active_slots0] = 2
    for value_embed in step0_ctx.active_vocab["value_embeds"].values():
        value_embed.grad = torch.zeros_like(value_embed)
        value_embed.grad[active_slots0] = 3
    runtime.step(step0_ctx)

    assert not torch.allclose(wte.weight[1], original_rows[0])
    assert torch.allclose(wte.weight[3], original_rows[1])
    assert torch.allclose(wte.weight[7], original_rows[2])

    step1 = build_fixed_step_meta(
        slot_to_global=[-1, 3, 7, -1, -1, -1],
        stage_slots=[],
        stage_ids=[],
        writeback_slots=[],
        writeback_ids=[],
        is_last_step=True,
    )
    step1_ctx = runtime.prepare_step(step1)
    active_slots1 = step1_ctx.active_slot_ids_cpu
    assert active_slots1 is not None
    step1_ctx.active_vocab["wte"].grad = torch.zeros_like(step1_ctx.active_vocab["wte"])
    step1_ctx.active_vocab["wte"].grad[active_slots1] = 1
    step1_ctx.active_vocab["lm_head"].grad = torch.zeros_like(step1_ctx.active_vocab["lm_head"])
    step1_ctx.active_vocab["lm_head"].grad[active_slots1] = 2
    for value_embed in step1_ctx.active_vocab["value_embeds"].values():
        value_embed.grad = torch.zeros_like(value_embed)
        value_embed.grad[active_slots1] = 3
    runtime.step(step1_ctx)

    assert not torch.allclose(wte.weight[3], original_rows[1])
    assert not torch.allclose(wte.weight[7], original_rows[2])


def test_fixed_u_masked_logits_hide_inactive_slots():
    torch.manual_seed(0)
    model = build_tiny_model(vocab_size=8)
    runtime = DynamicVocabRuntime(
        model,
        device="cpu",
        embedding_lr=0.01,
        value_embedding_lr=0.01,
        unembedding_lr=0.01,
        fixed_u_max=6,
    )
    step_meta = build_fixed_step_meta(
        slot_to_global=[0, 1, 2, 3, -1, -1],
        stage_slots=[0, 1, 2, 3],
        stage_ids=[0, 1, 2, 3],
        writeback_slots=[0, 1, 2, 3],
        writeback_ids=[0, 1, 2, 3],
        is_last_step=True,
    )
    step_ctx = runtime.prepare_step(step_meta)
    idx = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    targets = torch.tensor([[1, 2, 3, 0]], dtype=torch.long)

    logits = model(idx, active_vocab=step_ctx.active_vocab)
    scaled_logits = model(idx, active_vocab=step_ctx.active_vocab, logit_scale=0.5)
    loss = model(idx, targets, active_vocab=step_ctx.active_vocab)

    assert torch.isfinite(loss)
    assert torch.all(logits[..., 4:] < -1e8)
    assert torch.all(scaled_logits[..., 4:] < -1e8)


def test_fixed_u_runtime_survives_inference_mode_materialization():
    torch.manual_seed(0)
    model = build_tiny_model(vocab_size=8)
    runtime = DynamicVocabRuntime(
        model,
        device="cpu",
        embedding_lr=0.01,
        value_embedding_lr=0.01,
        unembedding_lr=0.01,
        fixed_u_max=6,
    )

    step_meta = build_fixed_step_meta(
        slot_to_global=[0, 1, 2, 3, -1, -1],
        stage_slots=[0, 1, 2, 3],
        stage_ids=[0, 1, 2, 3],
        writeback_slots=[0, 1, 2, 3],
        writeback_ids=[0, 1, 2, 3],
        is_last_step=False,
    )
    step_ctx = runtime.prepare_step(step_meta)
    active_slots = step_ctx.active_slot_ids_cpu
    assert active_slots is not None
    step_ctx.active_vocab["wte"].grad = torch.zeros_like(step_ctx.active_vocab["wte"])
    step_ctx.active_vocab["wte"].grad[active_slots] = 1
    step_ctx.active_vocab["lm_head"].grad = torch.zeros_like(step_ctx.active_vocab["lm_head"])
    step_ctx.active_vocab["lm_head"].grad[active_slots] = 1
    for value_embed in step_ctx.active_vocab["value_embeds"].values():
        value_embed.grad = torch.zeros_like(value_embed)
        value_embed.grad[active_slots] = 1
    runtime.step(step_ctx)

    with torch.inference_mode():
        with runtime.materialize_dense_params():
            _ = model(torch.tensor([[0, 1, 2, 3]], dtype=torch.long))

    next_ctx = runtime.prepare_step(step_meta)
    next_active_slots = next_ctx.active_slot_ids_cpu
    assert next_active_slots is not None
    next_ctx.active_vocab["wte"].grad = torch.zeros_like(next_ctx.active_vocab["wte"])
    next_ctx.active_vocab["wte"].grad[next_active_slots] = 1
    next_ctx.active_vocab["lm_head"].grad = torch.zeros_like(next_ctx.active_vocab["lm_head"])
    next_ctx.active_vocab["lm_head"].grad[next_active_slots] = 1
    for value_embed in next_ctx.active_vocab["value_embeds"].values():
        value_embed.grad = torch.zeros_like(value_embed)
        value_embed.grad[next_active_slots] = 1
    runtime.step(next_ctx)


def test_sparse_logit_scale_changes_sparse_loss_for_same_targets():
    torch.manual_seed(0)
    model = build_tiny_model(vocab_size=8)
    runtime = DynamicVocabRuntime(
        model,
        device="cpu",
        embedding_lr=0.01,
        value_embedding_lr=0.01,
        unembedding_lr=0.01,
    )
    step_ctx = runtime.prepare_step(torch.arange(model.config.vocab_size, dtype=torch.long))
    idx = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)

    baseline_logits = model(idx, active_vocab=step_ctx.active_vocab)
    targets = baseline_logits.argmax(dim=-1)
    baseline_loss = model(idx, targets, active_vocab=step_ctx.active_vocab)
    scaled_loss = model(idx, targets, active_vocab=step_ctx.active_vocab, logit_scale=0.5)

    assert torch.isfinite(baseline_loss)
    assert torch.isfinite(scaled_loss)
    assert scaled_loss > baseline_loss


def test_sparse_cold_logit_bias_grows_with_absence_steps_and_batch_scale():
    torch.manual_seed(0)

    def build_runtime():
        return DynamicVocabRuntime(
            build_tiny_model(vocab_size=8),
            device="cpu",
            embedding_lr=0.01,
            value_embedding_lr=0.01,
            unembedding_lr=0.01,
            cold_bias_reference_tokens=8,
        )

    runtime_small = build_runtime()
    step0_small = runtime_small.prepare_step(torch.tensor([0], dtype=torch.long), cold_bias_scale=1.0, cold_bias_tokens_per_step=8)
    assert torch.allclose(step0_small.active_vocab["cold_logit_bias"], torch.zeros(1))
    set_zero_sparse_grads(step0_small)
    runtime_small.step(step0_small)

    step1_small = runtime_small.prepare_step(torch.tensor([1], dtype=torch.long), cold_bias_scale=1.0, cold_bias_tokens_per_step=8)
    set_zero_sparse_grads(step1_small)
    runtime_small.step(step1_small)

    revisit_small = runtime_small.prepare_step(torch.tensor([0], dtype=torch.long), cold_bias_scale=1.0, cold_bias_tokens_per_step=8)
    expected_small = torch.log1p(torch.tensor([1.0]))
    assert torch.allclose(revisit_small.active_vocab["cold_logit_bias"], expected_small, atol=1e-6)

    runtime_large = build_runtime()
    step0_large = runtime_large.prepare_step(torch.tensor([0], dtype=torch.long), cold_bias_scale=1.0, cold_bias_tokens_per_step=16)
    set_zero_sparse_grads(step0_large)
    runtime_large.step(step0_large)

    step1_large = runtime_large.prepare_step(torch.tensor([1], dtype=torch.long), cold_bias_scale=1.0, cold_bias_tokens_per_step=16)
    set_zero_sparse_grads(step1_large)
    runtime_large.step(step1_large)

    revisit_large = runtime_large.prepare_step(torch.tensor([0], dtype=torch.long), cold_bias_scale=1.0, cold_bias_tokens_per_step=16)
    expected_large = torch.log1p(torch.tensor([2.0]))
    assert torch.allclose(revisit_large.active_vocab["cold_logit_bias"], expected_large, atol=1e-6)
    assert revisit_large.active_vocab["cold_logit_bias"].item() > revisit_small.active_vocab["cold_logit_bias"].item()


def test_fixed_u_cold_logit_bias_aligns_with_active_slots():
    torch.manual_seed(0)
    model = build_tiny_model(vocab_size=8)
    runtime = DynamicVocabRuntime(
        model,
        device="cpu",
        embedding_lr=0.01,
        value_embedding_lr=0.01,
        unembedding_lr=0.01,
        fixed_u_max=6,
        cold_bias_reference_tokens=8,
    )

    step0 = build_fixed_step_meta(
        slot_to_global=[0, 2, -1, -1, -1, -1],
        stage_slots=[0, 1],
        stage_ids=[0, 2],
        writeback_slots=[0, 1],
        writeback_ids=[0, 2],
        is_last_step=True,
    )
    step0_ctx = runtime.prepare_step(step0, cold_bias_scale=1.0, cold_bias_tokens_per_step=8)
    assert torch.allclose(step0_ctx.active_vocab["cold_logit_bias"][:2], torch.zeros(2))
    set_zero_sparse_grads(step0_ctx)
    runtime.step(step0_ctx)

    warm_other = build_fixed_step_meta(
        slot_to_global=[5, -1, -1, -1, -1, -1],
        stage_slots=[0],
        stage_ids=[5],
        writeback_slots=[0],
        writeback_ids=[5],
        is_last_step=True,
    )
    warm_other_ctx = runtime.prepare_step(warm_other, cold_bias_scale=1.0, cold_bias_tokens_per_step=8)
    set_zero_sparse_grads(warm_other_ctx)
    runtime.step(warm_other_ctx)

    revisit = build_fixed_step_meta(
        slot_to_global=[-1, 0, -1, 2, -1, -1],
        stage_slots=[1, 3],
        stage_ids=[0, 2],
        writeback_slots=[1, 3],
        writeback_ids=[0, 2],
        is_last_step=True,
    )
    revisit_ctx = runtime.prepare_step(revisit, cold_bias_scale=1.0, cold_bias_tokens_per_step=8)
    cold_bias = revisit_ctx.active_vocab["cold_logit_bias"]
    expected = torch.log1p(torch.tensor(1.0))
    assert torch.allclose(cold_bias[[1, 3]], expected.repeat(2), atol=1e-6)
    assert torch.allclose(cold_bias[[0, 2, 4, 5]], torch.zeros(4), atol=1e-6)


def test_sparse_cold_logit_bias_clamps_for_extreme_absence():
    runtime = DynamicVocabRuntime(
        build_tiny_model(vocab_size=8),
        device="cpu",
        embedding_lr=0.01,
        value_embedding_lr=0.01,
        unembedding_lr=0.01,
        cold_bias_reference_tokens=8,
    )

    cold_steps = torch.tensor([0.0, 2.0, 1_000.0], dtype=torch.float32)
    cold_bias = runtime._compute_cold_logit_bias_cpu(
        cold_steps,
        cold_bias_scale=512.0,
        cold_bias_tokens_per_step=8,
    )

    assert cold_bias[0].item() == 0.0
    assert cold_bias[1].item() == COLD_LOGIT_BIAS_CLAMP_MAX
    assert cold_bias[2].item() == COLD_LOGIT_BIAS_CLAMP_MAX


def test_dense_cold_logit_bias_uses_same_clamp_bounds():
    runtime = DynamicVocabRuntime(
        build_tiny_model(vocab_size=8),
        device="cpu",
        embedding_lr=0.01,
        value_embedding_lr=0.01,
        unembedding_lr=0.01,
        cold_bias_reference_tokens=8,
    )
    runtime.runtime_step = 5
    runtime.last_seen_step_cpu.copy_(torch.tensor([4, 2, -996, 4, 4, 4, 4, 4], dtype=torch.float32))

    dense_bias = runtime.get_dense_cold_logit_bias(
        cold_bias_scale=512.0,
        cold_bias_tokens_per_step=8,
    )

    assert dense_bias is not None
    assert dense_bias[0].item() == 0.0
    assert dense_bias[1].item() == COLD_LOGIT_BIAS_CLAMP_MAX
    assert dense_bias[2].item() == COLD_LOGIT_BIAS_CLAMP_MAX


def test_cold_logit_bias_returns_zeros_for_non_positive_scale():
    runtime = DynamicVocabRuntime(
        build_tiny_model(vocab_size=8),
        device="cpu",
        embedding_lr=0.01,
        value_embedding_lr=0.01,
        unembedding_lr=0.01,
        cold_bias_reference_tokens=8,
    )

    cold_steps = torch.tensor([0.0, 2.0, 1_000.0], dtype=torch.float32)
    cold_bias = runtime._compute_cold_logit_bias_cpu(
        cold_steps,
        cold_bias_scale=0.0,
        cold_bias_tokens_per_step=8,
    )

    assert torch.allclose(cold_bias, torch.zeros_like(cold_steps))


def test_fixed_u_sparse_grad_accumulation_matches_single_union_update():
    torch.manual_seed(0)
    model = build_tiny_model(vocab_size=10)
    reference_model = build_tiny_model(vocab_size=10)
    reference_model.load_state_dict(model.state_dict())

    runtime = DynamicVocabRuntime(
        model,
        device="cpu",
        embedding_lr=0.05,
        value_embedding_lr=0.04,
        unembedding_lr=0.03,
        fixed_u_max=3,
        grad_accum_u_max=4,
    )
    reference_runtime = DynamicVocabRuntime(
        reference_model,
        device="cpu",
        embedding_lr=0.05,
        value_embedding_lr=0.04,
        unembedding_lr=0.03,
    )
    union_ids = [1, 3, 7, 9]

    step0 = build_fixed_step_meta(
        slot_to_global=[1, 3, 7],
        stage_slots=[0, 1, 2],
        stage_ids=[1, 3, 7],
        writeback_slots=[0],
        writeback_ids=[1],
        is_last_step=False,
        grad_accum_ids=union_ids,
        grad_accum_steps=2,
        grad_accum_micro_step=0,
        is_grad_accum_boundary=False,
    )
    step0_ctx = runtime.prepare_step(step0)
    slots0 = step0_ctx.active_slot_ids_cpu
    assert slots0 is not None
    step0_ctx.active_vocab["wte"].grad = torch.zeros_like(step0_ctx.active_vocab["wte"])
    step0_ctx.active_vocab["wte"].grad[slots0[0]] = 1
    step0_ctx.active_vocab["wte"].grad[slots0[1]] = 2
    step0_ctx.active_vocab["wte"].grad[slots0[2]] = 3
    step0_ctx.active_vocab["lm_head"].grad = torch.zeros_like(step0_ctx.active_vocab["lm_head"])
    step0_ctx.active_vocab["lm_head"].grad[slots0[0]] = 10
    step0_ctx.active_vocab["lm_head"].grad[slots0[1]] = 20
    step0_ctx.active_vocab["lm_head"].grad[slots0[2]] = 30
    for value_embed in step0_ctx.active_vocab["value_embeds"].values():
        value_embed.grad = torch.zeros_like(value_embed)
        value_embed.grad[slots0[0]] = 100
        value_embed.grad[slots0[1]] = 200
        value_embed.grad[slots0[2]] = 300
    runtime.accumulate_gradients(step0_ctx)

    assert step0_ctx.grad_accum_queue_count == 1
    assert step0_ctx.grad_accum_resident_count == 2
    assert runtime.fixed_params["wte"].grad is not None
    assert torch.allclose(runtime.fixed_params["wte"].grad[slots0[0]], torch.zeros_like(runtime.fixed_params["wte"].grad[slots0[0]]))
    assert torch.allclose(runtime.fixed_params["wte"].grad[slots0[1]], torch.ones_like(runtime.fixed_params["wte"].grad[slots0[1]]) * 2)
    assert torch.allclose(runtime.fixed_params["wte"].grad[slots0[2]], torch.ones_like(runtime.fixed_params["wte"].grad[slots0[2]]) * 3)

    wte_param = runtime.table_specs["wte"]["param"]
    assert runtime.state[wte_param]["step"] == 0

    step1 = build_fixed_step_meta(
        slot_to_global=[9, 3, 7],
        stage_slots=[0],
        stage_ids=[9],
        writeback_slots=[1],
        writeback_ids=[3],
        is_last_step=False,
        grad_accum_ids=union_ids,
        grad_accum_steps=2,
        grad_accum_micro_step=1,
        is_grad_accum_boundary=True,
    )
    step1_ctx = runtime.prepare_step(step1)
    slots1 = step1_ctx.active_slot_ids_cpu
    assert slots1 is not None
    assert step1_ctx.active_vocab["wte"].grad is not None
    step1_ctx.active_vocab["wte"].grad[slots1[0]] += 6
    step1_ctx.active_vocab["wte"].grad[slots1[1]] += 4
    step1_ctx.active_vocab["wte"].grad[slots1[2]] += 5
    assert step1_ctx.active_vocab["lm_head"].grad is not None
    step1_ctx.active_vocab["lm_head"].grad[slots1[0]] += 60
    step1_ctx.active_vocab["lm_head"].grad[slots1[1]] += 40
    step1_ctx.active_vocab["lm_head"].grad[slots1[2]] += 50
    for value_embed in step1_ctx.active_vocab["value_embeds"].values():
        assert value_embed.grad is not None
        value_embed.grad[slots1[0]] += 600
        value_embed.grad[slots1[1]] += 400
        value_embed.grad[slots1[2]] += 500
    runtime.accumulate_gradients(step1_ctx)
    metrics = runtime.apply_accumulated_gradients()
    runtime.flush_active_to_cpu()

    assert metrics.grad_accum_queue_count == 1
    assert metrics.grad_accum_resident_count == 3

    reference_ctx = reference_runtime.prepare_step(torch.tensor(union_ids, dtype=torch.long))
    reference_ctx.active_vocab["wte"].grad = torch.stack([
        torch.ones_like(reference_ctx.active_vocab["wte"][0]) * 1,
        torch.ones_like(reference_ctx.active_vocab["wte"][1]) * 6,
        torch.ones_like(reference_ctx.active_vocab["wte"][2]) * 8,
        torch.ones_like(reference_ctx.active_vocab["wte"][3]) * 6,
    ])
    reference_ctx.active_vocab["lm_head"].grad = torch.stack([
        torch.ones_like(reference_ctx.active_vocab["lm_head"][0]) * 10,
        torch.ones_like(reference_ctx.active_vocab["lm_head"][1]) * 60,
        torch.ones_like(reference_ctx.active_vocab["lm_head"][2]) * 80,
        torch.ones_like(reference_ctx.active_vocab["lm_head"][3]) * 60,
    ])
    for value_embed in reference_ctx.active_vocab["value_embeds"].values():
        value_embed.grad = torch.stack([
            torch.ones_like(value_embed[0]) * 100,
            torch.ones_like(value_embed[1]) * 600,
            torch.ones_like(value_embed[2]) * 800,
            torch.ones_like(value_embed[3]) * 600,
        ])
    reference_runtime.step(reference_ctx)

    union_ids_tensor = torch.tensor(union_ids, dtype=torch.long)
    for name in runtime.table_specs:
        runtime_param = runtime.table_specs[name]["param"]
        reference_param = reference_runtime.table_specs[name]["param"]
        assert torch.allclose(runtime_param[union_ids_tensor], reference_param[union_ids_tensor])
        assert runtime.state[runtime_param]["step"] == reference_runtime.state[reference_param]["step"]


def test_fixed_u_sparse_grad_accumulation_preserves_live_overlap_state():
    torch.manual_seed(0)
    model = build_tiny_model(vocab_size=10)
    runtime = DynamicVocabRuntime(
        model,
        device="cpu",
        embedding_lr=0.05,
        value_embedding_lr=0.04,
        unembedding_lr=0.03,
        fixed_u_max=3,
        grad_accum_u_max=4,
    )

    union_ids = [1, 3, 7, 9]
    step0 = build_fixed_step_meta(
        slot_to_global=[1, 3, 7],
        stage_slots=[0, 1, 2],
        stage_ids=[1, 3, 7],
        writeback_slots=[0],
        writeback_ids=[1],
        is_last_step=False,
        grad_accum_ids=union_ids,
        grad_accum_steps=2,
        grad_accum_micro_step=0,
        is_grad_accum_boundary=False,
    )
    step0_ctx = runtime.prepare_step(step0)
    slots0 = step0_ctx.active_slot_ids_cpu
    assert slots0 is not None
    for name, param in runtime.fixed_params.items():
        param.grad = torch.zeros_like(param)
        param.grad[slots0] = 1
    runtime.accumulate_gradients(step0_ctx)

    assert step0_ctx.grad_accum_queue_count == 1
    assert step0_ctx.grad_accum_resident_count == 2

    step1 = build_fixed_step_meta(
        slot_to_global=[9, 3, 7],
        stage_slots=[0],
        stage_ids=[9],
        writeback_slots=[1],
        writeback_ids=[3],
        is_last_step=False,
        grad_accum_ids=union_ids,
        grad_accum_steps=2,
        grad_accum_micro_step=1,
        is_grad_accum_boundary=True,
    )
    step1_ctx = runtime.prepare_step(step1)
    slots1 = step1_ctx.active_slot_ids_cpu
    assert slots1 is not None
    for name, param in runtime.fixed_params.items():
        assert param.grad is not None
        param.grad[slots1] += 2
    runtime.accumulate_gradients(step1_ctx)
    metrics = runtime.apply_accumulated_gradients()

    assert runtime._fixed_live_state
    assert runtime.fixed_slot_to_global_cpu is not None
    assert torch.equal(runtime.fixed_slot_to_global_cpu[:3], torch.tensor([9, 3, 7], dtype=torch.long))
    assert metrics.live_count == 3
    assert metrics.unique_count == 4
    assert metrics.u_capacity == 4
    assert metrics.grad_accum_queue_count == 1
    assert metrics.grad_accum_resident_count == 3

    next_step = build_fixed_step_meta(
        slot_to_global=[9, 5, 7],
        stage_slots=[1],
        stage_ids=[5],
        writeback_slots=[1],
        writeback_ids=[3],
        is_last_step=False,
        grad_accum_ids=[5, 7, 9],
        grad_accum_steps=1,
        grad_accum_micro_step=0,
        is_grad_accum_boundary=True,
    )
    next_ctx = runtime.prepare_step(next_step)
    assert next_ctx.stage_count == 1
