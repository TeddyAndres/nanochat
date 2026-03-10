"""
Tests for first-pass dynamic vocab runtime.

Run:
python -m pytest tests/test_dynamic_vocab.py -v
"""

import math

import torch
import torch.nn as nn

from nanochat.dynamic_vocab import DynamicVocabRuntime
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
    loss = model(idx, targets, active_vocab=step_ctx.active_vocab)

    assert torch.isfinite(loss)
    assert torch.all(logits[..., 4:] < -1e8)


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


def test_dynamic_vocab_sampled_cold_negatives_extend_logits_and_update_rows():
    torch.manual_seed(0)
    model = build_tiny_model(vocab_size=8)
    for param in model.parameters():
        param.data.zero_()
    runtime = DynamicVocabRuntime(
        model,
        device="cpu",
        embedding_lr=0.01,
        value_embedding_lr=0.01,
        unembedding_lr=0.01,
        sampled_negative_count=2,
    )
    active_ids = torch.tensor([0, 2, 4], dtype=torch.long)
    step_ctx = runtime.prepare_step(active_ids)

    sampled_ids = step_ctx.sampled_negative_ids_cpu
    assert sampled_ids is not None
    assert sampled_ids.numel() == 2
    assert not torch.isin(sampled_ids, active_ids).any()
    assert step_ctx.active_vocab is not None
    assert "lm_head_negatives" in step_ctx.active_vocab
    expected_bias = math.log((model.config.vocab_size - active_ids.numel()) / sampled_ids.numel())
    assert math.isclose(step_ctx.active_vocab["lm_head_negative_logit_bias"].item(), expected_bias, rel_tol=0.0, abs_tol=1e-6)

    idx = torch.tensor([[0, 1, 2, 0]], dtype=torch.long)
    targets = torch.tensor([[1, 2, 0, 1]], dtype=torch.long)
    logits = model(idx, active_vocab=step_ctx.active_vocab)
    assert logits.shape[-1] == active_ids.numel() + sampled_ids.numel()
    expected_negative_logit = 20.0 * math.tanh(expected_bias / 20.0)
    assert torch.allclose(logits[..., :active_ids.numel()], torch.zeros_like(logits[..., :active_ids.numel()]), atol=2e-3, rtol=0.0)
    assert torch.allclose(logits[..., -sampled_ids.numel():], torch.full_like(logits[..., -sampled_ids.numel():], expected_negative_logit), atol=2e-3, rtol=0.0)

    lm_head_param = runtime.table_specs["lm_head"]["param"]
    original_sampled_rows = lm_head_param[sampled_ids].clone()
    step_ctx.active_vocab["wte"].grad = torch.zeros_like(step_ctx.active_vocab["wte"])
    step_ctx.active_vocab["lm_head"].grad = torch.ones_like(step_ctx.active_vocab["lm_head"])
    step_ctx.active_vocab["lm_head_negatives"].grad = torch.ones_like(step_ctx.active_vocab["lm_head_negatives"])
    for value_embed in step_ctx.active_vocab["value_embeds"].values():
        value_embed.grad = torch.zeros_like(value_embed)

    runtime.step(step_ctx)

    assert runtime.state[lm_head_param]["step"] == 1
    assert not torch.allclose(lm_head_param[sampled_ids], original_sampled_rows)


def test_fixed_u_sampled_cold_negatives_append_after_masked_slots():
    torch.manual_seed(0)
    model = build_tiny_model(vocab_size=8)
    for param in model.parameters():
        param.data.zero_()
    runtime = DynamicVocabRuntime(
        model,
        device="cpu",
        embedding_lr=0.01,
        value_embedding_lr=0.01,
        unembedding_lr=0.01,
        fixed_u_max=6,
        sampled_negative_count=2,
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
    sampled_ids = step_ctx.sampled_negative_ids_cpu
    assert sampled_ids is not None
    assert sampled_ids.numel() == 2
    expected_bias = math.log((model.config.vocab_size - step_ctx.active_ids_cpu.numel()) / sampled_ids.numel())
    assert math.isclose(step_ctx.active_vocab["lm_head_negative_logit_bias"].item(), expected_bias, rel_tol=0.0, abs_tol=1e-6)

    idx = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    logits = model(idx, active_vocab=step_ctx.active_vocab)

    assert logits.shape[-1] == 8
    assert torch.all(logits[..., 4:6] < -1e8)
    expected_negative_logit = 20.0 * math.tanh(expected_bias / 20.0)
    assert torch.allclose(logits[..., 6:], torch.full_like(logits[..., 6:], expected_negative_logit), atol=2e-3, rtol=0.0)


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
        fixed_u_max=6,
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
        slot_to_global=[1, 3, 7, -1, -1, -1],
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

    wte_param = runtime.table_specs["wte"]["param"]
    assert runtime.state[wte_param]["step"] == 0

    step1 = build_fixed_step_meta(
        slot_to_global=[3, 7, 9, -1, -1, -1],
        stage_slots=[2],
        stage_ids=[9],
        writeback_slots=[0],
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
    step1_ctx.active_vocab["wte"].grad = torch.zeros_like(step1_ctx.active_vocab["wte"])
    step1_ctx.active_vocab["wte"].grad[slots1[0]] = 4
    step1_ctx.active_vocab["wte"].grad[slots1[1]] = 5
    step1_ctx.active_vocab["wte"].grad[slots1[2]] = 6
    step1_ctx.active_vocab["lm_head"].grad = torch.zeros_like(step1_ctx.active_vocab["lm_head"])
    step1_ctx.active_vocab["lm_head"].grad[slots1[0]] = 40
    step1_ctx.active_vocab["lm_head"].grad[slots1[1]] = 50
    step1_ctx.active_vocab["lm_head"].grad[slots1[2]] = 60
    for value_embed in step1_ctx.active_vocab["value_embeds"].values():
        value_embed.grad = torch.zeros_like(value_embed)
        value_embed.grad[slots1[0]] = 400
        value_embed.grad[slots1[1]] = 500
        value_embed.grad[slots1[2]] = 600
    runtime.accumulate_gradients(step1_ctx)
    runtime.apply_accumulated_gradients()

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
        fixed_u_max=6,
    )

    union_ids = [1, 3, 7, 9]
    step0 = build_fixed_step_meta(
        slot_to_global=[1, 3, 7, -1, -1, -1],
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

    step1 = build_fixed_step_meta(
        slot_to_global=[3, 7, 9, -1, -1, -1],
        stage_slots=[2],
        stage_ids=[9],
        writeback_slots=[0],
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
        param.grad = torch.zeros_like(param)
        param.grad[slots1] = 2
    runtime.accumulate_gradients(step1_ctx)
    metrics = runtime.apply_accumulated_gradients()

    assert runtime._fixed_live_state
    assert runtime.fixed_slot_to_global_cpu is not None
    assert torch.equal(runtime.fixed_slot_to_global_cpu[:3], torch.tensor([3, 7, 9], dtype=torch.long))
    assert metrics.live_count == 3
    assert metrics.unique_count == 4

    next_step = build_fixed_step_meta(
        slot_to_global=[7, 9, 5, -1, -1, -1],
        stage_slots=[2],
        stage_ids=[5],
        writeback_slots=[0],
        writeback_ids=[3],
        is_last_step=False,
        grad_accum_ids=[5, 7, 9],
        grad_accum_steps=1,
        grad_accum_micro_step=0,
        is_grad_accum_boundary=True,
    )
    next_ctx = runtime.prepare_step(next_step)
    assert next_ctx.stage_count == 1
