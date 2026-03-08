"""
Tests for first-pass dynamic vocab runtime.

Run:
python -m pytest tests/test_dynamic_vocab.py -v
"""

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
    step_ctx.active_vocab["wte"].grad = torch.ones_like(step_ctx.active_vocab["wte"])
    step_ctx.active_vocab["lm_head"].grad = 2 * torch.ones_like(step_ctx.active_vocab["lm_head"])
    for value_embed in step_ctx.active_vocab["value_embeds"].values():
        value_embed.grad = 3 * torch.ones_like(value_embed)

    runtime.step(step_ctx)

    assert step_ctx.bytes_h2d > 0
    assert step_ctx.bytes_d2h > 0
    assert step_ctx.h2d_ms >= 0.0
    assert step_ctx.d2h_ms >= 0.0
    assert not torch.allclose(wte.weight[active_ids], original_wte_active)
    assert torch.allclose(wte.weight[inactive_id], original_wte_inactive)


def test_dynamic_vocab_runtime_state_dict_round_trip():
    torch.manual_seed(0)
    model = build_tiny_model(vocab_size=10)
    runtime = DynamicVocabRuntime(model, device="cpu", embedding_lr=0.05, value_embedding_lr=0.04, unembedding_lr=0.03)
    active_ids = torch.tensor([1, 3, 7], dtype=torch.long)
    step_ctx = runtime.prepare_step(active_ids)
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
