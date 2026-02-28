import math

import torch
import torch.nn.functional as F

from nanochat.sparse_vocab import compute_batch_token_set, local_vocab_log_correction
from nanochat.gpt import GPT, GPTConfig


def test_compute_batch_token_set_roundtrip_and_targets():
    idx = torch.tensor([[1, 2, 3], [3, 4, 5]], dtype=torch.long)
    targets = torch.tensor([[2, 3, -1], [4, 5, 6]], dtype=torch.long)
    U, global_to_local, local_idx, local_targets = compute_batch_token_set(idx, targets, vocab_size=16)

    assert torch.equal(U[local_idx], idx)
    valid = targets >= 0
    assert torch.equal(U[local_targets[valid]], targets[valid])
    assert (local_targets[~valid] == -1).all()
    assert (global_to_local[U] >= 0).all()


def test_local_vocab_log_correction_full_vocab_is_zero():
    """When the local vocab equals the full vocab, the log correction is 0."""
    V = 32768
    corr = local_vocab_log_correction(V, V)
    assert math.isclose(corr, 0.0, abs_tol=1e-12)


def test_local_vocab_log_correction_partial_vocab():
    """Correction < 0 when local_vocab_size < vocab_size (makes loss lower, as expected)."""
    V, u = 32768, 8192
    corr = local_vocab_log_correction(V, u)
    assert corr == math.log(V / u)
    assert corr > 0  # positive correction shifts loss upward


def _build_sparse_context(model, idx, targets, device):
    """Build a minimal sparse_context dict with pre-fetched rows for a test model."""
    V = model.config.vocab_size
    U, _, local_idx, local_targets = compute_batch_token_set(idx, targets, vocab_size=V)
    U_size = U.numel()

    def _rows(w):
        rows = w.index_select(0, U)
        rows = rows.to(device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
        rows.requires_grad_(True)
        return rows

    W_U_wte = _rows(model.wte().weight)
    W_U_lm  = W_U_wte if model.config.tie_embeddings else _rows(model.lm_head.weight)
    W_U_ve  = {}
    for i_str, ve_mod in model.value_embeds.items():
        W_U_ve[i_str] = _rows(ve_mod.weight)

    return {
        "W_U_wte":        W_U_wte,
        "W_U_ve":         W_U_ve,
        "W_U_lm_head":    W_U_lm,
        "local_idx":      local_idx.to(device),
        "local_targets":  local_targets.to(device),
        "log_correction": torch.tensor(math.log(V) - math.log(U_size), dtype=torch.float32),
    }


def test_gpt_sparse_train_and_dense_inference_paths():
    config = GPTConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
        window_pattern="L",
        sparse_mode=True,
        sparse_ddp_union=False,
        tie_embeddings=True,
    )
    model = GPT(config, pad_vocab_size_to=1)
    model.init_weights()
    device = torch.device("cpu")

    idx = torch.randint(0, config.vocab_size, (2, 6), dtype=torch.long)
    targets = torch.randint(0, config.vocab_size, (2, 6), dtype=torch.long)
    targets[:, -1] = -1

    sparse_ctx = _build_sparse_context(model, idx, targets, device)
    loss = model(idx, targets, sparse_context=sparse_ctx)
    assert torch.isfinite(loss)

    # Dense inference path (no targets, no sparse_context)
    logits = model(idx)
    assert logits.shape == (2, 6, config.vocab_size)


def test_gpt_sparse_loss_finite_and_backward():
    """Verify loss is finite and gradients flow to W_U_wte in sparse mode."""
    config = GPTConfig(
        sequence_len=8,
        vocab_size=64,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        window_pattern="L",
        sparse_mode=True,
        sparse_ddp_union=False,
        tie_embeddings=True,
    )
    model = GPT(config, pad_vocab_size_to=1)
    model.init_weights()
    device = torch.device("cpu")

    idx = torch.randint(0, 20, (2, 4), dtype=torch.long)  # only 20 distinct tokens used
    targets = idx.roll(-1, dims=1)
    targets[:, -1] = -1

    sparse_ctx = _build_sparse_context(model, idx, targets, device)
    loss = model(idx, targets, sparse_context=sparse_ctx)
    assert torch.isfinite(loss), f"loss is not finite: {loss}"
    loss.backward()

    W_U_wte = sparse_ctx["W_U_wte"]
    assert W_U_wte.grad is not None, "W_U_wte should have received a gradient"
    assert torch.isfinite(W_U_wte.grad).all(), "W_U_wte gradient has non-finite values"
