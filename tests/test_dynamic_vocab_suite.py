import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.sparse_optim import SparseHybridOptimizer
from nanochat.sparse_vocab import compute_batch_token_set, sparse_logits


def _build_model(*, sparse_mode: bool, tie_embeddings: bool, vocab_size: int = 256):
    config = GPTConfig(
        sequence_len=32,
        vocab_size=vocab_size,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
        window_pattern="L",
        sparse_mode=sparse_mode,
        sparse_ddp_union=False,
        tie_embeddings=tie_embeddings,
    )
    model = GPT(config, pad_vocab_size_to=1)
    model.init_weights()
    return model


def _sample_batch(vocab_size: int, batch_size: int = 2, seq_len: int = 12):
    idx = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    targets = idx.roll(shifts=-1, dims=1)
    targets[:, -1] = -1
    return idx, targets


def test_standard_model_train_and_infer_still_work():
    model = _build_model(sparse_mode=False, tie_embeddings=False, vocab_size=128)
    optimizer = model.setup_optimizer()

    idx, targets = _sample_batch(vocab_size=128)
    loss = model(idx, targets)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    logits = model(idx)
    assert logits.shape == (idx.size(0), idx.size(1), 128)
    assert torch.isfinite(logits).all()


def test_sparse_mode_train_and_infer_work_with_sparse_optimizer():
    model = _build_model(sparse_mode=True, tie_embeddings=True, vocab_size=256)
    optimizer = model.setup_optimizer()
    assert isinstance(optimizer, SparseHybridOptimizer)

    idx, targets = _sample_batch(vocab_size=256)
    loss = model(idx, targets)
    assert torch.isfinite(loss)
    loss.backward()

    grad = model.wte().weight.grad
    assert grad is not None
    assert grad.is_sparse

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    logits = model(idx)
    assert logits.shape == (idx.size(0), idx.size(1), 256)
    assert torch.isfinite(logits).all()


def test_inference_is_non_destructive_under_sparse_toggle():
    dense_model = _build_model(sparse_mode=False, tie_embeddings=True, vocab_size=96)
    sparse_model = _build_model(sparse_mode=True, tie_embeddings=True, vocab_size=96)
    sparse_model.load_state_dict(dense_model.state_dict(), strict=True)

    idx, _ = _sample_batch(vocab_size=96)
    with torch.no_grad():
        dense_logits = dense_model(idx)
        sparse_logits_infer = sparse_model(idx)

    assert torch.allclose(dense_logits, sparse_logits_infer, atol=1e-5, rtol=1e-5)


def test_sparse_and_dense_match_when_local_vocab_is_full():
    dense_model = _build_model(sparse_mode=False, tie_embeddings=True, vocab_size=32)
    sparse_model = _build_model(sparse_mode=True, tie_embeddings=True, vocab_size=32)
    sparse_model.load_state_dict(dense_model.state_dict(), strict=True)

    idx = torch.arange(0, 32, dtype=torch.long).view(1, 32)
    targets = idx.roll(shifts=-1, dims=1)
    targets[:, -1] = 0

    dense_loss = dense_model(idx, targets)
    sparse_loss = sparse_model(idx, targets)
    assert torch.allclose(dense_loss, sparse_loss, atol=1e-4, rtol=1e-4)


def test_sparse_local_vocab_reduces_logit_tensor_size():
    V = 1000
    d = 64
    B, T = 2, 16
    x = torch.randn(B, T, d)
    weight = torch.randn(V, d)

    active_tokens = torch.randint(0, 20, (B, T), dtype=torch.long)
    targets = active_tokens.roll(shifts=-1, dims=1)
    targets[:, -1] = -1

    U, _, _, _ = compute_batch_token_set(active_tokens, targets, vocab_size=V)
    logits_dense = x @ weight.T
    logits_local = sparse_logits(x, weight, U)

    dense_numel = logits_dense.numel()
    local_numel = logits_local.numel()
    reduction = dense_numel / local_numel

    assert U.numel() < V
    assert local_numel < dense_numel
    assert reduction > 5.0


def test_ddp_union_flag_is_safe_without_dist_init():
    idx = torch.tensor([[1, 2, 2, 3], [3, 4, 4, 5]], dtype=torch.long)
    targets = idx.roll(shifts=-1, dims=1)
    targets[:, -1] = -1

    U_plain, _, _, _ = compute_batch_token_set(idx, targets, vocab_size=64, use_ddp_union=False)
    U_union, _, _, _ = compute_batch_token_set(idx, targets, vocab_size=64, use_ddp_union=True)
    assert torch.equal(U_plain, U_union)