import math

import torch
import torch.nn.functional as F

from nanochat.sparse_vocab import compute_batch_token_set, sparse_embedding, sparse_logits, local_vocab_log_correction
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


def test_sparse_embedding_matches_dense_rows():
    V, d = 32, 8
    weight = torch.randn(V, d)
    idx = torch.tensor([[1, 2, 3], [4, 5, 1]], dtype=torch.long)
    targets = idx.clone()

    U, _, local_idx, _ = compute_batch_token_set(idx, targets, vocab_size=V)
    out_sparse = sparse_embedding(weight, local_idx, U)
    out_dense = F.embedding(idx, weight)
    assert torch.allclose(out_sparse, out_dense, atol=0, rtol=0)


def test_sparse_logits_and_full_vocab_correction_identity():
    B, T, d, V = 2, 3, 6, 11
    x = torch.randn(B, T, d)
    weight = torch.randn(V, d)
    targets = torch.randint(0, V, (B, T), dtype=torch.long)

    U = torch.arange(V, dtype=torch.long)
    logits_sparse = sparse_logits(x, weight, U)
    logits_dense = x @ weight.T
    assert torch.allclose(logits_sparse, logits_dense, atol=0, rtol=0)

    loss_sparse = F.cross_entropy(logits_sparse.reshape(-1, V), targets.reshape(-1), ignore_index=-1, reduction="mean")
    corr = local_vocab_log_correction(V, V)
    loss_dense = F.cross_entropy(logits_dense.reshape(-1, V), targets.reshape(-1), ignore_index=-1, reduction="mean")

    assert math.isclose(corr, 0.0, abs_tol=1e-12)
    assert torch.allclose(loss_sparse + corr, loss_dense)


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

    idx = torch.randint(0, config.vocab_size, (2, 6), dtype=torch.long)
    targets = torch.randint(0, config.vocab_size, (2, 6), dtype=torch.long)
    targets[:, -1] = -1

    loss = model(idx, targets)
    assert torch.isfinite(loss)

    logits = model(idx)
    assert logits.shape == (2, 6, config.vocab_size)