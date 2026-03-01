import math
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.sparse_optim import VocabRowAdamW
from nanochat.sparse_vocab import compute_batch_token_set


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


def _build_sparse_context(model, idx, targets, device=None):
    """Simulate the pre-fetch that base_train does before the grad-accum loop."""
    if device is None:
        device = torch.device("cpu")
    V = model.config.vocab_size
    U, _, local_idx, local_targets = compute_batch_token_set(idx, targets, vocab_size=V)
    U_size = U.numel()

    def _rows(w):
        rows = w.detach().index_select(0, U).to(device)
        rows.requires_grad_(True)
        return rows

    W_U_wte = _rows(model.wte().weight)
    W_U_lm = W_U_wte if model.config.tie_embeddings else _rows(model.lm_head.weight)
    W_U_ve = {i_str: _rows(ve.weight) for i_str, ve in model.value_embeds.items()}

    return {
        "W_U_wte":        W_U_wte,
        "W_U_ve":         W_U_ve,
        "W_U_lm_head":    W_U_lm,
        "local_idx":      local_idx.to(device),
        "local_targets":  local_targets.to(device),
        "log_correction": torch.tensor(math.log(V) - math.log(U_size), dtype=torch.float32),
    }, U


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


def test_sparse_mode_train_and_infer_work_with_vocab_row_adamw():
    """Sparse mode: dense optimizer handles non-vocab params; VocabRowAdamW handles vocab tables."""
    model = _build_model(sparse_mode=True, tie_embeddings=True, vocab_size=256)
    optimizer = model.setup_optimizer()  # dense-only optimizer; vocab params excluded

    V, d = 256, 64
    vocab_table = {"wte": model.wte().weight.data}  # CPU master weight
    vocab_opt = VocabRowAdamW(
        tables=vocab_table,
        initial_lrs={"wte": 3e-4},
    )

    idx, targets = _sample_batch(vocab_size=256)
    device = torch.device("cpu")
    sparse_ctx, U_step = _build_sparse_context(model, idx, targets, device)

    loss = model(idx, targets, sparse_context=sparse_ctx)
    assert torch.isfinite(loss)
    loss.backward()

    # Collect grad from the pre-fetched GPU leaf
    vocab_grads = {"wte": sparse_ctx["W_U_wte"].grad.float()}
    vocab_opt.step(U_step, vocab_grads, vocab_table, lr_multiplier=1.0)
    optimizer.zero_grad(set_to_none=True)

    # Dense inference should work fine (no sparse_context)
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
    """When U covers the whole vocab, sparse and dense produce the same loss."""
    V = 32
    dense_model = _build_model(sparse_mode=False, tie_embeddings=True, vocab_size=V)
    sparse_model = _build_model(sparse_mode=True, tie_embeddings=True, vocab_size=V)
    sparse_model.load_state_dict(dense_model.state_dict(), strict=True)

    # idx covers all tokens so |U| == V
    idx = torch.arange(0, V, dtype=torch.long).view(1, V)
    targets = idx.roll(shifts=-1, dims=1)
    targets[:, -1] = 0

    dense_loss = dense_model(idx, targets)

    device = torch.device("cpu")
    sparse_ctx, _ = _build_sparse_context(sparse_model, idx, targets, device)
    sparse_loss = sparse_model(idx, targets, sparse_context=sparse_ctx)

    assert torch.allclose(dense_loss, sparse_loss, atol=1e-4, rtol=1e-4)


def test_ddp_union_flag_is_safe_without_dist_init():
    idx = torch.tensor([[1, 2, 2, 3], [3, 4, 4, 5]], dtype=torch.long)
    targets = idx.roll(shifts=-1, dims=1)
    targets[:, -1] = -1

    U_plain, _, _, _ = compute_batch_token_set(idx, targets, vocab_size=64, use_ddp_union=False)
    U_union, _, _, _ = compute_batch_token_set(idx, targets, vocab_size=64, use_ddp_union=True)
    assert torch.equal(U_plain, U_union)


def test_vocab_row_adamw_updates_only_active_rows():
    """VocabRowAdamW must only touch rows present in U_step; inactive rows must be byte-identical."""
    V, d = 128, 16
    wte_data = torch.randn(V, d)
    tables = {"wte": wte_data.clone()}
    inactive_row_before = tables["wte"][50].clone()

    opt = VocabRowAdamW(tables=tables, initial_lrs={"wte": 1e-3})

    U_step = torch.tensor([1, 5, 10, 20], dtype=torch.long)
    grads = {"wte": torch.randn(U_step.numel(), d)}
    opt.step(U_step, grads, tables, lr_multiplier=1.0)

    # Inactive row must be untouched
    assert torch.equal(tables["wte"][50], inactive_row_before), "row 50 should not have changed"
    # Active rows must have changed
    assert not torch.equal(tables["wte"][1], wte_data[1]), "row 1 should have been updated"


def test_vocab_row_adamw_state_dict_roundtrip():
    V, d = 64, 8
    tables = {"wte": torch.randn(V, d)}
    opt = VocabRowAdamW(tables=tables, initial_lrs={"wte": 2e-3})

    U_step = torch.arange(0, 10, dtype=torch.long)
    grads = {"wte": torch.randn(10, d)}
    opt.step(U_step, grads, {**tables}, lr_multiplier=1.0)

    sd = opt.state_dict()
    opt2 = VocabRowAdamW(tables={"wte": torch.randn(V, d)}, initial_lrs={"wte": 2e-3})
    opt2.load_state_dict(sd)
    assert torch.equal(opt2._m["wte"], opt._m["wte"])
    assert torch.equal(opt2._v["wte"], opt._v["wte"])
    assert opt2._t == opt._t


import pytest

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_vocab_row_adamw_gpu_path_matches_cpu_path():
    """GPU-side Adam arithmetic must produce numerically identical results to the CPU path.

    Also exercises the full prefetch_to_gpu() → step() → flush_pending_writes() flow.
    """
    V, d = 128, 32
    device = torch.device("cuda")

    # Shared initial state — clone so both optimizers start from the same tensors.
    wte_master = torch.randn(V, d, dtype=torch.float32)
    tables_cpu = {"wte": wte_master.clone()}
    tables_gpu = {"wte": wte_master.clone()}

    opt_cpu = VocabRowAdamW(tables=tables_cpu, initial_lrs={"wte": 1e-3}, device="cpu")
    opt_gpu = VocabRowAdamW(tables=tables_gpu, initial_lrs={"wte": 1e-3}, device="cuda")
    # Sync initial _m/_v state (both are zero, but ensure same object layout)
    opt_gpu._m["wte"].copy_(opt_cpu._m["wte"])
    opt_gpu._v["wte"].copy_(opt_cpu._v["wte"])

    U_step = torch.tensor([0, 5, 10, 20, 63], dtype=torch.long)
    # grad as GPU bf16 (matches training path)
    grad_gpu_bf16 = torch.randn(U_step.numel(), d, device=device, dtype=torch.bfloat16)

    # ---- CPU path: grad moved to CPU f32 inside step() ----
    opt_cpu.step(U_step, {"wte": grad_gpu_bf16}, tables_cpu, lr_multiplier=1.0)

    # ---- GPU path: use prefetch_to_gpu() which uses pre-allocated pinned buffers ----
    pm, pv, pw = opt_gpu.prefetch_to_gpu(U_step, device, tables_gpu)
    opt_gpu.step(U_step, {"wte": grad_gpu_bf16}, tables_gpu, lr_multiplier=1.0,
                 prefetched_m=pm, prefetched_v=pv, prefetched_w=pw)
    # D2H is async; must call flush_pending_writes() to commit to CPU tables.
    opt_gpu.flush_pending_writes()

    # Results must be exactly equal (both paths use f32 arithmetic on the same f32 inputs).
    assert torch.allclose(tables_cpu["wte"], tables_gpu["wte"], atol=1e-6, rtol=1e-6), \
        "master weight tables differ between CPU and GPU Adam paths"
    assert torch.allclose(opt_cpu._m["wte"], opt_gpu._m["wte"], atol=1e-6, rtol=1e-6), \
        "_m state differs between CPU and GPU Adam paths"
    assert torch.allclose(opt_cpu._v["wte"], opt_gpu._v["wte"], atol=1e-6, rtol=1e-6), \
        "_v state differs between CPU and GPU Adam paths"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_vocab_row_adamw_gpu_two_step_consistency():
    """Two consecutive GPU-path steps must each read the results committed by the previous step.

    This test specifically catches the wait_stream self-wait bug: if the D2H stream
    fires before Adam kernels complete, step 2 will read stale/garbage values from
    the pinned buffers and diverge from the CPU reference.
    """
    V, d = 128, 32
    device = torch.device("cuda")

    wte_master = torch.randn(V, d, dtype=torch.float32)
    tables_cpu = {"wte": wte_master.clone()}
    tables_gpu = {"wte": wte_master.clone()}

    opt_cpu = VocabRowAdamW(tables=tables_cpu, initial_lrs={"wte": 1e-3}, device="cpu")
    opt_gpu = VocabRowAdamW(tables=tables_gpu, initial_lrs={"wte": 1e-3}, device="cuda")
    opt_gpu._m["wte"].copy_(opt_cpu._m["wte"])
    opt_gpu._v["wte"].copy_(opt_cpu._v["wte"])

    # Use different U_step / grad each step to stress that committed state propagates.
    U_step1 = torch.tensor([0, 5, 10, 20, 63], dtype=torch.long)
    U_step2 = torch.tensor([0, 3, 10, 25, 63], dtype=torch.long)  # partial overlap
    grad1 = torch.randn(U_step1.numel(), d, device=device, dtype=torch.bfloat16)
    grad2 = torch.randn(U_step2.numel(), d, device=device, dtype=torch.bfloat16)

    # ---- Step 1 ----
    opt_cpu.step(U_step1, {"wte": grad1}, tables_cpu, lr_multiplier=1.0)

    pm, pv, pw = opt_gpu.prefetch_to_gpu(U_step1, device, tables_gpu)
    opt_gpu.step(U_step1, {"wte": grad1}, tables_gpu, lr_multiplier=1.0,
                 prefetched_m=pm, prefetched_v=pv, prefetched_w=pw)

    # ---- Step 2: flush first (as the training loop does), then prefetch ----
    opt_gpu.flush_pending_writes()
    pm2, pv2, pw2 = opt_gpu.prefetch_to_gpu(U_step2, device, tables_gpu)

    opt_cpu.step(U_step2, {"wte": grad2}, tables_cpu, lr_multiplier=1.0)
    opt_gpu.step(U_step2, {"wte": grad2}, tables_gpu, lr_multiplier=1.0,
                 prefetched_m=pm2, prefetched_v=pv2, prefetched_w=pw2)
    opt_gpu.flush_pending_writes()

    assert torch.allclose(tables_cpu["wte"], tables_gpu["wte"], atol=1e-6, rtol=1e-6), \
        "master weight tables diverged after two GPU-path steps (check wait_stream ordering)"
    assert torch.allclose(opt_cpu._m["wte"], opt_gpu._m["wte"], atol=1e-6, rtol=1e-6), \
        "_m state diverged after two GPU-path steps"
    assert torch.allclose(opt_cpu._v["wte"], opt_gpu._v["wte"], atol=1e-6, rtol=1e-6), \
        "_v state diverged after two GPU-path steps"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_vocab_row_adamw_overlap_cache_deferred_matches_cpu():
    """Overlap-cache + deferred-eviction writeback must match CPU AdamW exactly."""
    V, d = 128, 32
    device = torch.device("cuda")

    wte_master = torch.randn(V, d, dtype=torch.float32)
    tables_cpu = {"wte": wte_master.clone()}
    tables_gpu = {"wte": wte_master.clone()}

    opt_cpu = VocabRowAdamW(tables=tables_cpu, initial_lrs={"wte": 1e-3}, device="cpu")
    opt_gpu = VocabRowAdamW(
        tables=tables_gpu,
        initial_lrs={"wte": 1e-3},
        device="cuda",
        defer_writeback=True,
        overlap_cache=True,
    )
    opt_gpu._m["wte"].copy_(opt_cpu._m["wte"])
    opt_gpu._v["wte"].copy_(opt_cpu._v["wte"])

    U_steps = [
        torch.tensor([0, 5, 10, 20, 63], dtype=torch.long),
        torch.tensor([0, 3, 10, 25, 63], dtype=torch.long),
        torch.tensor([1, 3, 10, 25, 90], dtype=torch.long),
    ]
    grads = [
        torch.randn(U.numel(), d, device=device, dtype=torch.bfloat16)
        for U in U_steps
    ]

    for U_step, grad in zip(U_steps, grads):
        opt_cpu.step(U_step, {"wte": grad}, tables_cpu, lr_multiplier=1.0)
        pm, pv, pw = opt_gpu.prefetch_to_gpu(U_step, device, tables_gpu)
        opt_gpu.step(
            U_step,
            {"wte": grad},
            tables_gpu,
            lr_multiplier=1.0,
            prefetched_m=pm,
            prefetched_v=pv,
            prefetched_w=pw,
        )

    opt_gpu.flush_cache_to_master(tables_gpu)

    assert torch.allclose(tables_cpu["wte"], tables_gpu["wte"], atol=1e-6, rtol=1e-6), \
        "master weights diverged for overlap-cache deferred path"

    touched_rows = torch.unique(torch.cat(U_steps, dim=0), sorted=True)
    m_cpu_touched = opt_cpu._m["wte"].index_select(0, touched_rows)
    v_cpu_touched = opt_cpu._v["wte"].index_select(0, touched_rows)
    m_gpu_touched = opt_gpu._m["wte"].index_select(0, touched_rows)
    v_gpu_touched = opt_gpu._v["wte"].index_select(0, touched_rows)

    # Overlap-cache path uses lazy decay for inactive rows: normalize to current
    # step before comparing to CPU's eager global-decay reference.
    gaps = (opt_gpu._t - opt_gpu._last_seen_step.index_select(0, touched_rows)).to(torch.float32)
    decay_m = torch.pow(torch.tensor(opt_gpu.betas[0], dtype=torch.float32), gaps).unsqueeze(1)
    decay_v = torch.pow(torch.tensor(opt_gpu.betas[1], dtype=torch.float32), gaps).unsqueeze(1)
    m_gpu_touched = m_gpu_touched * decay_m
    v_gpu_touched = v_gpu_touched * decay_v

    assert torch.allclose(m_cpu_touched, m_gpu_touched, atol=1e-5, rtol=1e-5), \
        "_m state diverged on touched rows for overlap-cache deferred path"
    assert torch.allclose(v_cpu_touched, v_gpu_touched, atol=1e-5, rtol=1e-5), \
        "_v state diverged on touched rows for overlap-cache deferred path"
