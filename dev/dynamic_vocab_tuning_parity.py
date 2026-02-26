"""
Dynamic vocab tuning + parity study.

This script runs two studies and writes one JSON report:
1) Stability tuning: LR sweep comparing dense(Vstd) vs sparse(10x Vstd) over equal steps.
2) Parity validation: dense vs sparse with SAME vocab to verify corrected loss math is materially aligned.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass, asdict
from datetime import UTC, datetime

import torch
import torch._dynamo

from nanochat.gpt import GPT, GPTConfig


torch._dynamo.config.cache_size_limit = 512
torch._dynamo.config.accumulated_cache_size_limit = 2048


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def choose_num_heads(n_embd: int, target_head_dim: int = 64, max_heads: int = 32) -> int:
    divisors = [h for h in range(1, min(max_heads, n_embd) + 1) if n_embd % h == 0]
    if not divisors:
        return 1
    return min(divisors, key=lambda h: abs((n_embd // h) - target_head_dim))


def clear_mem(device: torch.device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def make_model(vocab_size: int, n_embd: int, depth: int, seq_len: int, sparse_mode: bool, device: torch.device, tie_embeddings: bool = True):
    n_head = choose_num_heads(n_embd)
    cfg = GPTConfig(
        sequence_len=seq_len,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=n_head,
        n_kv_head=n_head,
        n_embd=n_embd,
        window_pattern="L",
        sparse_mode=sparse_mode,
        sparse_ddp_union=False,
        tie_embeddings=tie_embeddings,
    )
    model = GPT(cfg, pad_vocab_size_to=1).to(device)
    model.init_weights()
    return model


def make_batch(vocab_size: int, batch_size: int, seq_len: int, active_vocab_frac: float, device: torch.device):
    active_vocab = max(64, int(vocab_size * active_vocab_frac))
    active_vocab = min(active_vocab, vocab_size)
    idx = torch.randint(0, active_vocab, (batch_size, seq_len), device=device, dtype=torch.long)
    targets = idx.roll(shifts=-1, dims=1)
    targets[:, -1] = -1
    return idx, targets


def sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@dataclass
class TrainRun:
    mode: str
    vocab_size: int
    n_embd: int
    depth: int
    batch_size: int
    seq_len: int
    lr: float
    steps_requested: int
    steps_completed: int
    is_finite: bool
    loss_final: float | None
    loss_mean: float | None
    loss_start: float | None
    loss_delta: float | None
    step_ms_mean: float | None
    tokens_per_sec: float | None
    peak_vram_mb: float | None


def train_run(model: GPT, vocab_size: int, mode: str, batch_size: int, seq_len: int, lr: float, steps: int, active_vocab_frac: float, device: torch.device) -> TrainRun:
    optimizer = model.setup_optimizer(embedding_lr=lr, unembedding_lr=lr, tied_embedding_lr=lr, matrix_lr=lr)
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()

    losses: list[float] = []
    times_ms: list[float] = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model.train()
    steps_completed = 0
    is_finite = True

    for _ in range(steps):
        idx, targets = make_batch(vocab_size, batch_size, seq_len, active_vocab_frac, device)

        optimizer.zero_grad(set_to_none=True)
        sync(device)
        t0 = time.perf_counter()

        with autocast_ctx:
            loss = model(idx, targets)

        if not torch.isfinite(loss):
            is_finite = False
            break

        loss.backward()
        optimizer.step()

        sync(device)
        t1 = time.perf_counter()

        losses.append(float(loss.detach().item()))
        times_ms.append((t1 - t0) * 1000.0)
        steps_completed += 1

    peak_vram = None
    if device.type == "cuda":
        peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    if not losses:
        return TrainRun(
            mode=mode,
            vocab_size=vocab_size,
            n_embd=model.config.n_embd,
            depth=model.config.n_layer,
            batch_size=batch_size,
            seq_len=seq_len,
            lr=lr,
            steps_requested=steps,
            steps_completed=0,
            is_finite=False,
            loss_final=None,
            loss_mean=None,
            loss_start=None,
            loss_delta=None,
            step_ms_mean=None,
            tokens_per_sec=None,
            peak_vram_mb=peak_vram,
        )

    step_ms_mean = sum(times_ms) / len(times_ms)
    tokens_per_sec = (batch_size * seq_len) * 1000.0 / step_ms_mean

    return TrainRun(
        mode=mode,
        vocab_size=vocab_size,
        n_embd=model.config.n_embd,
        depth=model.config.n_layer,
        batch_size=batch_size,
        seq_len=seq_len,
        lr=lr,
        steps_requested=steps,
        steps_completed=steps_completed,
        is_finite=is_finite and (steps_completed == steps),
        loss_final=losses[-1],
        loss_mean=sum(losses) / len(losses),
        loss_start=losses[0],
        loss_delta=losses[-1] - losses[0],
        step_ms_mean=step_ms_mean,
        tokens_per_sec=tokens_per_sec,
        peak_vram_mb=peak_vram,
    )


def stability_sweep(args, device: torch.device):
    out = []
    sparse_vocab = args.vstd * args.vocab_mult

    for width in parse_int_list(args.widths):
        for dense_depth in parse_int_list(args.dense_depths):
            for sparse_depth in parse_int_list(args.sparse_depths):
                for lr in parse_float_list(args.lrs):
                    # Dense baseline
                    dense_model = make_model(args.vstd, width, dense_depth, args.seq_len, sparse_mode=False, device=device)
                    dense_run = train_run(
                        dense_model,
                        vocab_size=args.vstd,
                        mode="dense",
                        batch_size=args.batch_size,
                        seq_len=args.seq_len,
                        lr=lr,
                        steps=args.steps,
                        active_vocab_frac=args.active_vocab_frac,
                        device=device,
                    )
                    clear_mem(device)

                    # Sparse 10x vocab
                    sparse_model = make_model(sparse_vocab, width, sparse_depth, args.seq_len, sparse_mode=True, device=device)
                    sparse_run = train_run(
                        sparse_model,
                        vocab_size=sparse_vocab,
                        mode="sparse",
                        batch_size=args.batch_size,
                        seq_len=args.seq_len,
                        lr=lr,
                        steps=args.steps,
                        active_vocab_frac=args.active_vocab_frac,
                        device=device,
                    )
                    clear_mem(device)

                    out.append({
                        "width": width,
                        "dense_depth": dense_depth,
                        "sparse_depth": sparse_depth,
                        "lr": lr,
                        "dense": asdict(dense_run),
                        "sparse": asdict(sparse_run),
                        "comparative": {
                            "both_finite": dense_run.is_finite and sparse_run.is_finite,
                            "loss_gap_final": None if (dense_run.loss_final is None or sparse_run.loss_final is None) else (sparse_run.loss_final - dense_run.loss_final),
                            "speed_ratio_sparse_over_dense": None if (dense_run.tokens_per_sec is None or sparse_run.tokens_per_sec is None) else (sparse_run.tokens_per_sec / dense_run.tokens_per_sec),
                            "vram_ratio_sparse_over_dense": None if (dense_run.peak_vram_mb is None or sparse_run.peak_vram_mb is None) else (sparse_run.peak_vram_mb / max(dense_run.peak_vram_mb, 1e-9)),
                        },
                    })
    return out


def parity_study(args, device: torch.device):
    results = {}
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()

    # Case A: exact parity when U == V (small vocab fully covered by one sequence)
    vocab_exact = args.parity_vocab
    seq_exact = vocab_exact
    exact_device = torch.device("cpu")
    model_dense = make_model(vocab_exact, args.parity_embd, args.parity_depth, seq_exact, sparse_mode=False, device=exact_device)
    model_sparse = make_model(vocab_exact, args.parity_embd, args.parity_depth, seq_exact, sparse_mode=True, device=exact_device)
    model_sparse.load_state_dict(model_dense.state_dict(), strict=True)

    idx = torch.arange(0, vocab_exact, device=exact_device, dtype=torch.long).unsqueeze(0)
    targets = idx.roll(shifts=-1, dims=1)

    model_dense.zero_grad(set_to_none=True)
    model_sparse.zero_grad(set_to_none=True)

    loss_dense = model_dense(idx, targets)
    loss_sparse = model_sparse(idx, targets)

    loss_dense.backward()
    loss_sparse.backward()

    g_dense = model_dense.wte().weight.grad
    g_sparse = model_sparse.wte().weight.grad.to_dense().to(g_dense.device)
    grad_diff = (g_dense - g_sparse).abs().max().item()

    results["exact_full_vocab"] = {
        "dense_loss": float(loss_dense.item()),
        "sparse_loss_corrected": float(loss_sparse.item()),
        "loss_abs_diff": abs(float(loss_dense.item() - loss_sparse.item())),
        "wte_grad_max_abs_diff": grad_diff,
    }

    clear_mem(exact_device)

    # Case B: practical same-vocab random batches (materially similar trend, not exact)
    vocab_practical = args.vstd
    model_dense2 = make_model(vocab_practical, args.parity_embd, args.parity_depth, args.seq_len, sparse_mode=False, device=device)
    model_sparse2 = make_model(vocab_practical, args.parity_embd, args.parity_depth, args.seq_len, sparse_mode=True, device=device)
    model_sparse2.load_state_dict(model_dense2.state_dict(), strict=True)

    diffs = []
    for _ in range(args.parity_batches):
        idx_b, targets_b = make_batch(vocab_practical, args.batch_size, args.seq_len, args.active_vocab_frac, device)
        with torch.no_grad(), autocast_ctx:
            d = float(model_dense2(idx_b, targets_b).item())
            s = float(model_sparse2(idx_b, targets_b).item())
        diffs.append(abs(d - s))

    results["practical_same_vocab"] = {
        "num_batches": args.parity_batches,
        "loss_abs_diff_min": min(diffs),
        "loss_abs_diff_max": max(diffs),
        "loss_abs_diff_mean": sum(diffs) / len(diffs),
    }

    clear_mem(device)
    return results


def summarize_stability(rows: list[dict]):
    stable = [r for r in rows if r["comparative"]["both_finite"]]
    best = []
    # pick best sparse loss per (width, dense_depth, sparse_depth)
    groups = {}
    for r in stable:
        key = (r["width"], r["dense_depth"], r["sparse_depth"])
        groups.setdefault(key, []).append(r)
    for key, items in groups.items():
        items = [x for x in items if x["sparse"]["loss_final"] is not None]
        if not items:
            continue
        best_item = min(items, key=lambda x: x["sparse"]["loss_final"])
        best.append({
            "width": key[0],
            "dense_depth": key[1],
            "sparse_depth": key[2],
            "best_lr": best_item["lr"],
            "sparse_loss_final": best_item["sparse"]["loss_final"],
            "dense_loss_final": best_item["dense"]["loss_final"],
            "loss_gap_final": best_item["comparative"]["loss_gap_final"],
        })
    return {
        "num_trials": len(rows),
        "num_stable": len(stable),
        "best_by_shape": best,
    }


def main():
    parser = argparse.ArgumentParser(description="Dynamic vocab tuning + parity study")
    parser.add_argument("--vstd", type=int, default=32768)
    parser.add_argument("--vocab-mult", type=int, default=10)
    parser.add_argument("--widths", type=str, default="512,768")
    parser.add_argument("--dense-depths", type=str, default="8,12")
    parser.add_argument("--sparse-depths", type=str, default="4,6,8,12")
    parser.add_argument("--lrs", type=str, default="0.0005,0.001,0.002,0.004")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--active-vocab-frac", type=float, default=0.1)
    parser.add_argument("--parity-vocab", type=int, default=64)
    parser.add_argument("--parity-embd", type=int, default=128)
    parser.add_argument("--parity-depth", type=int, default=2)
    parser.add_argument("--parity-batches", type=int, default=20)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output = args.output or f"dev/benchmark_results/dynamic_vocab_tuning_parity_{ts}.json"
    os.makedirs(os.path.dirname(output), exist_ok=True)

    stability = stability_sweep(args, device)
    parity = parity_study(args, device)

    payload = {
        "timestamp_utc": ts,
        "device": str(device),
        "torch_version": torch.__version__,
        "args": vars(args),
        "stability": {
            "rows": stability,
            "summary": summarize_stability(stability),
        },
        "parity": parity,
    }

    with open(output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote tuning+parity report: {output}")
    print(json.dumps(payload["stability"]["summary"], indent=2))
    print(json.dumps(payload["parity"], indent=2))


if __name__ == "__main__":
    main()
