"""
Dynamic vocab scaling study.

Goal:
1) Run dense baseline at vocab Vstd
2) Run sparse at vocab 10xVstd
3) Increase sparse batch size to match dense VRAM footprint
4) Sweep model depths and widths
5) Compare loss after same number of optimization steps
6) Estimate how much shallower sparse models can be while matching dense loss

Outputs JSON to dev/benchmark_results/.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Iterable

import torch
import torch._dynamo

from nanochat.gpt import GPT, GPTConfig


# Large sweep runs can trigger many valid recompilations in compiled optimizer kernels.
# Raise cache limits so the study completes instead of failing early.
torch._dynamo.config.cache_size_limit = 512
torch._dynamo.config.accumulated_cache_size_limit = 2048


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def choose_num_heads(n_embd: int, target_head_dim: int = 64, max_heads: int = 32) -> int:
    divisors = [h for h in range(1, min(max_heads, n_embd) + 1) if n_embd % h == 0]
    if not divisors:
        return 1
    return min(divisors, key=lambda h: abs((n_embd // h) - target_head_dim))


@dataclass
class DenseConfig:
    vocab_size: int
    n_embd: int
    depth: int
    batch_size: int
    seq_len: int


@dataclass
class SparseConfig:
    vocab_size: int
    n_embd: int
    depth: int
    batch_size: int
    seq_len: int


def build_model(vocab_size: int, n_embd: int, depth: int, seq_len: int, sparse_mode: bool, device: torch.device) -> GPT:
    n_head = choose_num_heads(n_embd)
    config = GPTConfig(
        sequence_len=seq_len,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=n_head,
        n_kv_head=n_head,
        n_embd=n_embd,
        window_pattern="L",
        sparse_mode=sparse_mode,
        sparse_ddp_union=False,
        tie_embeddings=True,
    )
    model = GPT(config, pad_vocab_size_to=1).to(device)
    model.init_weights()
    return model


def make_batch(vocab_size: int, batch_size: int, seq_len: int, active_vocab_frac: float, device: torch.device):
    active_vocab = max(64, int(vocab_size * active_vocab_frac))
    active_vocab = min(active_vocab, vocab_size)
    idx = torch.randint(0, active_vocab, (batch_size, seq_len), device=device, dtype=torch.long)
    targets = idx.roll(shifts=-1, dims=1)
    targets[:, -1] = -1
    return idx, targets


def sync_if_cuda(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def clear_cuda_if_needed(device: torch.device):
    if device.type == "cuda":
        torch.cuda.empty_cache()


def train_n_steps(
    model: GPT,
    steps: int,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    active_vocab_frac: float,
    device: torch.device,
    lr: float,
):
    optimizer = model.setup_optimizer(embedding_lr=lr, unembedding_lr=lr, tied_embedding_lr=lr, matrix_lr=lr)
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()

    losses = []
    times_ms = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model.train()
    steps_completed = 0
    non_finite = False
    for _ in range(steps):
        idx, targets = make_batch(vocab_size, batch_size, seq_len, active_vocab_frac, device)

        sync_if_cuda(device)
        t0 = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx:
            loss = model(idx, targets)
        if not torch.isfinite(loss):
            non_finite = True
            break
        loss.backward()
        optimizer.step()

        sync_if_cuda(device)
        t1 = time.perf_counter()

        losses.append(float(loss.detach().item()))
        times_ms.append((t1 - t0) * 1000.0)
        steps_completed += 1

    if steps_completed == 0:
        return {
            "losses": [],
            "loss_final": None,
            "loss_mean": None,
            "step_ms_mean": None,
            "tokens_per_sec": None,
            "peak_vram_mb": (torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if device.type == "cuda" else None,
            "steps_completed": 0,
            "steps_requested": steps,
            "is_finite": False,
        }

    peak_vram_mb = None
    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    return {
        "losses": losses,
        "loss_final": losses[-1],
        "loss_mean": float(sum(losses) / len(losses)),
        "step_ms_mean": float(sum(times_ms) / len(times_ms)),
        "tokens_per_sec": float((batch_size * seq_len) * 1000.0 / (sum(times_ms) / len(times_ms))),
        "peak_vram_mb": peak_vram_mb,
        "steps_completed": steps_completed,
        "steps_requested": steps,
        "is_finite": (not non_finite) and (steps_completed == steps),
    }


def try_sparse_run_for_vram(
    base_state_dict: dict,
    sparse_vocab: int,
    n_embd: int,
    depth: int,
    seq_len: int,
    batch_size: int,
    steps: int,
    active_vocab_frac: float,
    device: torch.device,
    lr: float,
):
    model = None
    try:
        model = build_model(sparse_vocab, n_embd, depth, seq_len, sparse_mode=True, device=device)
        # load overlapping tensors only (vocab-size tensors differ in first dim)
        state = model.state_dict()
        for k, v in base_state_dict.items():
            if k not in state:
                continue
            if state[k].shape == v.shape:
                state[k].copy_(v)
        out = train_n_steps(
            model=model,
            steps=steps,
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=sparse_vocab,
            active_vocab_frac=active_vocab_frac,
            device=device,
            lr=lr,
        )
        return {"ok": True, "result": out}
    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" in msg or "cuda error" in msg:
            return {"ok": False, "oom": True, "error": str(e)}
        return {"ok": False, "oom": False, "error": str(e)}
    finally:
        if model is not None:
            del model
        gc.collect()
        clear_cuda_if_needed(device)


def find_sparse_batch_for_target_vram(
    base_state_dict: dict,
    sparse_vocab: int,
    n_embd: int,
    depth: int,
    seq_len: int,
    start_batch: int,
    target_vram_mb: float,
    steps: int,
    active_vocab_frac: float,
    device: torch.device,
    lr: float,
    max_search_batch: int,
):
    # Exponential grow phase
    best = None
    lo = start_batch
    hi = start_batch

    while hi <= max_search_batch:
        out = try_sparse_run_for_vram(
            base_state_dict=base_state_dict,
            sparse_vocab=sparse_vocab,
            n_embd=n_embd,
            depth=depth,
            seq_len=seq_len,
            batch_size=hi,
            steps=steps,
            active_vocab_frac=active_vocab_frac,
            device=device,
            lr=lr,
        )
        if not out["ok"]:
            break
        peak = out["result"]["peak_vram_mb"]
        if peak is None:
            # CPU path: no VRAM matching possible; accept this batch
            best = (hi, out["result"])
            break
        if peak <= target_vram_mb:
            best = (hi, out["result"])
            lo = hi
            hi *= 2
        else:
            break

    if best is None:
        # fallback to start batch (even if OOM, caller will record failure)
        out = try_sparse_run_for_vram(
            base_state_dict=base_state_dict,
            sparse_vocab=sparse_vocab,
            n_embd=n_embd,
            depth=depth,
            seq_len=seq_len,
            batch_size=start_batch,
            steps=steps,
            active_vocab_frac=active_vocab_frac,
            device=device,
            lr=lr,
        )
        if out["ok"]:
            return {"batch": start_batch, "run": out["result"]}
        return {"batch": start_batch, "run": None, "error": out.get("error", "failed")}

    # Binary search refine
    left = lo
    right = min(hi, max_search_batch)
    best_batch, best_run = best
    while left <= right:
        mid = (left + right) // 2
        out = try_sparse_run_for_vram(
            base_state_dict=base_state_dict,
            sparse_vocab=sparse_vocab,
            n_embd=n_embd,
            depth=depth,
            seq_len=seq_len,
            batch_size=mid,
            steps=steps,
            active_vocab_frac=active_vocab_frac,
            device=device,
            lr=lr,
        )
        if not out["ok"]:
            right = mid - 1
            continue
        peak = out["result"]["peak_vram_mb"]
        if peak is None or peak <= target_vram_mb:
            best_batch, best_run = mid, out["result"]
            left = mid + 1
        else:
            right = mid - 1

    return {"batch": best_batch, "run": best_run}


def main():
    parser = argparse.ArgumentParser(description="Dense vs sparse 10x vocab scaling study")
    parser.add_argument("--vstd", type=int, default=32768, help="Dense baseline vocab")
    parser.add_argument("--vocab-mult", type=int, default=10, help="Sparse vocab multiplier")
    parser.add_argument("--widths", type=str, default="512,768")
    parser.add_argument("--dense-depths", type=str, default="8,12")
    parser.add_argument("--sparse-depths", type=str, default="4,6,8,12")
    parser.add_argument("--batch-size", type=int, default=4, help="Dense baseline batch size")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--active-vocab-frac", type=float, default=0.1, help="Fraction of vocab activated per batch")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--loss-match-eps", type=float, default=0.02, help="Accept sparse as matched if final_loss <= dense_loss + eps")
    parser.add_argument("--max-search-batch", type=int, default=64)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    widths = parse_int_list(args.widths)
    dense_depths = parse_int_list(args.dense_depths)
    sparse_depths = parse_int_list(args.sparse_depths)

    if not args.device:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_path = args.output or f"dev/benchmark_results/dynamic_vocab_scaling_study_{ts}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    sparse_vocab = args.vstd * args.vocab_mult

    payload = {
        "timestamp_utc": ts,
        "device": str(device),
        "torch_version": torch.__version__,
        "args": vars(args),
        "results": [],
    }

    total = len(widths) * len(dense_depths)
    case_idx = 0

    for width in widths:
        for dense_depth in dense_depths:
            case_idx += 1
            print(f"[{case_idx}/{total}] width={width}, dense_depth={dense_depth}")

            dense_model = build_model(args.vstd, width, dense_depth, args.seq_len, sparse_mode=False, device=device)
            dense_result = train_n_steps(
                model=dense_model,
                steps=args.steps,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                vocab_size=args.vstd,
                active_vocab_frac=args.active_vocab_frac,
                device=device,
                lr=args.lr,
            )
            dense_state = {k: v.detach().clone() for k, v in dense_model.state_dict().items()}
            target_vram = dense_result["peak_vram_mb"]

            sparse_trials = []
            for sparse_depth in sparse_depths:
                search = find_sparse_batch_for_target_vram(
                    base_state_dict=dense_state,
                    sparse_vocab=sparse_vocab,
                    n_embd=width,
                    depth=sparse_depth,
                    seq_len=args.seq_len,
                    start_batch=args.batch_size,
                    target_vram_mb=target_vram if target_vram is not None else float("inf"),
                    steps=args.steps,
                    active_vocab_frac=args.active_vocab_frac,
                    device=device,
                    lr=args.lr,
                    max_search_batch=args.max_search_batch,
                )
                sparse_trials.append({
                    "sparse_depth": sparse_depth,
                    "matched_batch_size": search["batch"],
                    "run": search.get("run"),
                    "error": search.get("error"),
                })

            matched_depth = None
            dense_final = dense_result["loss_final"]
            for t in sorted(sparse_trials, key=lambda x: x["sparse_depth"]):
                run = t.get("run")
                if run is None:
                    continue
                if not run.get("is_finite", False):
                    continue
                if run["loss_final"] <= dense_final + args.loss_match_eps:
                    matched_depth = t["sparse_depth"]
                    break

            result = {
                "width": width,
                "dense": {
                    "config": asdict(DenseConfig(
                        vocab_size=args.vstd,
                        n_embd=width,
                        depth=dense_depth,
                        batch_size=args.batch_size,
                        seq_len=args.seq_len,
                    )),
                    "metrics": dense_result,
                },
                "sparse": {
                    "vocab_size": sparse_vocab,
                    "trials": sparse_trials,
                    "min_depth_meeting_loss": matched_depth,
                    "depth_reduction": (dense_depth - matched_depth) if matched_depth is not None else None,
                },
            }
            payload["results"].append(result)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

            del dense_model
            del dense_state
            gc.collect()
            clear_cuda_if_needed(device)

    # Aggregate summary
    depth_reductions = [r["sparse"]["depth_reduction"] for r in payload["results"] if r["sparse"]["depth_reduction"] is not None]
    speedup_vals = []
    vram_match_ratios = []
    for r in payload["results"]:
        dense_tps = r["dense"]["metrics"].get("tokens_per_sec")
        dense_vram = r["dense"]["metrics"]["peak_vram_mb"]
        for t in r["sparse"]["trials"]:
            run = t.get("run")
            if run is None:
                continue
            if run.get("is_finite", False) and dense_tps is not None and run.get("tokens_per_sec") is not None:
                speedup_vals.append(run["tokens_per_sec"] / max(dense_tps, 1e-9))
            if dense_vram is not None and run["peak_vram_mb"] is not None:
                vram_match_ratios.append(run["peak_vram_mb"] / max(dense_vram, 1e-9))

    def stat(xs):
        if not xs:
            return None
        return {"min": min(xs), "max": max(xs), "mean": sum(xs) / len(xs)}

    payload["summary"] = {
        "num_results": len(payload["results"]),
        "depth_reduction": stat(depth_reductions),
        "speedup_vs_dense": stat(speedup_vals),
        "vram_ratio_vs_dense_target": stat(vram_match_ratios),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Study complete")
    print(f"Wrote JSON results to: {output_path}")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
