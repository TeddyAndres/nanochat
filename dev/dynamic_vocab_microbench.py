"""
Dynamic vocabulary micro-benchmark for nanochat.

Evaluates dense vs sparse-mode training micro-steps over a grid of:
- vocab size
- embedding dimension
- batch size
- sequence length
- model depth

Outputs JSON with speed, VRAM, and calibration-proxy metrics.

Example:
python -m dev.dynamic_vocab_microbench \
  --vocab-sizes 32768,65536 \
  --embd-sizes 512,768 \
  --batch-sizes 2,4 \
  --seq-lens 256,512 \
  --depths 8,12 \
  --steps 8 --warmup 3
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from dataclasses import asdict, dataclass
from contextlib import nullcontext
from datetime import datetime, UTC
from typing import Iterable

import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.sparse_vocab import compute_batch_token_set


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def choose_num_heads(n_embd: int, target_head_dim: int = 64, max_heads: int = 16) -> int:
    divisors = [h for h in range(1, min(max_heads, n_embd) + 1) if n_embd % h == 0]
    if not divisors:
        return 1
    return min(divisors, key=lambda h: abs((n_embd // h) - target_head_dim))


def generate_batch(vocab_size: int, batch_size: int, seq_len: int, active_vocab_frac: float, device: torch.device):
    active_vocab = max(32, int(vocab_size * active_vocab_frac))
    active_vocab = min(active_vocab, vocab_size)
    idx = torch.randint(0, active_vocab, (batch_size, seq_len), dtype=torch.long, device=device)
    targets = idx.roll(shifts=-1, dims=1)
    targets[:, -1] = -1
    return idx, targets, active_vocab


def maybe_sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_step_time(model: GPT, idx: torch.Tensor, targets: torch.Tensor, warmup: int, steps: int, autocast_ctx):
    model.train()
    device = idx.device

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    times_ms: list[float] = []
    for i in range(warmup + steps):
        maybe_sync(device)
        t0 = time.perf_counter()

        with autocast_ctx:
            loss = model(idx, targets)
        loss.backward()
        model.zero_grad(set_to_none=True)

        maybe_sync(device)
        t1 = time.perf_counter()
        if i >= warmup:
            times_ms.append((t1 - t0) * 1000.0)

    peak_vram_mb = None
    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    avg_ms = sum(times_ms) / max(1, len(times_ms))
    return {
        "step_ms_avg": avg_ms,
        "step_ms_min": min(times_ms) if times_ms else None,
        "step_ms_max": max(times_ms) if times_ms else None,
        "peak_vram_mb": peak_vram_mb,
    }


@torch.no_grad()
def calibration_proxy(dense_model: GPT, sparse_model: GPT, idx: torch.Tensor, targets: torch.Tensor, autocast_ctx):
    dense_model.eval()
    sparse_model.eval()
    with autocast_ctx:
        dense_loss = dense_model(idx, targets).item()
        sparse_loss_corrected = sparse_model(idx, targets).item()
    return {
        "dense_loss": dense_loss,
        "sparse_loss_corrected": sparse_loss_corrected,
        "loss_gap_abs": abs(dense_loss - sparse_loss_corrected),
    }


@dataclass
class BenchCase:
    vocab_size: int
    n_embd: int
    batch_size: int
    seq_len: int
    depth: int


def iter_cases(vocab_sizes: Iterable[int], embd_sizes: Iterable[int], batch_sizes: Iterable[int], seq_lens: Iterable[int], depths: Iterable[int]):
    for v, d, b, t, l in itertools.product(vocab_sizes, embd_sizes, batch_sizes, seq_lens, depths):
        yield BenchCase(vocab_size=v, n_embd=d, batch_size=b, seq_len=t, depth=l)


def run_case(case: BenchCase, device: torch.device, warmup: int, steps: int, active_vocab_frac: float):
    n_head = choose_num_heads(case.n_embd)

    dense_cfg = GPTConfig(
        sequence_len=case.seq_len,
        vocab_size=case.vocab_size,
        n_layer=case.depth,
        n_head=n_head,
        n_kv_head=n_head,
        n_embd=case.n_embd,
        window_pattern="L",
        sparse_mode=False,
        sparse_ddp_union=False,
        tie_embeddings=True,
    )
    sparse_cfg = GPTConfig(
        sequence_len=case.seq_len,
        vocab_size=case.vocab_size,
        n_layer=case.depth,
        n_head=n_head,
        n_kv_head=n_head,
        n_embd=case.n_embd,
        window_pattern="L",
        sparse_mode=True,
        sparse_ddp_union=False,
        tie_embeddings=True,
    )

    dense_model = GPT(dense_cfg, pad_vocab_size_to=1).to(device)
    dense_model.init_weights()

    sparse_model = GPT(sparse_cfg, pad_vocab_size_to=1).to(device)
    sparse_model.init_weights()
    sparse_model.load_state_dict(dense_model.state_dict(), strict=True)

    idx, targets, active_vocab = generate_batch(
        vocab_size=case.vocab_size,
        batch_size=case.batch_size,
        seq_len=case.seq_len,
        active_vocab_frac=active_vocab_frac,
        device=device,
    )

    U, _, _, _ = compute_batch_token_set(idx, targets, vocab_size=case.vocab_size, use_ddp_union=False)

    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()

    dense_perf = benchmark_step_time(dense_model, idx, targets, warmup=warmup, steps=steps, autocast_ctx=autocast_ctx)
    sparse_perf = benchmark_step_time(sparse_model, idx, targets, warmup=warmup, steps=steps, autocast_ctx=autocast_ctx)
    calib = calibration_proxy(dense_model, sparse_model, idx, targets, autocast_ctx=autocast_ctx)

    tokens_per_step = case.batch_size * case.seq_len
    dense_tps = 1000.0 * tokens_per_step / dense_perf["step_ms_avg"]
    sparse_tps = 1000.0 * tokens_per_step / sparse_perf["step_ms_avg"]

    vram_speedup = None
    if dense_perf["peak_vram_mb"] is not None and sparse_perf["peak_vram_mb"] is not None:
        vram_speedup = dense_perf["peak_vram_mb"] / max(sparse_perf["peak_vram_mb"], 1e-9)

    return {
        "case": asdict(case),
        "n_head": n_head,
        "active_vocab": active_vocab,
        "local_vocab_size": int(U.numel()),
        "local_vocab_frac": float(U.numel() / case.vocab_size),
        "dense": {
            **dense_perf,
            "tokens_per_sec": dense_tps,
        },
        "sparse": {
            **sparse_perf,
            "tokens_per_sec": sparse_tps,
        },
        "benefits": {
            "speedup_x": sparse_tps / max(dense_tps, 1e-9),
            "vram_reduction_x": vram_speedup,
        },
        "calibration": calib,
    }


def summarize(results: list[dict]):
    speedups = [r["benefits"]["speedup_x"] for r in results]
    vram = [r["benefits"]["vram_reduction_x"] for r in results if r["benefits"]["vram_reduction_x"] is not None]
    gaps = [r["calibration"]["loss_gap_abs"] for r in results]

    def stats(xs: list[float]):
        if not xs:
            return None
        return {
            "min": min(xs),
            "max": max(xs),
            "mean": sum(xs) / len(xs),
        }

    return {
        "num_cases": len(results),
        "speedup_x": stats(speedups),
        "vram_reduction_x": stats(vram),
        "calibration_loss_gap_abs": stats(gaps),
    }


def main():
    parser = argparse.ArgumentParser(description="Dynamic vocab micro-benchmark")
    parser.add_argument("--vocab-sizes", type=str, default="32768,65536")
    parser.add_argument("--embd-sizes", type=str, default="512,768")
    parser.add_argument("--batch-sizes", type=str, default="2,4")
    parser.add_argument("--seq-lens", type=str, default="256,512")
    parser.add_argument("--depths", type=str, default="8,12")
    parser.add_argument("--active-vocab-frac", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--max-cases", type=int, default=0, help="0 means run all")
    parser.add_argument("--device", type=str, default="", help="cuda|cpu|mps (empty=auto)")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    device_str = args.device
    if not device_str:
        if torch.cuda.is_available():
            device_str = "cuda"
        elif torch.backends.mps.is_available():
            device_str = "mps"
        else:
            device_str = "cpu"
    device = torch.device(device_str)

    vocab_sizes = parse_int_list(args.vocab_sizes)
    embd_sizes = parse_int_list(args.embd_sizes)
    batch_sizes = parse_int_list(args.batch_sizes)
    seq_lens = parse_int_list(args.seq_lens)
    depths = parse_int_list(args.depths)

    all_cases = list(iter_cases(vocab_sizes, embd_sizes, batch_sizes, seq_lens, depths))
    if args.max_cases > 0:
        all_cases = all_cases[:args.max_cases]

    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_path = args.output
    if not output_path:
        output_dir = os.path.join("dev", "benchmark_results")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"dynamic_vocab_microbench_{ts}.json")
    else:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print(f"Running {len(all_cases)} cases on device={device} ...")

    results: list[dict] = []
    for i, case in enumerate(all_cases, start=1):
        print(f"[{i}/{len(all_cases)}] case={asdict(case)}")
        case_result = run_case(
            case=case,
            device=device,
            warmup=args.warmup,
            steps=args.steps,
            active_vocab_frac=args.active_vocab_frac,
        )
        results.append(case_result)

        payload = {
            "timestamp_utc": ts,
            "device": str(device),
            "torch_version": torch.__version__,
            "args": vars(args),
            "summary": summarize(results),
            "results": results,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    print("Benchmark complete")
    print(f"Wrote JSON results to: {output_path}")
    summary = summarize(results)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
