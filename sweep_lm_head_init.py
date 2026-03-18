#!/usr/bin/env python3
"""
Sweep sparse startup stability over lm_head initialization std values.

The goal is to compare the first few sparse updates under a fixed manifest and
fixed cold-bias setting, then summarize which init scale gives the smoothest
early grad-norm ramp without changing the rest of the sparse recipe.
"""

from __future__ import annotations

import re
import subprocess
import sys


STEP_RE = re.compile(
    r"step\s+(\d{5})/\d+.*?loss:\s+([0-9.]+).*?grad_norm:\s+([0-9.]+).*?\|\s+U:\s+([0-9,]+)",
    re.S,
)


def parse_metrics(output: str) -> dict:
    metrics = {
        "steps": 0,
        "grad_norms": [],
        "losses": [],
        "u_counts": [],
        "max_grad_norm": None,
        "step_of_max_grad_norm": None,
        "max_grad_norm_first_10": None,
        "max_grad_norm_first_20": None,
        "final_loss": None,
    }
    for match in STEP_RE.finditer(output):
        step = int(match.group(1))
        loss = float(match.group(2))
        grad_value = float(match.group(3))
        unique_count = int(match.group(4).replace(",", ""))
        metrics["steps"] = max(metrics["steps"], step + 1)
        metrics["losses"].append((step, loss))
        metrics["u_counts"].append((step, unique_count))
        metrics["final_loss"] = loss
        metrics["grad_norms"].append((step, grad_value))
        if metrics["max_grad_norm"] is None or grad_value > metrics["max_grad_norm"]:
            metrics["max_grad_norm"] = grad_value
            metrics["step_of_max_grad_norm"] = step
        if step < 10 and (metrics["max_grad_norm_first_10"] is None or grad_value > metrics["max_grad_norm_first_10"]):
            metrics["max_grad_norm_first_10"] = grad_value
        if step < 20 and (metrics["max_grad_norm_first_20"] is None or grad_value > metrics["max_grad_norm_first_20"]):
            metrics["max_grad_norm_first_20"] = grad_value
    return metrics


def summarize(std: float, init_dist: str, returncode: int, metrics: dict, unembed_warmup: int = 0, max_grad_norm: float = 0.0, first_hot_unembedding_lr: float = 0.0, hot_ramp_activations: int = 0, hot_ramp_start_lr: float = 0.0) -> str:
    if returncode != 0:
        return f"lm_head_std={std:.6f} | unembed_warmup={unembed_warmup:3d} | first_hot_unembed_lr={first_hot_unembedding_lr:.4f} | hot_ramp={hot_ramp_activations:2d}@{hot_ramp_start_lr:.4f} | grad_clip={max_grad_norm:.1f} | FAILED"
    mgn = metrics["max_grad_norm"]
    final_loss = metrics["final_loss"]
    if mgn is None or final_loss is None:
        return f"lm_head_std={std:.6f} | unembed_warmup={unembed_warmup:3d} | first_hot_unembed_lr={first_hot_unembedding_lr:.4f} | hot_ramp={hot_ramp_activations:2d}@{hot_ramp_start_lr:.4f} | grad_clip={max_grad_norm:.1f} | no grad_norm captured"
    return (
        f"lm_head_std={std:.6f} | unembed_warmup={unembed_warmup:3d} | first_hot_unembed_lr={first_hot_unembedding_lr:.4f} | hot_ramp={hot_ramp_activations:2d}@{hot_ramp_start_lr:.4f} | grad_clip={max_grad_norm:.1f}"
        f" | max_grad_norm_10={metrics['max_grad_norm_first_10']:.4f}"
        f" | max_grad_norm_20={metrics['max_grad_norm_first_20']:.4f}"
        f" | max_grad_norm_all={mgn:.4f}"
        f" @ step {metrics['step_of_max_grad_norm']:02d} | final_loss={final_loss:.4f}"
    )


def run_training(lm_head_std: float, lm_head_dist: str, cold_bias_scale: float, num_iterations: int,
                 unembed_warmup_steps: int = 0, max_grad_norm: float = 0.0, first_hot_unembedding_lr: float = 0.0,
                 hot_ramp_activations: int = 0, hot_ramp_start_lr: float = 0.0) -> tuple[int, str]:
    cmd = [
        sys.executable, "-m", "scripts.base_train",
        "--run=dummy",
        "--depth=6",
        f"--num-iterations={num_iterations}",
        "--device-batch-size=32",
        "--sparse-mode",
        "--window-pattern", "L",
        "--sparse-manifest", "manifests/d6_2kseq_32batch_250step_noaccum.json",
        "--log-every", "1",
        "--eval-every", "-1",
        "--grad-norm-every", "1",
        "--unembedding-lr=0.018",
        "--core-metric-every", "-1",
        "--total-batch-size", "65536",
        "--sparse-logit-scale", "1",
        "--aspect-ratio", "64",
        "--warmup-ratio", "0",
        "--warmdown-ratio", "0.5",
        f"--sparse-cold-bias-scale={cold_bias_scale}",
        f"--lm-head-init-std={lm_head_std}",
        f"--lm-head-init-dist={lm_head_dist}",
        f"--sparse-unembed-warmup-steps={unembed_warmup_steps}",
        f"--sparse-first-hot-unembedding-lr={first_hot_unembedding_lr}",
        f"--sparse-hot-unembed-ramp-activations={hot_ramp_activations}",
        f"--sparse-hot-unembed-ramp-start-lr={hot_ramp_start_lr}",
        f"--max-grad-norm={max_grad_norm}",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = proc.stdout + proc.stderr
    return proc.returncode, output


def main() -> None:
    cold_bias_scale = 2.2
    num_iterations = 250
    lm_head_std = 0.00084
    lm_head_dist = "normal"

    first_hot_unembedding_lr = 0.0
    hot_ramp_activations_values = [0, 1, 2, 4, 8]
    hot_ramp_start_lr = 0.004
    warmup_steps = 0
    max_grad_norm = 0.0

    print("Sparse startup control sweep: per-token hot-activation lm_head LR ramp")
    print(f"Fixed: lm_head_std={lm_head_std}, lm_head_dist={lm_head_dist}, cold_bias_scale={cold_bias_scale}, iterations={num_iterations}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Max grad norm (clip): {max_grad_norm}")
    print(f"First-hot unembedding LR: {first_hot_unembedding_lr}")
    print(f"Hot ramp activations: {hot_ramp_activations_values}")
    print(f"Hot ramp start LR: {hot_ramp_start_lr}")

    summaries = []
    for hot_ramp_activations in hot_ramp_activations_values:
        label = f"hot_ramp={hot_ramp_activations:2d}@{hot_ramp_start_lr:.4f}"
        print(f"\n{'-' * 60}")
        print(f"Running: {label}")
        print(f"{'-' * 60}")
        returncode, output = run_training(
            lm_head_std,
            lm_head_dist=lm_head_dist,
            cold_bias_scale=cold_bias_scale,
            num_iterations=num_iterations,
            unembed_warmup_steps=warmup_steps,
            max_grad_norm=max_grad_norm,
            first_hot_unembedding_lr=first_hot_unembedding_lr,
            hot_ramp_activations=hot_ramp_activations,
            hot_ramp_start_lr=hot_ramp_start_lr,
        )
        metrics = parse_metrics(output)
        summary = summarize(
            lm_head_std,
            lm_head_dist,
            returncode,
            metrics,
            unembed_warmup=warmup_steps,
            max_grad_norm=max_grad_norm,
            first_hot_unembedding_lr=first_hot_unembedding_lr,
            hot_ramp_activations=hot_ramp_activations,
            hot_ramp_start_lr=hot_ramp_start_lr,
        )
        summaries.append((hot_ramp_activations, returncode, metrics, summary, output))
        print(summary)
        if returncode != 0:
            print(output)

    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    for _, _, _, summary, _ in summaries:
        print(summary)

    successful = [item for item in summaries if item[1] == 0 and item[2]["max_grad_norm"] is not None]
    if successful:
        best = min(
            successful,
            key=lambda item: (
                item[2]["max_grad_norm_first_10"],
                item[2]["max_grad_norm_first_20"],
                item[2]["max_grad_norm"],
            ),
        )
        print(
            f"\nBest: hot_ramp_activations={best[0]} @ start_lr={hot_ramp_start_lr:.4f} "
            "based on minimum observed early grad-norm peak."
        )


if __name__ == "__main__":
    main()