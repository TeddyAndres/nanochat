#!/usr/bin/env python3
"""
Flexible sweep script for running multiple training configurations sequentially.
Define a list of training commands to execute one after another.

Usage:
    1. Edit the RUNS list below with your desired training commands.
    2. Run: python sweep_runs.py
    3. Or use --dry-run to preview commands without executing.
"""

import subprocess
import sys
import os
import argparse
import shlex
import signal
import time
import gc
from datetime import datetime


# ============================================================
# DEFINE YOUR RUNS HERE
# ============================================================
# Each entry is a list of command-line arguments passed to python -m scripts.base_train.
# Do NOT include "python -m scripts.base_train" itself; the script adds it automatically.
#
# Example:
# RUNS = [
#     {
#         "name": "run1",
#         "args": [
#             "--depth=12",
#             "--num-iterations=20000",
#             ... etc ...
#         ],
#     },
#     {
#         "name": "run2",
#         "args": [
#             ... different args ...
#         ],
#     },
# ]
#
# You can also use the helper function to build args from a string:
#     parse_cmd_string("-m scripts.base_train --depth=12 ...")
# which strips the "-m scripts.base_train" prefix automatically.

def parse_cmd_string(cmd_str):
    """Parse a command string (like the one you'd paste into a shell) into args list."""
    parts = shlex.split(cmd_str)
    # Strip leading "-m scripts.base_train" if present
    idx = 0
    while idx < len(parts) - 1:
        if parts[idx] == "-m" and parts[idx + 1] == "scripts.base_train":
            idx += 2
            break
        idx += 1
    return parts[idx:]


# ============================================================
# RUN CONFIGURATIONS
# ============================================================
RUNS = [
    {
        "name": "d12_65kvocab_10k_nocoldbias",
        "args": parse_cmd_string(
            '-m scripts.base_train '
            '--sparse-mode --window-pattern L --fp8 '
            '--token-cache-dir "" --token-cache-workers 8 '
            '--depth=12 --aspect-ratio 128 '
            '--total-batch-size 262144 --max-seq-len 2048 --device-batch-size=16 '
            '--num-iterations=10000 '            
            '--run \"d12 65kvoc 2kseq 16batch 8accum 128aspect 10kstep unembedlr0.01 0.001warmup 0.6warmdown 0.2embedlr 0.01matrix 0.27scalar\" '
            '--sparse-manifest manifests/65kvocab_2kseq_16batch_8accum_10kstep.json '
            '--warmup-ratio 0.001 --warmdown-ratio 0.6 '
            '--embedding-lr 0.2 --unembedding-lr=0.01 --matrix-lr 0.01 --scalar-lr 0.27 '                 
            '--sparse-cold-bias-scale 2 '            
            '--log-every 10 --eval-every 250 --core-metric-every 2500 '
        ),
    },
    # Add more runs below:
    # {
    #     "name": "d12_sparse_20k_variant",
    #     "args": parse_cmd_string(
    #         '-m scripts.base_train --depth=12 --num-iterations=20000 '
    #         '--device-batch-size=64 '
    #         '--run="sparse d12 variant" '
    #         '--sparse-mode --window-pattern L '
    #         '--sparse-manifest manifests/d6_2kseq_64batch_aspect64_20kstep_noaccum.json '
    #         '--fp8'
    #     ),
    # },
]

# ============================================================
# END OF RUN CONFIGURATIONS
# ============================================================


def build_full_command(args):
    """Build the full subprocess command list."""
    return [sys.executable, "-m", "scripts.base_train"] + args


def _process_group_alive(process_group_id):
    try:
        os.killpg(process_group_id, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _cleanup_process_group(process_group_id, *, run_name, reason, grace_seconds=5.0):
    if process_group_id is None or not _process_group_alive(process_group_id):
        return

    print(f"Cleaning up process group for '{run_name}' ({reason})")
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        return

    deadline = time.monotonic() + float(grace_seconds)
    while time.monotonic() < deadline:
        if not _process_group_alive(process_group_id):
            return
        time.sleep(0.1)

    if _process_group_alive(process_group_id):
        print(f"Escalating to SIGKILL for lingering processes in '{run_name}'")
        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            return


def run_training(name, args, run_idx, total_runs):
    """Execute a single training run."""
    cmd = build_full_command(args)

    print(f"\n{'='*60}")
    print(f"RUN [{run_idx}/{total_runs}]: {name}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()

    start_time = datetime.now()
    process = None
    process_group_id = None

    try:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )
        process_group_id = os.getpgid(process.pid)
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)
        elapsed = datetime.now() - start_time
        _cleanup_process_group(process_group_id, run_name=name, reason="post-run cleanup")
        gc.collect()
        print(f"\nCompleted successfully ({elapsed})")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = datetime.now() - start_time
        _cleanup_process_group(process_group_id, run_name=name, reason=f"exit code {e.returncode}")
        gc.collect()
        print(f"\nFailed after {elapsed} (exit code {e.returncode})")
        return False
    except KeyboardInterrupt:
        elapsed = datetime.now() - start_time
        _cleanup_process_group(process_group_id, run_name=name, reason="keyboard interrupt")
        gc.collect()
        print(f"\nInterrupted after {elapsed}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Run a sweep of training configurations sequentially."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without executing them.",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=0,
        help="Index (0-based) of the run to start from. Useful for resuming a failed sweep.",
    )
    args = parser.parse_args()

    if not RUNS:
        print("ERROR: No runs defined. Edit RUNS list in sweep_runs.py first.")
        sys.exit(1)

    print("Starting sweep")
    print(f"Total runs: {len(RUNS)}")
    if args.start_from > 0:
        print(f"Resuming from run index {args.start_from}")

    if args.dry_run:
        print("\n--- DRY RUN (commands not executed) ---\n")
        for i, run in enumerate(RUNS[args.start_from:], start=args.start_from + 1):
            name = run.get("name", f"run_{i}")
            cmd = build_full_command(run["args"])
            print(f"[{i}/{len(RUNS)}] {name}")
            print(f"  {' '.join(cmd)}\n")
        print("--- End of dry run ---")
        return

    results = {}
    total = len(RUNS)

    try:
        for i, run in enumerate(RUNS[args.start_from:], start=args.start_from + 1):
            name = run.get("name", f"run_{i}")
            success = run_training(name, run["args"], i, total)
            results[name] = success

            if not success:
                print(f"Warning: Training failed for run '{name}'")
    except KeyboardInterrupt:
        print("\nSweep interrupted by user")

    print(f"\n{'='*60}")
    print("SWEEP COMPLETED")
    print(f"{'='*60}")
    print("Results summary:")

    passed = sum(1 for s in results.values() if s)
    failed = len(results) - passed

    for name, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed} succeeded, {failed} failed out of {len(results)} runs")

    if failed > 0:
        print(f"\nTo resume from the first failed run, use:")
        first_failed_idx = next(
            i for i, (name, ok) in enumerate(results.items()) if not ok
        )
        print(f"  python {sys.argv[0]} --start-from {first_failed_idx}")


if __name__ == "__main__":
    main()