"""
Build a sparse-manifest JSON for hybrid fixed-U sparse training.

Example:
python -m scripts.build_sparse_manifest --num-iterations 2000 --grad-accum-steps 1 --output manifests/d6_sparse.json --device-batch-size 16 --total-batch-size 262144
"""

import argparse
import os
from pathlib import Path
from typing import cast

import torch

from nanochat.common import get_dist_info, print0
from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.sparse_manifest import (
    build_manifest_shard_payload,
    build_sharded_manifest_payload,
    compute_next_transition,
    save_sparse_manifest,
    tensor_ids_to_list,
)
from nanochat.tokenizer import get_tokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build sparse hybrid manifest for base pretraining")
    parser.add_argument("--output", type=str, required=True, help="output JSON path")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"], help="dataset split")
    parser.add_argument("--num-iterations", type=int, required=True, help="number of optimizer steps to precompute")
    parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="sequence length")
    parser.add_argument(
        "--total-batch-size",
        type=int,
        default=-1,
        help="total batch size in tokens (-1 = derive from grad_accum_steps or default to one micro-batch)",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=-1,
        help="gradient accumulation steps represented by the manifest (-1 = derive from total_batch_size or default to 1)",
    )
    parser.add_argument(
        "--ddp-world-size",
        type=int,
        default=1,
        help="target DDP world size represented by the manifest (single-process builder simulates all ranks)",
    )
    parser.add_argument("--tokenizer-threads", type=int, default=4, help="tokenizer worker threads")
    parser.add_argument("--tokenizer-batch-size", type=int, default=128, help="documents per tokenizer batch")
    parser.add_argument("--buffer-size", type=int, default=1000, help="best-fit document buffer size")
    parser.add_argument(
        "--token-cache-dir",
        type=str,
        default="",
        help="token cache directory (empty = sibling folder next to the dataset)",
    )
    parser.add_argument(
        "--token-cache-shard-batches",
        type=int,
        default=256,
        help="number of tokenized document batches to store per cache shard",
    )
    parser.add_argument(
        "--token-cache-workers",
        type=int,
        default=0,
        help="number of worker processes to use when building a token cache (0 = auto)",
    )
    parser.add_argument(
        "--shard-steps",
        type=int,
        default=500,
        help="number of completed optimizer steps to buffer per shard before flushing to disk",
    )
    return parser


def resolve_batch_geometry(
    *,
    device_batch_size: int,
    max_seq_len: int,
    total_batch_size: int,
    grad_accum_steps: int,
    ddp_world_size: int = 1,
) -> tuple[int, int]:
    microbatch_tokens = int(device_batch_size) * int(max_seq_len) * int(ddp_world_size)
    if microbatch_tokens <= 0:
        raise ValueError(f"microbatch token count must be positive, got {microbatch_tokens}")

    resolved_total_batch_size = int(total_batch_size)
    resolved_grad_accum_steps = int(grad_accum_steps)
    if resolved_total_batch_size < 0 and resolved_grad_accum_steps < 0:
        resolved_grad_accum_steps = 1
        resolved_total_batch_size = microbatch_tokens
    elif resolved_total_batch_size < 0:
        resolved_total_batch_size = microbatch_tokens * resolved_grad_accum_steps
    elif resolved_grad_accum_steps < 0:
        if resolved_total_batch_size % microbatch_tokens != 0:
            raise ValueError(
                f"total_batch_size {resolved_total_batch_size} must be divisible by microbatch token count {microbatch_tokens}"
            )
        resolved_grad_accum_steps = resolved_total_batch_size // microbatch_tokens

    if resolved_grad_accum_steps <= 0:
        raise ValueError(f"grad_accum_steps must be positive, got {resolved_grad_accum_steps}")
    if resolved_total_batch_size <= 0:
        raise ValueError(f"total_batch_size must be positive, got {resolved_total_batch_size}")
    if resolved_total_batch_size % microbatch_tokens != 0:
        raise ValueError(
            f"total_batch_size {resolved_total_batch_size} must be divisible by microbatch token count {microbatch_tokens}"
        )
    derived_grad_accum_steps = resolved_total_batch_size // microbatch_tokens
    if resolved_grad_accum_steps != derived_grad_accum_steps:
        raise ValueError(
            f"grad_accum_steps mismatch: total_batch_size={resolved_total_batch_size} and microbatch={microbatch_tokens} imply {derived_grad_accum_steps}, got {resolved_grad_accum_steps}"
        )
    return resolved_total_batch_size, resolved_grad_accum_steps


def main() -> None:
    args = build_parser().parse_args()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    del ddp_rank, ddp_local_rank, ddp_world_size
    if ddp:
        raise ValueError("Sparse manifest builder is single-process only; run it without torchrun")
    target_ddp_world_size = int(args.ddp_world_size)
    if target_ddp_world_size <= 0:
        raise ValueError(f"--ddp-world-size must be positive, got {target_ddp_world_size}")
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    total_batch_size, grad_accum_steps = resolve_batch_geometry(
        device_batch_size=args.device_batch_size,
        max_seq_len=args.max_seq_len,
        total_batch_size=args.total_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        ddp_world_size=target_ddp_world_size,
    )

    resolved_token_cache_workers = max(1, args.token_cache_workers) if args.token_cache_workers > 0 else max(1, min(8, os.cpu_count() or 1))
    loaders = [
        tokenizing_distributed_data_loader_with_state_bos_bestfit(
            tokenizer,
            args.device_batch_size,
            args.max_seq_len,
            split=args.split,
            tokenizer_threads=args.tokenizer_threads,
            tokenizer_batch_size=args.tokenizer_batch_size,
            device="cpu",
            resume_state_dict=None,
            buffer_size=args.buffer_size,
            return_sequence_recipe=False,
            vocab_size=vocab_size,
            token_cache_dir=args.token_cache_dir,
            token_cache_shard_batches=args.token_cache_shard_batches,
            token_cache_workers=resolved_token_cache_workers,
            ddp_rank_override=rank,
            ddp_world_size_override=target_ddp_world_size,
        )
        for rank in range(target_ddp_world_size)
    ]

    print0(
        f"Building sparse manifest for {args.num_iterations:,} steps | "
        f"world={target_ddp_world_size} B={args.device_batch_size} T={args.max_seq_len} total_batch={total_batch_size:,}"
    )

    if args.shard_steps <= 0:
        raise ValueError(f"--shard-steps must be positive, got {args.shard_steps}")

    output_path = Path(args.output)
    shard_dir = output_path.parent / f"{output_path.stem}_shards"
    shard_steps: list[dict] = []
    shard_entries: list[dict] = []
    shard_index = 0
    global_u_max = 0
    global_grad_accum_u_max = 0

    def flush_shard() -> None:
        nonlocal shard_index, shard_steps, global_u_max, global_grad_accum_u_max
        if not shard_steps:
            return
        shard_start_step = int(shard_steps[0]["step"])
        shard_payload = build_manifest_shard_payload(
            shard_index=shard_index,
            start_step=shard_start_step,
            steps=shard_steps,
        )
        shard_filename = f"{output_path.stem}.shard{shard_index:05d}.json"
        shard_path = shard_dir / shard_filename
        save_sparse_manifest(shard_path, shard_payload)
        shard_entries.append({
            "shard_index": shard_index,
            "path": str(shard_path.relative_to(output_path.parent)),
            "start_step": shard_start_step,
            "num_steps": int(shard_payload["num_steps"]),
            "u_max": int(shard_payload["u_max"]),
            "grad_accum_u_max": int(shard_payload["grad_accum_u_max"]),
        })
        global_u_max = max(global_u_max, int(shard_payload["u_max"]))
        global_grad_accum_u_max = max(global_grad_accum_u_max, int(shard_payload["grad_accum_u_max"]))
        print0(
            f"  wrote shard {shard_index + 1:,}: steps {shard_start_step:,}-"
            f"{shard_start_step + int(shard_payload['num_steps']) - 1:,}"
        )
        shard_index += 1
        shard_steps = []

    def append_completed_step(step_entry: dict) -> None:
        shard_steps.append(step_entry)
        if len(shard_steps) >= args.shard_steps:
            flush_shard()

    previous_active_ids = None
    previous_microstep_entry = None
    pending_step_entry = None
    for step_idx in range(args.num_iterations):
        microsteps: list[dict] = []
        active_ids_list = []
        for micro_idx in range(grad_accum_steps):
            rank_active_ids: list[torch.Tensor] = []
            for loader in loaders:
                batch = next(loader)
                inputs_cpu = cast(torch.Tensor, batch[0]).to(device="cpu")
                targets_cpu = cast(torch.Tensor, batch[1]).to(device="cpu")
                state_dict = dict(cast(dict, batch[2]))
                local_active_ids_cpu = torch.unique(torch.cat((inputs_cpu.reshape(-1), targets_cpu.reshape(-1))), sorted=True)
                rank_active_ids.append(local_active_ids_cpu)
            active_ids_cpu = torch.unique(torch.cat(rank_active_ids), sorted=True)
            active_ids_list.append(active_ids_cpu)
            if previous_active_ids is not None and previous_microstep_entry is not None:
                next_common_ids, next_leaving_ids, next_new_ids = compute_next_transition(previous_active_ids, active_ids_cpu)
                previous_microstep_entry["next_common_ids"] = tensor_ids_to_list(next_common_ids)
                previous_microstep_entry["next_leaving_ids"] = tensor_ids_to_list(next_leaving_ids)
                previous_microstep_entry["next_new_ids"] = tensor_ids_to_list(next_new_ids)
            microstep_entry = {
                "microstep": micro_idx,
                "u_size": int(active_ids_cpu.numel()),
                "active_ids": tensor_ids_to_list(active_ids_cpu),
                "next_common_ids": [],
                "next_leaving_ids": [],
                "next_new_ids": [],
            }
            microsteps.append(microstep_entry)
            previous_active_ids = active_ids_cpu
            previous_microstep_entry = microstep_entry

        grad_accum_ids_cpu = torch.unique(torch.cat(active_ids_list), sorted=True)
        current_step_entry = {
            "step": step_idx,
            "grad_accum_u_size": int(grad_accum_ids_cpu.numel()),
            "grad_accum_active_ids": tensor_ids_to_list(grad_accum_ids_cpu),
            "microsteps": microsteps,
        }
        if pending_step_entry is not None:
            append_completed_step(pending_step_entry)
        pending_step_entry = current_step_entry
        if (step_idx + 1) % 100 == 0 or step_idx + 1 == args.num_iterations:
            print0(f"  processed {step_idx + 1:,}/{args.num_iterations:,} steps")

    if pending_step_entry is not None:
        append_completed_step(pending_step_entry)
    flush_shard()
    payload = build_sharded_manifest_payload(
        split=args.split,
        vocab_size=vocab_size,
        device_batch_size=args.device_batch_size,
        max_seq_len=args.max_seq_len,
        total_batch_size=total_batch_size,
        grad_accum_steps=grad_accum_steps,
        ddp_world_size=target_ddp_world_size,
        num_iterations=args.num_iterations,
        buffer_size=args.buffer_size,
        u_max=global_u_max,
        grad_accum_u_max=global_grad_accum_u_max,
        shard_step_count=args.shard_steps,
        shards=shard_entries,
    )
    save_sparse_manifest(args.output, payload)
    print0(
        f"Saved sparse manifest to {args.output} | u_max={payload['u_max']:,} | "
        f"grad_accum_u_max={payload.get('grad_accum_u_max', payload['u_max']):,} | "
        f"steps={payload['num_steps']:,} | grad_accum_steps={grad_accum_steps:,} | "
        f"shards={payload['num_shards']:,}"
    )

if __name__ == "__main__":
    main()
