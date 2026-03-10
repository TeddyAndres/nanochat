"""
Build a sparse-manifest JSON for hybrid fixed-U sparse training.

Example:
python -m scripts.build_sparse_manifest --num-iterations 2000 --output manifests/d6_sparse.json
"""

import argparse

import torch

from nanochat.common import get_dist_info, print0
from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit_dynamic
from nanochat.gpt import GPT, GPTConfig
from nanochat.sparse_manifest import build_manifest_payload, compute_next_transition, save_sparse_manifest, tensor_ids_to_list
from nanochat.tokenizer import get_tokenizer


parser = argparse.ArgumentParser(description="Build sparse hybrid manifest for base pretraining")
parser.add_argument("--output", type=str, required=True, help="output JSON path")
parser.add_argument("--split", type=str, default="train", choices=["train", "val"], help="dataset split")
parser.add_argument("--num-iterations", type=int, required=True, help="number of optimizer steps to precompute")
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size")
parser.add_argument("--max-seq-len", type=int, default=2048, help="sequence length")
parser.add_argument("--total-batch-size", type=int, default=-1, help="total batch size in tokens")
parser.add_argument("--grad-accum-steps", type=int, default=-1, help="gradient accumulation steps represented by the manifest (-1 = derive from total batch size)")
parser.add_argument("--depth", type=int, default=20, help="model depth used for auto batch-size calculation")
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio for auto batch-size calculation")
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for auto batch-size calculation")
parser.add_argument("--target-param-data-ratio", type=float, default=10.5, help="target data:param ratio for auto batch-size calculation")
parser.add_argument("--tokenizer-threads", type=int, default=4, help="tokenizer worker threads")
parser.add_argument("--tokenizer-batch-size", type=int, default=128, help="documents per tokenizer batch")
parser.add_argument("--buffer-size", type=int, default=1000, help="best-fit document buffer size")
args = parser.parse_args()


def build_model_meta(depth: int, vocab_size: int):
    base_dim = depth * args.aspect_ratio
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    num_heads = model_dim // args.head_dim
    config = GPTConfig(
        sequence_len=args.max_seq_len,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern="L",
    )
    with torch.device("meta"):
        return GPT(config)


def get_scaling_params(model) -> int:
    param_counts = model.num_scaling_params()
    return param_counts["transformer_matrices"] + param_counts["lm_head"]


def main() -> None:
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if ddp:
        raise ValueError("Sparse manifest builder is single-process only; run it without torchrun")
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    microbatch_tokens = args.device_batch_size * args.max_seq_len
    total_batch_size = args.total_batch_size
    grad_accum_steps = args.grad_accum_steps
    if total_batch_size < 0:
        if grad_accum_steps > 0:
            total_batch_size = microbatch_tokens * grad_accum_steps
        else:
            d_ref = build_model_meta(12, vocab_size)
            model = build_model_meta(args.depth, vocab_size)
            target_tokens = int(args.target_param_data_ratio * get_scaling_params(model))
            d_ref_tokens = args.target_param_data_ratio * get_scaling_params(d_ref)
            b_ref = 2**19
            batch_size_ratio = target_tokens / d_ref_tokens
            predicted_batch_size = b_ref * batch_size_ratio ** 0.383
            total_batch_size = 2 ** round(torch.log2(torch.tensor(float(predicted_batch_size))).item())
    if total_batch_size % microbatch_tokens != 0:
        raise ValueError(
            f"total_batch_size {total_batch_size} must be divisible by microbatch token count {microbatch_tokens}"
        )
    derived_grad_accum_steps = total_batch_size // microbatch_tokens
    if grad_accum_steps < 0:
        grad_accum_steps = derived_grad_accum_steps
    elif grad_accum_steps != derived_grad_accum_steps:
        raise ValueError(
            f"grad_accum_steps mismatch: total_batch_size={total_batch_size} and microbatch={microbatch_tokens} imply {derived_grad_accum_steps}, got {grad_accum_steps}"
        )

    loader = tokenizing_distributed_data_loader_with_state_bos_bestfit_dynamic(
        tokenizer,
        args.device_batch_size,
        args.max_seq_len,
        split=args.split,
        tokenizer_threads=args.tokenizer_threads,
        tokenizer_batch_size=args.tokenizer_batch_size,
        device="cpu",
        resume_state_dict=None,
        buffer_size=args.buffer_size,
        vocab_size=vocab_size,
    )

    print0(
        f"Building sparse manifest for {args.num_iterations:,} steps | "
        f"B={args.device_batch_size} T={args.max_seq_len} total_batch={total_batch_size:,}"
    )

    steps: list[dict] = []
    previous_active_ids = None
    previous_microstep_entry = None
    for step_idx in range(args.num_iterations):
        microsteps: list[dict] = []
        active_ids_list = []
        for micro_idx in range(grad_accum_steps):
            _, _, active_ids_cpu, _ = next(loader)
            active_ids_cpu = active_ids_cpu.to(device="cpu")
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
        steps.append({
            "step": step_idx,
            "grad_accum_u_size": int(grad_accum_ids_cpu.numel()),
            "grad_accum_active_ids": tensor_ids_to_list(grad_accum_ids_cpu),
            "microsteps": microsteps,
        })
        if (step_idx + 1) % 100 == 0 or step_idx + 1 == args.num_iterations:
            print0(f"  processed {step_idx + 1:,}/{args.num_iterations:,} steps")

    payload = build_manifest_payload(
        split=args.split,
        vocab_size=vocab_size,
        device_batch_size=args.device_batch_size,
        max_seq_len=args.max_seq_len,
        total_batch_size=total_batch_size,
        grad_accum_steps=grad_accum_steps,
        ddp_world_size=ddp_world_size,
        num_iterations=args.num_iterations,
        tokenizer_batch_size=args.tokenizer_batch_size,
        tokenizer_threads=args.tokenizer_threads,
        buffer_size=args.buffer_size,
        steps=steps,
    )
    save_sparse_manifest(args.output, payload)
    print0(
        f"Saved sparse manifest to {args.output} | u_max={payload['u_max']:,} | steps={payload['num_steps']:,} | grad_accum_steps={grad_accum_steps:,}"
    )


if __name__ == "__main__":
    main()
