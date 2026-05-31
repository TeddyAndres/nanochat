#!/usr/bin/env python3
"""Minimal reproducible timing harness for sparse fixed-U CPU offload work.

Run with:
    python -m dev.sparse_cpu_offload_bench --steps 5 --debug

It exercises the DynamicVocabRuntime (fixed-U + grad accum + overlap) path
and prints the key sparse_prep_* counters that the plan cares about
(prep_persistent_slot_ms, boundary overhead, etc.).

This is the canonical script referenced by the plan for Phase 0 baselines
and Phase 5 before/after claims. Extend it with real manifest loading when
a small canned manifest is available.
"""
import argparse
import time
import torch

from nanochat.dynamic_vocab import DynamicVocabRuntime
from nanochat.gpt import GPT, GPTConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3, help="Number of optimizer steps (windows)")
    parser.add_argument("--grad-accum", type=int, default=2, help="grad_accum_steps")
    parser.add_argument("--debug", action="store_true", help="Enable NANOCHAT_SPARSE_DEBUG=1 for the guard")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    if args.debug:
        import os
        os.environ["NANOCHAT_SPARSE_DEBUG"] = "1"

    torch.manual_seed(0)
    config = GPTConfig(
        sequence_len=4,
        vocab_size=128,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=16,
        window_pattern="L",
    )
    model = GPT(config)
    model.init_weights()

    runtime = DynamicVocabRuntime(
        model,
        device=args.device,
        embedding_lr=0.01,
        value_embedding_lr=0.01,
        unembedding_lr=0.01,
        fixed_u_max=8,
        lm_head_u_max=8,
        grad_accum_u_max=8,
        adam_betas=(0.9, 0.95),
        weight_decay=0.0,
    )
    runtime.disable_fixed_overlap_reuse = False

    print(f"Running {args.steps} windows with grad_accum_steps={args.grad_accum} on {args.device}...")

    # Tiny synthetic loop mimicking the training script's prepare/prefetch/apply cycle
    window_ids = [0, 1, 2, 3]
    for step in range(args.steps):
        for micro in range(args.grad_accum):
            meta = {
                "active_ids_cpu": torch.tensor(window_ids, dtype=torch.long),
                "active_slot_ids_cpu": torch.arange(len(window_ids), dtype=torch.long),
                "active_mask_cpu": torch.ones(len(window_ids), dtype=torch.bool),
                "slot_to_global_cpu": torch.tensor(window_ids, dtype=torch.long),
                "stage_ids_cpu": torch.tensor(window_ids, dtype=torch.long),
                "stage_slot_ids_cpu": torch.arange(len(window_ids), dtype=torch.long),
                "writeback_ids_cpu": torch.empty(0, dtype=torch.long),
                "writeback_slot_ids_cpu": torch.empty(0, dtype=torch.long),
                "grad_accum_ids_cpu": torch.tensor(window_ids, dtype=torch.long),
                "grad_accum_steps": args.grad_accum,
                "grad_accum_micro_step": micro,
                "is_grad_accum_boundary": micro == args.grad_accum - 1,
                "is_last_step": (step == args.steps - 1) and (micro == args.grad_accum - 1),
                "inputs_union_cpu_local": torch.tensor([[0, 1], [2, 0]], dtype=torch.long),
                "targets_union_cpu_local": torch.tensor([[1, 2], [3, 1]], dtype=torch.long),
            }

            if micro == 0 and step > 0:
                # Simulate the training loop prefetch of the *next* meta
                runtime.prefetch_step(meta)

            t0 = time.perf_counter()
            ctx = runtime.prepare_step(meta)
            prep_ms = (time.perf_counter() - t0) * 1000.0

            # Minimal forward/backward/accum/apply to keep state alive.
            # On CUDA we skip the actual model forward in this synthetic bench
            # (device handling for rotary + union tensors is done properly in
            # the real training loop via stage_batch_to_device). The goal here
            # is to exercise and time the prepare/prefetch/persistent-slot path.
            if args.device == "cpu":
                x = ctx.union_inputs
                y = ctx.union_targets
                loss = runtime.model(x, y, active_vocab=ctx.active_vocab)
                loss.backward()
                runtime.accumulate_gradients(ctx)
                if ctx.is_grad_accum_boundary:
                    runtime.apply_accumulated_gradients()
                    runtime.model.zero_grad(set_to_none=True)
            else:
                # Still need to call accumulate/apply for state machine correctness
                # even if we skip the real forward on CUDA in this tiny bench.
                runtime.accumulate_gradients(ctx)
                if ctx.is_grad_accum_boundary:
                    runtime.apply_accumulated_gradients()

        # Accumulate a couple of the interesting counters
        print(f"step {step}: prep_total~{prep_ms:.2f}ms  "
              f"persistent_slot={getattr(ctx, 'prep_persistent_slot_ms', 0):.2f}ms  "
              f"boundary_overhead~{getattr(ctx, 'prep_active_vocab_build_ms', 0):.3f}ms")

    print("\nDone. Key counters (last ctx):")
    for attr in ("prep_persistent_slot_ms", "prep_cpu_reuse_map_ms", "prep_active_vocab_build_ms",
                 "prep_union_io_h2d_ms", "prep_writeback_wait_ms"):
        print(f"  {attr}: {getattr(ctx, attr, 'n/a')}")


if __name__ == "__main__":
    main()
