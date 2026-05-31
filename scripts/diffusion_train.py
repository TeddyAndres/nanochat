#!/usr/bin/env python3
"""
LLaDA-style diffusion pretraining entrypoint for nanochat (dense only for now).

This demonstrates the full LLaDA masked diffusion objective using a real
bidirectional GPT (causal=False) + the exact forward_process + weighted
masked loss from the LLaDA paper.

Usage (MANDATORY: use .venv-5090):
    source /home/teddy/Desktop/dev/repo/nanochat/.venv-5090/bin/activate
    python -m scripts.diffusion_train --llada-mode --depth 4 --num-iterations 100 ...

All Python execution on this machine for this project must go through .venv-5090.
"""

import argparse
import torch

from nanochat.gpt import GPT, GPTConfig
import nanochat.diffusion as diffusion
from nanochat.diffusion import forward_process, compute_llada_loss, DEFAULT_MASK_ID
from nanochat.common import print0


def main():
    parser = argparse.ArgumentParser(description="LLaDA-style diffusion pretraining (dense)")
    parser.add_argument("--llada-mode", action="store_true", help="Enable LLaDA masked diffusion objective")
    parser.add_argument("--depth", type=int, default=4, help="Model depth (number of layers)")
    parser.add_argument("--num-iterations", type=int, default=100, help="Number of training steps")
    parser.add_argument("--device-batch-size", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=1024, help="Tiny vocab for fast skeleton runs")
    args = parser.parse_args()

    if not args.llada_mode:
        print("This entrypoint currently requires --llada-mode. Exiting.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print0(f"Running LLaDA-style dense pretrain on {device} using .venv-5090")

    # Choose a mask id that fits in the current vocab.
    # In a real run this would be a properly reserved token in the tokenizer.
    mask_id = args.vocab_size - 1
    print0(f"Using MASK_ID = {mask_id} (fits in vocab_size={args.vocab_size})")

    # Create a tiny GPT config (same style as base_train)
    config = GPTConfig(
        sequence_len=args.max_seq_len,
        vocab_size=args.vocab_size,
        n_layer=args.depth,
        n_head=4,
        n_kv_head=4,
        n_embd=128,
        window_pattern="L",   # full context everywhere for diffusion
    )

    model = GPT(config).to(device)
    model.eval()  # no dropout etc. for this skeleton

    # Optimizer for the tiny model (just to make it a real training loop)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for step in range(args.num_iterations):
        # Simulate clean token batch (real version will come from dataloader + manifest)
        input_ids = torch.randint(0, args.vocab_size, (args.device_batch_size, args.max_seq_len), device=device)

        # === Exact LLaDA forward (noising) process ===
        noisy_batch, masked_indices, p_mask = forward_process(input_ids, mask_id=mask_id)

        # Demonstration: how sparse code should compute required tokens for U
        required_for_sparse = diffusion.get_tokens_for_active_vocab(input_ids, mask_id)
        # In a real sparse run you would union this with any other always-hot tokens
        # and feed it into DynamicVocabRuntime / active_vocab construction.

        if step == 0:
            print0(f"  [sparse-compat] tokens that would be required for active_vocab: {required_for_sparse.tolist()}")

        # === Real model forward with causal=False (bidirectional) ===
        # This is the key line that exercises the new diffusion path we just added.
        logits = model(noisy_batch, causal=False)  # (B, T, V) — note: no targets, just features + lm_head

        # === Exact LLaDA weighted masked loss ===
        loss = compute_llada_loss(logits, input_ids, masked_indices, p_mask)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 10 == 0 or step == args.num_iterations - 1:
            num_masked = int(masked_indices.sum().item())
            avg_p = float(p_mask[masked_indices].mean().item()) if num_masked > 0 else 0.0
            print0(f"step {step:04d} | loss {loss.item():.4f} | masked {num_masked:5d} | avg_p {avg_p:.3f}")

    print0("\nLLaDA dense skeleton run completed successfully.")
    print0("We just ran real bidirectional (causal=False) GPT + exact LLaDA loss.")
    print0("Next (per plan): wire --sparse-mode + real dataloader, then low-conf remasking sampler.")


if __name__ == "__main__":
    main()
