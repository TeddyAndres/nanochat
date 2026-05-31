#!/usr/bin/env python3
"""
Minimal LLaDA-style diffusion training entrypoint for nanochat.

This is the very first skeleton on the feature/dynamic-vocab-diffusion branch.
It demonstrates the core LLaDA forward process + weighted masked loss
(using the new nanochat.diffusion module) without yet modifying the
autoregressive paths in gpt.py or base_train.py.

Usage (with the REQUIRED .venv-5090):
    source /home/teddy/Desktop/dev/repo/nanochat/.venv-5090/bin/activate
    python -m scripts.diffusion_train --llada-mode --depth 4 --num-iterations 50 ...

Later we will:
- Wire the real model forward (after adding bidirectional support)
- Add --sparse-mode + manifest support (exact same contract as base_train)
- Add the SFT path (prompt kept clean)
- Implement the low-confidence remasking sampler

For now this file exists so the branch has a clean first commit that only
adds new files and proves the LLaDA loss math can be imported and executed.
"""

import argparse
import torch

from nanochat.diffusion import forward_process, compute_llada_loss, DEFAULT_MASK_ID


def main():
    parser = argparse.ArgumentParser(description="LLaDA-style diffusion pretraining skeleton")
    parser.add_argument("--llada-mode", action="store_true", help="Enable LLaDA masked diffusion objective (skeleton)")
    parser.add_argument("--depth", type=int, default=4, help="Tiny model depth for skeleton testing")
    parser.add_argument("--num-iterations", type=int, default=20, help="How many dummy steps to run")
    parser.add_argument("--device-batch-size", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=128)
    args = parser.parse_args()

    if not args.llada_mode:
        print("This skeleton currently only supports --llada-mode. Exiting.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running LLaDA skeleton on {device}")
    print(f"Using MASK_ID = {DEFAULT_MASK_ID}")

    # Dummy "model" that just produces random logits over a tiny vocab for testing the loss path.
    # Real version will call the actual bidirectional GPT once we add the causal=False path.
    vocab_size = 1024  # tiny for skeleton
    dummy_linear = torch.nn.Linear(64, vocab_size, bias=False).to(device)

    for step in range(args.num_iterations):
        # Simulate a clean batch of token ids (in real code this comes from the dataloader)
        input_ids = torch.randint(0, vocab_size, (args.device_batch_size, args.max_seq_len), device=device)

        # === LLaDA forward process (exact) ===
        noisy_batch, masked_indices, p_mask = forward_process(input_ids)

        # === Dummy forward (will be replaced by real model(x_t) later) ===
        # For now we just embed the noisy tokens with a tiny dummy projection
        # so we can exercise the loss math end-to-end.
        dummy_hidden = torch.randn(args.device_batch_size, args.max_seq_len, 64, device=device)
        logits = dummy_linear(dummy_hidden)  # (b, l, V)

        # === LLaDA loss (exact) ===
        loss = compute_llada_loss(logits, input_ids, masked_indices, p_mask)

        if step % 5 == 0 or step == args.num_iterations - 1:
            num_masked = masked_indices.sum().item()
            avg_p = p_mask[masked_indices].mean().item() if num_masked > 0 else 0.0
            print(f"step {step:03d} | loss {loss.item():.4f} | masked {num_masked} | avg_p_mask {avg_p:.3f}")

    print("\nLLaDA skeleton finished successfully.")
    print("Next steps (see approved plan):")
    print("  1. Add bidirectional (causal=False) support in gpt.py")
    print("  2. Wire the real model forward + sparse active_vocab path")
    print("  3. Implement low-confidence remasking sampler in nanochat/diffusion.py")


if __name__ == "__main__":
    main()
