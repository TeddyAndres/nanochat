#!/usr/bin/env python3
"""
LLaDA-style diffusion pretraining entrypoint for nanochat (with sparse support).

This demonstrates the full LLaDA masked diffusion objective using a real
bidirectional GPT (causal=False) + the exact forward_process + weighted
masked loss from the LLaDA paper, with optional DynamicVocabRuntime integration.

Usage (MANDATORY: use .venv-5090 on this machine):
    source /home/teddy/Desktop/dev/repo/nanochat/.venv-5090/bin/activate

Example for d6 experimentation with the exact 65k manifest dimensions (2048 seq, 16 batch, 1 accum):

    # Recommended test command using token ID 44241 (64 dashes) as temporary MASK.
    # 44241 is a very rare/synthetic token in the current 65k tokenizer and is safe
    # to repurpose for short diffusion experiments without retraining the tokenizer.
    #
    # --token-cache-dir is optional. If omitted, it uses the same default as base_train.py
    # (sibling folder next to the dataset, e.g. nanochat_train_token_cache_v3).
    python -m scripts.diffusion_train \
        --llada-mode \
        --depth 6 \
        --num-iterations 5 \          # cut down aggressively from 200k
        --sparse-mode \
        --sparse-manifest manifests/65k_2kseq_16batch_1accum_200ksteps.json \
        --mask-id 44241
        # (omit --token-cache-dir to use the same auto-default as base_train.py)

The script will automatically force the manifest's exact dimensions:
  device_batch_size=16, max_seq_len=2048, vocab_size=65536

All Python execution on this machine for this project must go through .venv-5090.

MASK token contract (important):
  - MASK is **not** stored in manifests.
  - It is treated as a diffusion-level always-hot token (always forced into the active set).
  - For this 65k manifest, use --mask-id 44241 during testing (the 64-dash token).
    For production you should retrain the tokenizer with a proper reserved <|mask|> token.
"""

import argparse
import torch

from nanochat.gpt import GPT, GPTConfig
import nanochat.diffusion as diffusion
from nanochat.diffusion import forward_process, compute_llada_loss, DEFAULT_MASK_ID
from nanochat.common import print0

# Sparse + dataloader imports (modeled on base_train.py)
from nanochat.dynamic_vocab import DynamicVocabRuntime
from nanochat.sparse_manifest import (
    load_sparse_manifest_header,
    resolve_sparse_manifest_grad_accum_u_max,
)
from nanochat.dataloader import (
    tokenizing_distributed_data_loader_with_state_bos_bestfit_manifest,
)
from nanochat.prefetch import AsyncLoaderPrefetcher
from nanochat.tokenizer import get_tokenizer


def main():
    parser = argparse.ArgumentParser(description="LLaDA-style diffusion pretraining (dense)")
    parser.add_argument("--llada-mode", action="store_true", help="Enable LLaDA masked diffusion objective")
    parser.add_argument("--depth", type=int, default=6, help="Model depth (number of layers)")
    parser.add_argument("--num-iterations", type=int, default=100, help="Number of training steps")
    parser.add_argument("--device-batch-size", type=int, default=16, help="Per-device batch size. Overridden by --sparse-manifest if provided.")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Sequence length. Overridden by --sparse-manifest if provided.")
    parser.add_argument("--vocab-size", type=int, default=65536, help="Vocab size. Overridden by --sparse-manifest (use 65536 for the 65k manifests).")
    parser.add_argument("--mask-id", type=int, default=44241, help="Token id to use as [MASK] for diffusion (required when using real 65k manifests).")

    # Token cache arguments (passed through to the manifest dataloader)
    parser.add_argument(
        "--token-cache-dir",
        type=str,
        default="",
        help="token cache directory (empty = sibling folder next to the dataset, same default as base_train.py)",
    )
    parser.add_argument("--token-cache-shard-batches", type=int, default=256)
    parser.add_argument("--token-cache-workers", type=int, default=1)

    # Sparse / dynamic vocab flags (subset of base_train.py for now)
    parser.add_argument("--sparse-mode", action="store_true", help="Enable dynamic/sparse vocabulary using DynamicVocabRuntime")
    parser.add_argument("--sparse-manifest", type=str, default="", help="Path to sparse manifest JSON (required for hybrid sparse mode)")
    parser.add_argument("--sparse-logit-scale", type=float, default=1.0, help="Multiply sparse logits by this factor before loss")
    args = parser.parse_args()

    if not args.llada_mode:
        print("This entrypoint currently requires --llada-mode. Exiting.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print0(f"Running LLaDA-style diffusion on {device} using .venv-5090")

    # ------------------------------------------------------------------
    # If a sparse manifest is provided, it dictates the exact geometry.
    # This is mandatory for the 65k manifests the user is targeting.
    # ------------------------------------------------------------------
    manifest_geometry = None
    if args.sparse_manifest:
        manifest = load_sparse_manifest_header(args.sparse_manifest)
        manifest_B = int(manifest.get("device_batch_size", 0))
        manifest_T = int(manifest.get("max_seq_len", 0))
        manifest_vocab = int(manifest.get("vocab_size", 0))
        manifest_grad_accum = int(manifest.get("grad_accum_steps", 1))

        print0(f"Manifest {args.sparse_manifest} dictates exact dimensions:")
        print0(f"  device_batch_size = {manifest_B}")
        print0(f"  max_seq_len       = {manifest_T}")
        print0(f"  vocab_size        = {manifest_vocab}")
        print0(f"  grad_accum_steps  = {manifest_grad_accum}")

        # Force the values (with loud notice)
        if args.device_batch_size != manifest_B:
            print0(f"  WARNING: Overriding --device-batch-size {args.device_batch_size} -> {manifest_B} (required by manifest)")
            args.device_batch_size = manifest_B
        if args.max_seq_len != manifest_T:
            print0(f"  WARNING: Overriding --max-seq-len {args.max_seq_len} -> {manifest_T} (required by manifest)")
            args.max_seq_len = manifest_T
        if args.vocab_size != manifest_vocab:
            print0(f"  WARNING: Overriding --vocab-size {args.vocab_size} -> {manifest_vocab} (required by manifest)")
            args.vocab_size = manifest_vocab

        manifest_geometry = {
            "batch_size": manifest_B,
            "seq_len": manifest_T,
            "vocab_size": manifest_vocab,
            "grad_accum_steps": manifest_grad_accum,
        }

    # =====================================================================
    # MASK token handling (diffusion + sparse contract)
    # =====================================================================
    # MASK is treated as a diffusion-specific "always-hot" token.
    # It is never baked into manifests. It is always unioned at runtime.
    #
    # For testing with the current 65k tokenizer + manifest, we are using
    # token ID 44241 (a 64-dash sequence) as a temporary MASK. This is a
    # very rare/synthetic token and safe to repurpose for short experiments.
    #
    # For any real 65k-vocab manifest run you *must* supply --mask-id with
    # the actual reserved token id from your tokenizer.
    if args.mask_id is None:
        if args.vocab_size >= 20000:   # heuristic: real large-vocab manifest run
            raise ValueError(
                "--mask-id is **required** when using --llada-mode + --sparse-mode "
                "with a real large manifest (65k vocab etc.).\n"
                "You must provide the exact reserved MASK token id that exists "
                "in the tokenizer used to build the manifest."
            )
        mask_id = args.vocab_size - 1
        print0(f"WARNING: No --mask-id given. Using last token ({mask_id}) as MASK. "
               "This is only acceptable for toy experiments.")
    else:
        mask_id = args.mask_id

    print0(f"Using MASK_ID = {mask_id} (vocab_size={args.vocab_size})")
    print0("MASK will be unconditionally included in every active set for the sparse runtime.")

    # Critical validation: the mask token must actually exist in the vocabulary.
    # This catches the common mistake of using LLaDA's default 126336 on a 65k vocab.
    # Current testing choice for this 65k setup: 44241 (the 64-dash token).
    if mask_id >= args.vocab_size or mask_id < 0:
        raise ValueError(
            f"--mask-id {mask_id} is invalid for vocab_size={args.vocab_size}.\n"
            f"The token ID must satisfy 0 <= mask_id < {args.vocab_size}.\n\n"
            "LLaDA's default 126336 will **not** work on a 65k (65536) vocabulary.\n"
            "You must choose (or reserve) a valid MASK token ID that actually exists "
            "in the tokenizer used with this manifest."
        )

    # Create GPT config using dimensions that are now guaranteed to match the manifest
    # (when a manifest was supplied). For the 65k manifest this means 65536 vocab + 2048 seq.
    config = GPTConfig(
        sequence_len=args.max_seq_len,
        vocab_size=args.vocab_size,
        n_layer=args.depth,
        n_head=4,
        n_kv_head=4,
        n_embd=128,                 # small width for d6 experimentation
        window_pattern="L",         # full bidirectional context for diffusion
    )

    print0(f"Creating GPT (depth={args.depth}, vocab={args.vocab_size}, seq={args.max_seq_len})")
    model = GPT(config).to(device)
    model.eval()  # no dropout etc. for this skeleton

    # Optimizer for the tiny model (just to make it a real training loop)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # -------------------------------------------------------------------------
    # Sparse / Dynamic Vocab Runtime (option A wiring)
    # -------------------------------------------------------------------------
    dynamic_vocab = None
    if args.sparse_mode:
        print0("Sparse mode enabled for LLaDA diffusion objective")
        fixed_u_max = 0
        grad_accum_u_max = 0
        if args.sparse_manifest:
            sparse_manifest = load_sparse_manifest_header(args.sparse_manifest)
            fixed_u_max = int(sparse_manifest.get("u_max", 0))
            try:
                grad_accum_u_max = resolve_sparse_manifest_grad_accum_u_max(args.sparse_manifest, sparse_manifest)
            except Exception:
                grad_accum_u_max = fixed_u_max
            print0(f"  Using manifest: {args.sparse_manifest} (u_max={fixed_u_max}, grad_accum_u_max={grad_accum_u_max})")
        else:
            print0("  Using dynamic (on-the-fly) U discovery (no manifest)")

        dynamic_vocab = DynamicVocabRuntime(
            model,
            device=device,
            embedding_lr=0.2,
            unembedding_lr=0.005,
            value_embedding_lr=0.2,
            fixed_u_max=fixed_u_max if fixed_u_max > 0 else None,
            lm_head_u_max=fixed_u_max if fixed_u_max > 0 else None,
            # For fixed-U manifest mode the runtime requires grad_accum_u_max >= fixed_u_max
            grad_accum_u_max=grad_accum_u_max if grad_accum_u_max > 0 else fixed_u_max,
            capacity_round_multiple=1,
            adam_betas=(0.8, 0.95),
            weight_decay=0.0,
        )
        print0(f"  DynamicVocabRuntime initialized (fixed_u_mode={bool(fixed_u_max)})")
        print0("  MASK will be force-included in every prepare_step() call (diffusion contract).")

    # -------------------------------------------------------------------------
    # Real data loader from manifest (when provided) - required for 65k vocab manifests
    # -------------------------------------------------------------------------
    train_iter = None
    using_real_loader = False
    if args.sparse_manifest:
        print0("Setting up real manifest-driven dataloader for 65k vocab data...")
        try:
            tokenizer = get_tokenizer()

            # The manifest loader is strict about geometry — we already forced the right values above.
            loader = tokenizing_distributed_data_loader_with_state_bos_bestfit_manifest(
                tokenizer,
                args.device_batch_size,
                args.max_seq_len,
                split="train",
                manifest_path=args.sparse_manifest,
                device="cpu",
                pin_memory_output=(device.type == "cuda"),
                vocab_size=args.vocab_size,
                token_cache_dir=args.token_cache_dir,
                token_cache_shard_batches=args.token_cache_shard_batches,
                token_cache_workers=args.token_cache_workers,
            )
            train_iter = AsyncLoaderPrefetcher(loader, max_prefetch=2)
            using_real_loader = True
            print0("  Real manifest dataloader + prefetcher ready. Using pre-tokenized 65k data.")
        except Exception as e:
            print0(f"  ERROR: Failed to initialize real manifest dataloader: {e}")
            effective_cache = args.token_cache_dir or "(auto sibling next to dataset)"
            print0(f"  token_cache_dir used: {effective_cache}")
            print0("  Make sure the token cache for this manifest exists (same as used by base_train.py).")
            raise

    for step in range(args.num_iterations):
        # =====================================================================
        # Data acquisition + active set computation
        # =====================================================================
        if using_real_loader and train_iter is not None:
            try:
                batch = next(train_iter)
                # The real manifest dataloader yields:
                #   (inputs, targets, step_meta, state_dict)
                # 'inputs' and 'targets' are **already remapped** to local indices
                # using the manifest's active set for this exact microstep.
                if isinstance(batch, (list, tuple)) and len(batch) >= 3:
                    inputs, targets, step_meta = batch[0], batch[1], batch[2]
                    noisy_batch = inputs.to(device) if hasattr(inputs, "to") else torch.as_tensor(inputs, device=device)

                    # For the LLaDA loss we need the original GLOBAL clean token IDs
                    slot_to_global = step_meta.get("slot_to_global_cpu")
                    if slot_to_global is not None:
                        slot_to_global = slot_to_global.to(device)
                        clean_global = slot_to_global[targets.to(device) if hasattr(targets, "to") else torch.as_tensor(targets, device=device)]
                    else:
                        clean_global = targets.to(device) if hasattr(targets, "to") else torch.as_tensor(targets, device=device)
                else:
                    noisy_batch = torch.as_tensor(batch[0] if isinstance(batch, (list,tuple)) else batch, device=device)
                    clean_global = noisy_batch.clone()
            except StopIteration:
                print0("Loader exhausted — stopping early.")
                break
            except Exception as e:
                print0(f"Error unpacking real manifest batch: {e}")
                raise

            # === Correct active set for the sparse runtime in manifest mode ===
            # Use the manifest's declared global active IDs + our MASK token.
            manifest_active_global = step_meta.get("active_ids_cpu")
            if manifest_active_global is not None:
                mask_tensor = torch.tensor([mask_id], dtype=torch.long, device=manifest_active_global.device)
                required_tokens = torch.unique(torch.cat([manifest_active_global, mask_tensor]))
            else:
                required_tokens = diffusion.get_diffusion_active_tokens(clean_global, mask_id)

        else:
            # Synthetic / non-manifest path
            noisy_batch = torch.randint(0, args.vocab_size, (args.device_batch_size, args.max_seq_len), device=device)
            clean_global = noisy_batch.clone()
            required_tokens = diffusion.get_diffusion_active_tokens(clean_global, mask_id)

        # === Exact LLaDA forward (noising) process ===
        noisy_for_loss, masked_indices, p_mask = forward_process(clean_global, mask_id=mask_id)

        model_input = noisy_batch   # may be remapped below for synthetic + sparse

        # === Sparse-aware model call + required remapping ===
        if dynamic_vocab is not None:
            sparse_step_ctx = dynamic_vocab.prepare_step(required_tokens)

            # In the synthetic/dynamic path we must remap the batch to local indices
            # because we generated raw global IDs.
            if not using_real_loader:
                active_global = sparse_step_ctx.active_ids_cpu.to(device)
                global_to_local = torch.full((args.vocab_size,), -1, dtype=torch.long, device=device)
                global_to_local[active_global] = torch.arange(len(active_global), device=device)
                model_input = global_to_local[noisy_batch]

            logits = model(
                model_input,
                causal=False,
                active_vocab=sparse_step_ctx.active_vocab,
            )
            do_sparse_step = True

            # Remap targets for loss (works for both manifest and synthetic paths)
            active_global = getattr(sparse_step_ctx, 'active_ids_cpu', None)
            if active_global is not None:
                active_global = active_global.to(device)
                global_to_local = torch.full((args.vocab_size,), -1, dtype=torch.long, device=device)
                global_to_local[active_global] = torch.arange(len(active_global), device=device)
                local_targets = global_to_local[clean_global]
            else:
                local_targets = clean_global
        else:
            logits = model(model_input, causal=False)
            sparse_step_ctx = None
            do_sparse_step = False
            local_targets = clean_global
            local_targets = clean_global

        # === Exact LLaDA weighted masked loss ===
        loss = compute_llada_loss(logits, local_targets, masked_indices, p_mask)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if do_sparse_step and sparse_step_ctx is not None:
            dynamic_vocab.step(sparse_step_ctx)

        if step % 10 == 0 or step == args.num_iterations - 1:
            num_masked = int(masked_indices.sum().item())
            avg_p = float(p_mask[masked_indices].mean().item()) if num_masked > 0 else 0.0
            mode_str = "sparse" if dynamic_vocab is not None else "dense"
            print0(f"step {step:04d} | mode={mode_str} | loss {loss.item():.4f} | masked {num_masked:5d} | avg_p {avg_p:.3f}")

    print0("\nLLaDA diffusion run completed.")
    if args.sparse_mode and args.sparse_manifest:
        print0("Sparse + real manifest mode was active. This run exercised DynamicVocabRuntime with actual 65k data.")
    elif args.sparse_mode:
        print0("Sparse mode was active (dynamic U discovery).")


if __name__ == "__main__":
    main()
