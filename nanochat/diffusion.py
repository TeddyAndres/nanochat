"""
LLaDA-style masked diffusion utilities for nanochat.

This module implements the core forward (noising) process and loss
from "Large Language Diffusion Models" (arXiv:2502.09992, ML-GSAI/LLaDA).

Key properties (matching LLaDA exactly):
- Per-example t ~ Uniform[0, 1]
- Linear masking schedule with small eps floor
- Model predicts original (clean) tokens ONLY at masked positions
- Loss is cross-entropy on masked positions, re-weighted by 1/p_mask
  This gives a proper variational upper bound on NLL.

=== Important contract for the dynamic/sparse vocabulary runtime ===

**MASK is a diffusion-level always-hot token.**

When using LLaDA-style diffusion together with DynamicVocabRuntime (sparse mode):

- The set of tokens that must be active for a step is:
      unique(clean tokens)  ∪  {mask_id}

- The MASK token must be **unconditionally** included in every active set
  passed to `DynamicVocabRuntime.prepare_step(...)`, even if it did not
  appear in the current clean batch.

- We deliberately do **not** bloat manifests with MASK. It is injected at
  the training script level (see `get_diffusion_active_tokens` below).

Recommended pattern:

    clean_ids = ...                    # from dataloader / manifest loader
    mask_id = args.mask_id             # required for real runs
    required = diffusion.get_diffusion_active_tokens(clean_ids, mask_id)
    # optionally union any other permanent hot tokens here
    ctx = dynamic_vocab.prepare_step(required)
    ...
    dynamic_vocab.step(ctx)

This keeps manifests clean (they only describe the original data) while
guaranteeing MASK is always resident when needed for noising / prediction.

This module itself stays completely independent of the sparse runtime.
The wiring + enforcement lives in the training script.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# LLaDA uses token id 126336 as [MASK]. We keep the same default here
# so manifests and tokenizers are easy to share. This can be overridden
# at runtime via the training script if the nanochat tokenizer uses a
# different reserved id.
DEFAULT_MASK_ID: int = 126336


def forward_process(
    input_ids: torch.Tensor,
    eps: float = 1e-3,
    mask_id: int = DEFAULT_MASK_ID,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """LLaDA forward (noising) process.

    Args:
        input_ids: (b, l) clean token ids
        eps: small floor so p_mask never hits exactly 0 or 1
        mask_id: the token id that represents "masked"

    Returns:
        noisy_batch: (b, l) with some positions replaced by mask_id
        masked_indices: (b, l) boolean, True where the token was masked
        p_mask: (b, l) the per-token masking probability used for this example
    """
    b, l = input_ids.shape
    device = input_ids.device

    t = torch.rand(b, device=device)                    # t ~ U[0,1] per example
    p_mask = (1 - eps) * t + eps                        # linear schedule
    p_mask = p_mask[:, None].repeat(1, l)               # broadcast to sequence

    masked_indices = torch.rand((b, l), device=device) < p_mask
    noisy_batch = torch.where(masked_indices, mask_id, input_ids)

    return noisy_batch, masked_indices, p_mask


def compute_llada_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    masked_indices: torch.Tensor,
    p_mask: torch.Tensor,
) -> torch.Tensor:
    """LLaDA training loss (weighted masked cross-entropy).

    Only positions that were masked in the forward process contribute to the loss.
    The 1/p_mask reweighting is what turns the objective into a valid
    variational upper bound on negative log-likelihood (see LLaDA paper §2.1
    and the RADD/SMDM lineage).

    This formulation is fully compatible with the existing dynamic-vocab
    sparse path: we still produce logits over the active vocab U (via the
    same active_vocab injection), and the targets are still clean token ids
    from the original sequence. The only difference is *which* positions
    we select for the CE and the extra division by p_mask.
    """
    # logits and input_ids are expected to be (b, l, V) and (b, l)
    # masked_indices selects the positions that were noised
    token_loss = F.cross_entropy(
        logits[masked_indices],
        input_ids[masked_indices],
        reduction="none",
    ) / p_mask[masked_indices]

    # Standard LLaDA normalization (sum over all selected tokens, divide by batch*seq)
    loss = token_loss.sum() / (input_ids.shape[0] * input_ids.shape[1])
    return loss


def sample_timesteps(batch_size: int, device: torch.device) -> torch.Tensor:
    """Convenience helper if you ever need raw t values (not currently required)."""
    return torch.rand(batch_size, device=device)


def get_tokens_for_active_vocab(
    input_ids: torch.Tensor,
    mask_id: int,
) -> torch.Tensor:
    """Returns the set of token ids that must be present in the active vocab U
    for a diffusion (LLaDA-style) training or sampling step.

    This is the union of:
      - all unique clean tokens appearing in `input_ids`
      - the `mask_id`

    This is the recommended helper for sparse training code when computing
    the active set for DynamicVocabRuntime.

    The caller is still responsible for also including any other always-hot
    tokens (e.g. BOS, EOS, or other special tokens required by the model).
    """
    flat = input_ids.reshape(-1)
    unique_clean = torch.unique(flat)
    # Ensure mask_id is included even if it didn't appear in this batch
    mask_tensor = torch.tensor([mask_id], device=flat.device, dtype=flat.dtype)
    combined = torch.cat([unique_clean, mask_tensor])
    return torch.unique(combined)


def get_diffusion_active_tokens(
    clean_ids: torch.Tensor,
    mask_id: int,
) -> torch.Tensor:
    """Convenience wrapper specifically for LLaDA/diffusion + sparse runtime.

    Returns the exact set of token ids that must be active for one diffusion
    training/sampling step:

        unique(clean tokens)  ∪  {mask_id}

    This set should be passed directly to
    `DynamicVocabRuntime.prepare_step(...)` (or unioned with any other
    permanent hot tokens).

    Using this function makes the "MASK is a diffusion-level always-hot token"
    contract explicit in the training code.
    """
    return get_tokens_for_active_vocab(clean_ids, mask_id)
