import math

import torch
import torch.distributed as dist


def _ddp_union_tokens(tokens: torch.Tensor) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized()):
        return tokens
    world_size = dist.get_world_size()
    if world_size == 1:
        return tokens

    local_len = torch.tensor([tokens.numel()], dtype=torch.long, device=tokens.device)
    gathered_lens = [torch.zeros_like(local_len) for _ in range(world_size)]
    dist.all_gather(gathered_lens, local_len)
    max_len = int(torch.stack(gathered_lens).max().item())

    padded = torch.full((max_len,), -1, dtype=torch.long, device=tokens.device)
    padded[:tokens.numel()] = tokens
    gathered = [torch.empty_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded)

    flat = torch.cat(gathered)
    flat = flat[flat >= 0]
    return torch.unique(flat, sorted=True)


def compute_batch_token_set(idx: torch.Tensor, targets: torch.Tensor, vocab_size: int, use_ddp_union: bool = False):
    valid_targets = targets[targets >= 0]
    if valid_targets.numel() > 0:
        tokens = torch.cat([idx.reshape(-1), valid_targets.reshape(-1)])
    else:
        tokens = idx.reshape(-1)
    U = torch.unique(tokens, sorted=True)
    if use_ddp_union:
        U = _ddp_union_tokens(U)

    global_to_local = torch.full((vocab_size,), -1, dtype=torch.long, device=idx.device)
    global_to_local[U] = torch.arange(U.numel(), device=idx.device, dtype=torch.long)

    local_idx = global_to_local[idx]
    local_targets = torch.full_like(targets, -1)
    valid = targets >= 0
    local_targets[valid] = global_to_local[targets[valid]]
    return U, global_to_local, local_idx, local_targets


def local_vocab_log_correction(vocab_size: int, local_vocab_size: int) -> float:
    return math.log(vocab_size / local_vocab_size)
