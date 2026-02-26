import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.autograd import Function


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


class _SparseEmbeddingFn(Function):
    @staticmethod
    def forward(ctx, weight: torch.Tensor, U: torch.Tensor, local_idx: torch.Tensor):
        weight_on_cpu = (weight.device.type == "cpu")
        if weight_on_cpu:
            U_cpu = U.to("cpu", non_blocking=True)
            W_U = weight.index_select(0, U_cpu).to(local_idx.device, non_blocking=True)
            ctx.save_for_backward(U_cpu, local_idx)
        else:
            W_U = weight.index_select(0, U)
            ctx.save_for_backward(U, local_idx)
        ctx.weight_on_cpu = weight_on_cpu
        ctx.weight_shape = weight.shape
        ctx.weight_dtype = weight.dtype
        return F.embedding(local_idx, W_U)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        U, local_idx = ctx.saved_tensors
        d = grad_out.size(-1)
        grad_W_U = torch.zeros((U.numel(), d), dtype=grad_out.dtype, device=grad_out.device)
        grad_W_U.index_add_(0, local_idx.reshape(-1), grad_out.reshape(-1, d))
        grad_W_U = grad_W_U.to(dtype=ctx.weight_dtype)
        if ctx.weight_on_cpu:
            grad_W_U = grad_W_U.to("cpu", non_blocking=True)
            sparse_device = "cpu"
        else:
            sparse_device = grad_out.device
        grad_weight = torch.sparse_coo_tensor(
            U.unsqueeze(0),
            grad_W_U,
            size=ctx.weight_shape,
            dtype=grad_W_U.dtype,
            device=sparse_device,
        ).coalesce()
        return grad_weight, None, None


def sparse_embedding(weight: torch.Tensor, local_idx: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    return _SparseEmbeddingFn.apply(weight, U, local_idx)


class _SparseLogitsFn(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, U: torch.Tensor):
        weight_on_cpu = (weight.device.type == "cpu")
        if weight_on_cpu:
            U_cpu = U.to("cpu", non_blocking=True)
            W_U = weight.index_select(0, U_cpu).to(x.device, non_blocking=True)
            ctx.save_for_backward(x, U_cpu, W_U)
        else:
            W_U = weight.index_select(0, U)
            ctx.save_for_backward(x, U, W_U)
        ctx.weight_on_cpu = weight_on_cpu
        ctx.weight_shape = weight.shape
        ctx.weight_dtype = weight.dtype
        return x @ W_U.T

    @staticmethod
    def backward(ctx, grad_logits: torch.Tensor):
        x, U, W_U = ctx.saved_tensors
        grad_x = grad_logits @ W_U.to(dtype=grad_logits.dtype)
        grad_W_U = grad_logits.float().reshape(-1, grad_logits.size(-1)).T @ x.float().reshape(-1, x.size(-1))
        grad_W_U = grad_W_U.to(dtype=ctx.weight_dtype)
        if ctx.weight_on_cpu:
            grad_W_U = grad_W_U.to("cpu", non_blocking=True)
            sparse_device = "cpu"
        else:
            sparse_device = grad_logits.device
        grad_weight = torch.sparse_coo_tensor(
            U.unsqueeze(0),
            grad_W_U,
            size=ctx.weight_shape,
            dtype=grad_W_U.dtype,
            device=sparse_device,
        ).coalesce()
        return grad_x, grad_weight, None


def sparse_logits(x: torch.Tensor, weight: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    return _SparseLogitsFn.apply(x, weight, U)


def local_vocab_log_correction(vocab_size: int, local_vocab_size: int) -> float:
    return math.log(vocab_size / local_vocab_size)