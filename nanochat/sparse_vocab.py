import math
import time

import numpy as np
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


def _pin(t: torch.Tensor) -> torch.Tensor:
    """Pin *t* to page-locked memory if not already pinned.

    ``index_select`` on a pinned master weight returns a **pageable** tensor.
    Without explicit pinning, ``.to(device, non_blocking=True)`` silently falls
    back to a staged synchronous copy (pageable → internal bounce buffer → DMA)
    instead of a direct single-step DMA from pinned memory.  The staged path is
    the cause of slow H2D times even when non_blocking=True is set.
    """
    if torch.cuda.is_available() and not t.is_pinned():
        return t.pin_memory()
    return t


class _SparseEmbeddingFn(Function):
    @staticmethod
    def forward(ctx, weight: torch.Tensor, U: torch.Tensor, local_idx: torch.Tensor):
        weight_on_cpu = (weight.device.type == "cpu")
        if weight_on_cpu:
            U_cpu = U.to("cpu", non_blocking=True)
            W_U = _pin(weight.index_select(0, U_cpu)).to(local_idx.device, non_blocking=True)
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


class _SparseEmbeddingCachedFn(Function):
    @staticmethod
    def forward(ctx, weight: torch.Tensor, U: torch.Tensor, local_idx: torch.Tensor, W_U: torch.Tensor):
        ctx.save_for_backward(U, local_idx)
        ctx.weight_on_cpu = (weight.device.type == "cpu")
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
        return grad_weight, None, None, None


def sparse_embedding_cached(weight: torch.Tensor, local_idx: torch.Tensor, U: torch.Tensor, W_U: torch.Tensor) -> torch.Tensor:
    return _SparseEmbeddingCachedFn.apply(weight, U, local_idx, W_U)


class _SparseLogitsFn(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, U: torch.Tensor):
        weight_on_cpu = (weight.device.type == "cpu")
        if weight_on_cpu:
            U_saved = U.to("cpu", non_blocking=True)
            W_U = _pin(weight.index_select(0, U_saved)).to(x.device, non_blocking=True)
        else:
            U_saved = U
            W_U = weight.index_select(0, U_saved)
        ctx.save_for_backward(x, U_saved, weight)
        ctx.weight_on_cpu = weight_on_cpu
        ctx.weight_shape = weight.shape
        ctx.weight_dtype = weight.dtype
        return x @ W_U.T

    @staticmethod
    def backward(ctx, grad_logits: torch.Tensor):
        x, U, weight = ctx.saved_tensors
        if ctx.weight_on_cpu:
            W_U = _pin(weight.index_select(0, U)).to(grad_logits.device, non_blocking=True)
        else:
            W_U = weight.index_select(0, U)
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


class _SparseLogitsCachedFn(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, U: torch.Tensor, W_U: torch.Tensor):
        ctx.save_for_backward(x, U, W_U)
        ctx.weight_on_cpu = (weight.device.type == "cpu")
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
        return grad_x, grad_weight, None, None


def sparse_logits_cached(x: torch.Tensor, weight: torch.Tensor, U: torch.Tensor, W_U: torch.Tensor) -> torch.Tensor:
    return _SparseLogitsCachedFn.apply(x, weight, U, W_U)


def local_vocab_log_correction(vocab_size: int, local_vocab_size: int) -> float:
    return math.log(vocab_size / local_vocab_size)


class SparseVocabPool:
    """Per-table GPU row cache for sparse vocab training with arbitrarily-large vocabs.

    Design invariants
    -----------------
    * Master weight tables live on CPU (or NVMe) **forever** — they never move.
    * Only rows for batch-unique tokens (~|U_batch|) ever occupy VRAM.
    * VRAM scales with |U_batch |× row_dim, not vocab_size × row_dim.
    * vocab_size can be arbitrarily large; the index costs 4 bytes/token (int32).

    Critical performance note
    -------------------------
    The previous GPU-tensor-based index caused `.item()` / `.any()` calls that
    forced a GPU→CPU sync on EVERY `get_rows` call.  At training time this
    stalled the CPU until the *prior micro-step's backward pass* drained on the
    GPU, contributing ~480 ms of dead time per optimizer step despite only ~2 ms
    of actual PCIe transfer.

    This version keeps the hit/miss index entirely in CPU numpy arrays.
    `get_rows` contains **zero GPU synchronization** in its hot path.  The only
    GPU work it enqueues is:
      - non-blocking H2D of miss-rows (into the default CUDA stream)
      - in-place GPU scatter of those rows into self.rows (same stream)
    CUDA stream ordering guarantees both complete before any downstream kernel
    that consumes the returned tensor.

    Data structures
    ---------------
    rows              (capacity, row_dim)  bf16  GPU   row data only
    global_to_slot_np (vocab_size,)        int32 CPU numpy  -1 = absent
    slot_to_global_np (capacity,)          int32 CPU numpy  -1 = empty
    """

    def __init__(
        self,
        vocab_size: int,
        row_dim: int,
        capacity: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.vocab_size = vocab_size
        self.row_dim    = row_dim
        self.capacity   = min(capacity, vocab_size)
        self.device     = device

        # ---- GPU: row data only (no index tensors on GPU) ---------------
        self.rows = torch.empty(
            (self.capacity, row_dim), dtype=dtype, device=device
        )

        # ---- CPU numpy: hit/miss index (zero GPU sync ever needed) ------
        self.global_to_slot_np = np.full(vocab_size,    -1, dtype=np.int32)
        self.slot_to_global_np = np.full(self.capacity, -1, dtype=np.int32)
        self._evict_ptr: int = 0

        # ---- stats (reset by pop_stats) ---------------------------------
        self._hits: int = 0
        self._misses: int = 0
        self._bytes_loaded: int = 0
        self._load_ms: float = 0.0

    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_rows(self, tokens: torch.Tensor, master_weight: torch.Tensor) -> torch.Tensor:
        """Return (|tokens|, row_dim) bf16 GPU tensor.

        tokens        – sorted 1-D **CPU** LongTensor of global token ids.
                        Must be on CPU so no D2H sync is required here.
                        Callers compute desired_U on CPU before calling.
        master_weight – CPU-resident (optionally pinned) embedding table.

        Zero GPU synchronization in this method.  All GPU writes are enqueued
        non-blocking in the default CUDA stream; downstream kernels wait via
        CUDA stream ordering.
        """
        t0 = time.perf_counter()
        assert tokens.device.type == "cpu", "get_rows requires a CPU token tensor (no GPU sync)"
        n = tokens.numel()
        if n == 0:
            return self.rows.new_empty((0, self.row_dim))

        # ---- CPU classification — ZERO GPU sync ----------------------
        tokens_np = tokens.numpy()                        # view, no copy
        slots_np  = self.global_to_slot_np[tokens_np]    # pure numpy, instant
        hit_mask  = slots_np >= 0
        miss_mask = ~hit_mask
        n_miss    = int(miss_mask.sum())                  # Python int from numpy, no GPU
        n_hit     = n - n_miss

        out = self.rows.new_empty((n, self.row_dim))      # GPU alloc only

        # ---- gather hits from GPU cache (no sync) --------------------
        if n_hit > 0:
            hit_slots   = torch.from_numpy(slots_np[hit_mask].astype(np.int64)).to(self.device, non_blocking=True)
            hit_out_pos = torch.from_numpy(np.where(hit_mask)[0].astype(np.int64)).to(self.device, non_blocking=True)
            out[hit_out_pos] = self.rows[hit_slots]

        # ---- load misses from CPU master_weight ----------------------
        if n_miss > 0:
            miss_tokens_np  = tokens_np[miss_mask]                 # sorted view (no copy)
            miss_tokens_cpu = torch.from_numpy(miss_tokens_np.astype(np.int64))  # CPU tensor
            # Sorted miss_tokens → sequential-ish row access → good CPU cache locality.
            miss_rows_cpu   = master_weight.index_select(0, miss_tokens_cpu)
            # pin_memory() is required: index_select on a pinned master weight
            # returns a pageable tensor. Without explicit pinning here,
            # non_blocking H2D silently degrades to a staged synchronous copy.
            # With pinning, CUDA issues a single direct DMA from the pinned buffer.
            miss_rows = _pin(miss_rows_cpu).to(self.device, dtype=self.rows.dtype, non_blocking=True)

            miss_out_pos = torch.from_numpy(np.where(miss_mask)[0].astype(np.int64)).to(self.device, non_blocking=True)
            out[miss_out_pos] = miss_rows  # GPU scatter, enqueued after H2D

            # ---- ring-buffer eviction (pure Python/numpy — no GPU) ----
            evict_slots_np = np.arange(self._evict_ptr, self._evict_ptr + n_miss, dtype=np.int64) % self.capacity
            self._evict_ptr = (self._evict_ptr + n_miss) % self.capacity

            old_globals = self.slot_to_global_np[evict_slots_np]
            valid_evict = old_globals >= 0
            if valid_evict.any():
                self.global_to_slot_np[old_globals[valid_evict]] = -1

            # write new rows into GPU buffer (enqueued after miss_rows H2D)
            evict_slots_t = torch.from_numpy(evict_slots_np).to(self.device, non_blocking=True)
            self.rows[evict_slots_t] = miss_rows

            # update CPU index
            self.slot_to_global_np[evict_slots_np] = miss_tokens_np.astype(np.int32)
            self.global_to_slot_np[miss_tokens_np] = evict_slots_np.astype(np.int32)

            self._bytes_loaded += n_miss * self.row_dim * self.rows.element_size()

        self._hits    += n_hit
        self._misses  += n_miss
        self._load_ms += (time.perf_counter() - t0) * 1000.0
        return out

    # ------------------------------------------------------------------
    def invalidate(self, tokens: torch.Tensor) -> None:
        """Evict *tokens* from the pool after the optimizer updates their rows.

        O(|tokens|), no GPU sync, no pool scan.
        tokens is expected to be a CPU tensor (from the dataloader's sparse_context).
        """
        if tokens.numel() == 0:
            return
        if tokens.device.type != "cpu":
            tokens = tokens.cpu()
        tokens_np = tokens.numpy().astype(np.int64)
        slots     = self.global_to_slot_np[tokens_np]
        valid     = slots >= 0
        if valid.any():
            self.global_to_slot_np[tokens_np[valid]] = -1
            self.slot_to_global_np[slots[valid]]     = -1

    # ------------------------------------------------------------------
    def pop_stats(self) -> dict:
        s = {
            "hits":         self._hits,
            "misses":       self._misses,
            "bytes_loaded": self._bytes_loaded,
            "load_ms":      self._load_ms,
        }
        self._hits = self._misses = self._bytes_loaded = 0
        self._load_ms = 0.0
        return s

    def vram_bytes(self) -> int:
        """Bytes of GPU VRAM held by this pool (rows buffer only; index is CPU numpy)."""
        return self.rows.numel() * self.rows.element_size()