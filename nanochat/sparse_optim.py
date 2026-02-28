"""
Vocabulary-row-sparse optimizer for sparse-mode training.

VocabRowAdamW maintains per-row Adam m/v state for every vocab table
(wte, value_embeds, lm_head) entirely on CPU.  Each optimizer step it
receives the accumulated gradient for only the |U_step| active rows
and updates exactly those rows in the master weight tensors — no sparse
COO tensors, no SparseAdam, no custom autograd Functions.

When device="cuda", the per-step Adam arithmetic runs on GPU using
row slices that were pre-fetched to GPU before the forward pass
(mirroring the weight-row prefetch pattern).  The full _m/_v state
tables always remain CPU-resident; only |U_step| rows are transiently
on GPU during the optimizer step.
"""

import torch
import torch.distributed as dist


class VocabRowAdamW:
    """Per-row AdamW for CPU-resident vocabulary tables.

    Parameters
    ----------
    tables : dict[str, torch.Tensor]
        Mapping of table key → master weight tensor (CPU, float32).
        Updated in-place by step().
    initial_lrs : dict[str, float]
        Per-table base learning rates (at lr_multiplier=1.0).
    betas : (float, float)
    eps : float
    weight_decay : float
    device : str
        Where to run Adam arithmetic: ``"cpu"`` (default) or ``"cuda"``.
        The full _m/_v tables always remain CPU-resident; ``"cuda"`` only
        controls whether per-step row arithmetic executes on GPU.
        Requires ``prefetched_m``, ``prefetched_v``, and ``prefetched_w``
        to be supplied to ``step()`` when using the CUDA path.
    """

    def __init__(
        self,
        tables: dict,
        initial_lrs: dict,
        betas: tuple = (0.8, 0.95),
        eps: float = 1e-10,
        weight_decay: float = 0.0,
        device: str = "cpu",
    ) -> None:
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self._initial_lrs: dict = dict(initial_lrs)
        self._t: int = 0
        self._device: str = device

        self._m: dict = {k: torch.zeros_like(w, dtype=torch.float32) for k, w in tables.items()}
        self._v: dict = {k: torch.zeros_like(w, dtype=torch.float32) for k, w in tables.items()}
        # Prevent first-update explosion on new tokens (sparse GPU bf16 path)
        for v in self._v.values():
            v.fill_(1e-6)

        # CUDA-only: pre-allocated pinned staging buffers eliminate the per-step
        # pin_memory() allocations and halve CPU bandwidth for prefetch/writeback.
        # Full vocab-size buffers persist; only |U_step| rows are used each step.
        if device == "cuda":
            self._pin_m: dict = {k: torch.zeros_like(w, pin_memory=True) for k, w in tables.items()}
            self._pin_v: dict = {k: torch.zeros_like(w, pin_memory=True) for k, w in tables.items()}
            self._pin_w: dict = {k: torch.zeros_like(w, pin_memory=True) for k, w in tables.items()}
            # Dedicated D2H stream so the updated-row DMA back to CPU doesn't
            # stall the main compute stream.  A CUDA event lets us fence
            # the completion at the start of the *next* step.
            self._d2h_stream: torch.cuda.Stream = torch.cuda.Stream()
            self._d2h_event: torch.cuda.Event = torch.cuda.Event()
            self._pending_U: torch.Tensor | None = None
            self._pending_tables: dict | None = None

    def prefetch_to_gpu(
        self,
        U_step: torch.Tensor,
        device,
        tables: dict,
    ) -> tuple:
        """Gather active rows into pre-allocated pinned buffers and fire non-blocking H2D.

        Replaces the per-step ``index_select() + pin_memory() + .to(device)`` chain
        with a single ``index_select(..., out=pre_pinned)`` per table — halving CPU
        memory bandwidth and eliminating all per-step pinned allocations.

        Returns (prefetched_m, prefetched_v, prefetched_w): dicts of GPU f32 tensors
        with shape (|U_step|, dim), ready for immediate use in ``step()``.
        Must be called after ``flush_pending_writes()`` for the current step.
        """
        U_size = U_step.numel()
        for key in self._pin_m:
            torch.index_select(self._m[key],  0, U_step, out=self._pin_m[key][:U_size])
            torch.index_select(self._v[key],  0, U_step, out=self._pin_v[key][:U_size])
            torch.index_select(tables[key],   0, U_step, out=self._pin_w[key][:U_size])
        pm = {k: self._pin_m[k][:U_size].to(device, non_blocking=True) for k in self._pin_m}
        pv = {k: self._pin_v[k][:U_size].to(device, non_blocking=True) for k in self._pin_v}
        pw = {k: self._pin_w[k][:U_size].to(device, non_blocking=True) for k in self._pin_w}
        return pm, pv, pw

    def flush_pending_writes(self) -> None:
        """Commit the previous step's deferred D2H results into the CPU master tables.

        Waits for the D2H DMA (which was fired asynchronously at the end of the
        previous step) to complete, then scatter-writes the updated rows into the
        full-vocab ``_m``, ``_v``, and master weight tables.

        Must be called at the start of each step's prefetch section, *before*
        ``prefetch_to_gpu()`` reads from those same tables.
        Is a no-op on the first step (nothing pending) and on the CPU path.
        """
        if self._device != "cuda" or not hasattr(self, '_pending_U') or self._pending_U is None:
            return
        # Block CPU until the D2H DMA stream has finished copying data into _pin_m/v/w.
        self._d2h_event.synchronize()
        U = self._pending_U
        U_size = U.numel()
        for key in self._pin_m:
            self._m[key][U] = self._pin_m[key][:U_size]
            self._v[key][U] = self._pin_v[key][:U_size]
        if self._pending_tables is not None:
            for key in self._pin_w:
                self._pending_tables[key][U] = self._pin_w[key][:U_size]
        self._pending_U = None
        self._pending_tables = None

    @torch.no_grad()
    def step(
        self,
        U_step: torch.Tensor,
        grads: dict,
        tables: dict,
        lr_multiplier: float = 1.0,
        prefetched_m: dict = None,
        prefetched_v: dict = None,
        prefetched_w: dict = None,
    ) -> None:
        """Apply one Adam update to the U_step rows of every table.

        U_step        : CPU sorted LongTensor of active global token indices.
        grads         : {key: GPU bf16 or CPU tensor} shape (|U_step|, dim).
        tables        : {key: master_weight_cpu_f32} — same references as at
                        __init__.  Updated in-place.
        lr_multiplier : LR schedule multiplier (lrm); effective lr = initial_lrs[key] * lr_multiplier.
        prefetched_m  : {key: GPU f32 (|U_step|, dim)} — active rows of _m,
                        pre-fetched from CPU before the forward pass.
                        Must be provided when device="cuda".
        prefetched_v  : Same structure for _v.
        prefetched_w  : {key: GPU f32 (|U_step|, dim)} — active rows of the
                        master weight table, pre-fetched from CPU.
                        Must be provided when device="cuda".
        """
        beta1, beta2 = self.betas
        self._t += 1
        t = self._t
        bc1 = 1.0 - beta1 ** t
        bc2 = 1.0 - beta2 ** t

        use_gpu = (
            self._device == "cuda"
            and prefetched_m is not None
            and prefetched_v is not None
            and prefetched_w is not None
        )

        

        for key, grad in grads.items():
            if grad is None:
                continue

            lr = self._initial_lrs[key] * lr_multiplier

            if use_gpu:
                # ---- GPU arithmetic path (uses pre-fetched pinned row slices) ----
                # grad is already GPU bf16 — upcast in-place on GPU, no H2D.
                # prefetched_m/v/w are GPU f32 row slices fetched before the forward pass.
                g = grad.detach().to(dtype=torch.float32)
                # Zero out ALL non-finite gradient elements before Adam arithmetic.
                # Dense has f32 grads (can't overflow); sparse has bf16 grads that can
                # produce ±inf from overflow (~65504 max). The upstream norm clip can
                # produce NaN from 0*inf=NaN in IEEE 754. Both are neutralised here.
                g = g.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

                m_rows = beta1 * prefetched_m[key] + (1.0 - beta1) * g
                v_rows = beta2 * prefetched_v[key] + (1.0 - beta2) * g.square()

                m_hat = m_rows / bc1
                v_hat = v_rows / bc2
                update = m_hat / (v_hat.sqrt_() + self.eps)
                update = update.clamp_(-1.0, 1.0)
                # Standard AdamW: weight decay on the pre-update weight, then subtract step.
                w_updated = prefetched_w[key] * (1.0 - lr * self.weight_decay) - lr * update
                w_updated = w_updated.clamp_(-3.0, 3.0)
                
                # Check for NaN/Inf in the updated weights before D2H transfer
                if torch.isnan(w_updated).any() or torch.isinf(w_updated).any():
                    print(f"[VocabRowAdamW DEBUG] NaN/Inf detected in {key} GPU updated weights, replacing with zeros")
                    w_updated[torch.isnan(w_updated) | torch.isinf(w_updated)] = 0.0

                # Non-blocking D2H into pre-allocated pinned staging buffers.
                # The D2H stream waits for the compute stream to finish the Adam
                # kernels above before issuing DMA — so we never copy stale data.
                # The main thread is NOT blocked; it continues immediately to
                # zero_grad / logging while DMA runs in the background.
                # flush_pending_writes() at the start of the *next* step will
                # block until DMA is done, then scatter-write into _m/_v/tables.
                #
                # IMPORTANT: capture current_stream() BEFORE entering the
                # `with torch.cuda.stream(self._d2h_stream):` context, because
                # that context switch makes current_stream() return _d2h_stream
                # itself — turning the wait_stream call into a self-wait no-op
                # that would let DMA fire before Adam kernels complete.
                _compute_stream = torch.cuda.current_stream()
                U_size = U_step.numel()
                with torch.cuda.stream(self._d2h_stream):
                    self._d2h_stream.wait_stream(_compute_stream)
                    self._pin_m[key][:U_size].copy_(m_rows, non_blocking=True)
                    self._pin_v[key][:U_size].copy_(v_rows, non_blocking=True)
                    self._pin_w[key][:U_size].copy_(w_updated, non_blocking=True)

            else:
                # ---- CPU arithmetic path (original) ----
                # Move grad to CPU float32 (blocking — GPU has already finished backward)
                g = grad.detach().to(dtype=torch.float32, device="cpu")
                # Same non-finite guard as GPU path.
                g = g.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

                m_rows = self._m[key][U_step]
                v_rows = self._v[key][U_step]

                m_rows = beta1 * m_rows + (1.0 - beta1) * g
                v_rows = beta2 * v_rows + (1.0 - beta2) * g.square()

                # Mimic dense-path behaviour: decay ALL rows (including inactive ones
                # that received no gradient this step). Inactive rows decay as
                # m *= beta1 each step, matching exactly what dense AdamW does when
                # grad=0.  Active rows are then overwritten with the correct update.
                self._m[key].mul_(beta1)
                self._v[key].mul_(beta2)
                self._m[key][U_step] = m_rows
                self._v[key][U_step] = v_rows

                m_hat = m_rows / bc1
                v_hat = v_rows / bc2

                update = m_hat / (v_hat.sqrt_() + self.eps)
                w_rows = tables[key][U_step]
                # Standard AdamW: weight decay on the pre-update weight, then subtract step.
                new_weights = w_rows * (1.0 - lr * self.weight_decay) - lr * update
                
                # Check for NaN/Inf in the new weights before updating
                if torch.isnan(new_weights).any() or torch.isinf(new_weights).any():
                    print(f"[VocabRowAdamW DEBUG] NaN/Inf detected in {key} new weights, replacing with zeros")
                    new_weights[torch.isnan(new_weights) | torch.isinf(new_weights)] = 0.0
                
                tables[key][U_step] = new_weights

        # GPU path: decay ALL rows on the CPU master tables to mimic dense behaviour
        # (inactive rows with no gradient this step should have m *= beta1, v *= beta2).
        # The CPU master tables still hold last-step's values for active rows because
        # flush_pending_writes() hasn't run yet — that's fine: the next call to
        # flush_pending_writes() will overwrite active rows with the correctly computed
        # m_rows/v_rows that were D2H'd above, so the intermediate decay is harmless.
        if use_gpu:
            for key in self._m:
                self._m[key].mul_(beta1)
                self._v[key].mul_(beta2)

        # After the per-key loop: record the D2H event and store pending state.
        # flush_pending_writes() at the start of the *next* step will synchronize
        # on this event before reading from _m/_v/tables.
        if use_gpu:
            self._d2h_event.record(self._d2h_stream)
            self._pending_U = U_step
            self._pending_tables = tables

    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        # Flush any in-flight D2H writes so the returned state is consistent.
        self.flush_pending_writes()
        return {
            "t": self._t,
            "m": {k: v.clone() for k, v in self._m.items()},
            "v": {k: v.clone() for k, v in self._v.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        self._t = int(state["t"])
        for k in self._m:
            if k in state["m"]:
                self._m[k].copy_(state["m"][k])
            if k in state["v"]:
                self._v[k].copy_(state["v"][k])
