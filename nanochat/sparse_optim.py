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

import time

import torch


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
        defer_writeback: bool = False,
        overlap_cache: bool = False,
    ) -> None:
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self._initial_lrs: dict = dict(initial_lrs)
        self._t: int = 0
        self._device: str = device
        self._defer_writeback: bool = bool(defer_writeback)
        self._overlap_cache: bool = bool(overlap_cache) and device == "cuda"
        self._tables_ref: dict = tables

        self._m: dict = {k: torch.zeros_like(w, dtype=torch.float32) for k, w in tables.items()}
        self._v: dict = {k: torch.zeros_like(w, dtype=torch.float32) for k, w in tables.items()}
        # Prevent first-update explosion on new tokens (sparse GPU bf16 path)
        for v in self._v.values():
            v.fill_(0.0001)

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
            self._pending_keys: tuple[str, ...] | None = None
            self._pending_size: int = 0
            self._pending_rows_m: dict[str, torch.Tensor] | None = None
            self._pending_rows_v: dict[str, torch.Tensor] | None = None
            self._pending_rows_w: dict[str, torch.Tensor] | None = None
            # Overlap mode can enqueue multiple deferred eviction writebacks.
            self._pending_overlap: list[dict] = []
            self._pending_overlap_limit: int = 2
            self.last_prefetch_stats: dict[str, float | int] = {}

            # Overlap cache (GPU-resident active rows from previous step)
            self._cache_U: torch.Tensor | None = None  # CPU sorted global ids
            self._cache_m: dict[str, torch.Tensor] = {}
            self._cache_v: dict[str, torch.Tensor] = {}
            self._cache_w: dict[str, torch.Tensor] = {}
            if self._overlap_cache and len(tables) > 0:
                vocab_size = next(iter(tables.values())).size(0)
                self._last_seen_step: torch.Tensor = torch.zeros(vocab_size, dtype=torch.int64)
            else:
                self._last_seen_step = torch.empty(0, dtype=torch.int64)

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
        """
        if self._overlap_cache:
            U_cpu = U_step.to(device="cpu")
            U_size = U_cpu.numel()

            if self._cache_U is None:
                pm, pv, pw = self._fetch_rows_from_cpu(U_cpu, device, tables)
                self._cache_U = U_cpu.clone()
                self._cache_m = pm
                self._cache_v = pv
                self._cache_w = pw
                self.last_prefetch_stats = {
                    "hit_count": 0,
                    "miss_count": int(U_size),
                    "evict_count": 0,
                    "flush_wait_ms": 0.0,
                }
                return self._cache_m, self._cache_v, self._cache_w

            old_U = self._cache_U
            old_size = old_U.numel()

            new_pos_in_old = torch.searchsorted(old_U, U_cpu)
            hit_mask = (new_pos_in_old < old_size)
            if hit_mask.any():
                hit_mask = hit_mask & (old_U[new_pos_in_old.clamp(max=max(old_size - 1, 0))] == U_cpu)
            hit_count = int(hit_mask.sum().item())
            miss_mask = ~hit_mask
            miss_ids = U_cpu[miss_mask]
            miss_count = int(miss_ids.numel())

            old_pos_in_new = torch.searchsorted(U_cpu, old_U)
            keep_old_mask = (old_pos_in_new < U_size)
            if keep_old_mask.any():
                keep_old_mask = keep_old_mask & (U_cpu[old_pos_in_new.clamp(max=max(U_size - 1, 0))] == old_U)
            evict_mask = ~keep_old_mask
            evict_ids = old_U[evict_mask]
            evict_count = int(evict_ids.numel())

            flush_info = self.flush_pending_writes(required_ids=miss_ids, force=False)

            if evict_count > 0:
                evict_pos = torch.nonzero(evict_mask, as_tuple=False).squeeze(-1)
                self._write_back_evicted(evict_ids, evict_pos, tables)

            miss_m = miss_v = miss_w = None
            if miss_count > 0:
                miss_m, miss_v, miss_w = self._fetch_rows_from_cpu(miss_ids, device, tables)

            old_cache_m = self._cache_m
            old_cache_v = self._cache_v
            old_cache_w = self._cache_w
            new_cache_m: dict[str, torch.Tensor] = {}
            new_cache_v: dict[str, torch.Tensor] = {}
            new_cache_w: dict[str, torch.Tensor] = {}

            for key in self._pin_m:
                dim = old_cache_m[key].size(1)
                prev_m = old_cache_m.get(key)
                prev_v = old_cache_v.get(key)
                prev_w = old_cache_w.get(key)
                rows_m = prev_m if (prev_m is not None and prev_m.size(0) == U_size) else torch.empty((U_size, dim), device=device, dtype=torch.float32)
                rows_v = prev_v if (prev_v is not None and prev_v.size(0) == U_size) else torch.empty((U_size, dim), device=device, dtype=torch.float32)
                rows_w = prev_w if (prev_w is not None and prev_w.size(0) == U_size) else torch.empty((U_size, dim), device=device, dtype=torch.float32)

                if hit_count > 0:
                    hit_pos_new = torch.nonzero(hit_mask, as_tuple=False).squeeze(-1)
                    hit_pos_old = new_pos_in_old[hit_mask]
                    rows_m[hit_pos_new] = old_cache_m[key][hit_pos_old]
                    rows_v[hit_pos_new] = old_cache_v[key][hit_pos_old]
                    rows_w[hit_pos_new] = old_cache_w[key][hit_pos_old]

                if miss_count > 0 and miss_m is not None and miss_v is not None and miss_w is not None:
                    miss_pos_new = torch.nonzero(miss_mask, as_tuple=False).squeeze(-1)
                    rows_m[miss_pos_new] = miss_m[key]
                    rows_v[miss_pos_new] = miss_v[key]
                    rows_w[miss_pos_new] = miss_w[key]

                new_cache_m[key] = rows_m
                new_cache_v[key] = rows_v
                new_cache_w[key] = rows_w

            self._cache_U = U_cpu.clone()
            self._cache_m = new_cache_m
            self._cache_v = new_cache_v
            self._cache_w = new_cache_w
            self.last_prefetch_stats = {
                "hit_count": hit_count,
                "miss_count": miss_count,
                "evict_count": evict_count,
                "flush_wait_ms": float(flush_info.get("wait_s", 0.0)) * 1000.0,
            }
            return self._cache_m, self._cache_v, self._cache_w

        U_size = U_step.numel()
        for key in self._pin_m:
            torch.index_select(self._m[key],  0, U_step, out=self._pin_m[key][:U_size])
            torch.index_select(self._v[key],  0, U_step, out=self._pin_v[key][:U_size])
            torch.index_select(tables[key],   0, U_step, out=self._pin_w[key][:U_size])
        pm = {k: self._pin_m[k][:U_size].to(device, non_blocking=True) for k in self._pin_m}
        pv = {k: self._pin_v[k][:U_size].to(device, non_blocking=True) for k in self._pin_v}
        pw = {k: self._pin_w[k][:U_size].to(device, non_blocking=True) for k in self._pin_w}
        self.last_prefetch_stats = {
            "hit_count": 0,
            "miss_count": int(U_size),
            "evict_count": 0,
            "flush_wait_ms": 0.0,
        }
        return pm, pv, pw

    def _fetch_rows_from_cpu(self, ids_cpu: torch.Tensor, device, tables: dict) -> tuple:
        n_rows = ids_cpu.numel()
        out_m: dict[str, torch.Tensor] = {}
        out_v: dict[str, torch.Tensor] = {}
        out_w: dict[str, torch.Tensor] = {}

        for key in self._pin_m:
            if self._overlap_cache:
                m_rows = self._m[key].index_select(0, ids_cpu)
                v_rows = self._v[key].index_select(0, ids_cpu)
                self._pin_m[key][:n_rows].copy_(m_rows)
                self._pin_v[key][:n_rows].copy_(v_rows)
            else:
                torch.index_select(self._m[key], 0, ids_cpu, out=self._pin_m[key][:n_rows])
                torch.index_select(self._v[key], 0, ids_cpu, out=self._pin_v[key][:n_rows])
            torch.index_select(tables[key], 0, ids_cpu, out=self._pin_w[key][:n_rows])
            out_m[key] = self._pin_m[key][:n_rows].to(device, non_blocking=True)
            out_v[key] = self._pin_v[key][:n_rows].to(device, non_blocking=True)
            out_w[key] = self._pin_w[key][:n_rows].to(device, non_blocking=True)

        if self._overlap_cache and self._last_seen_step.numel() > 0:
            self._last_seen_step[ids_cpu] = self._t
        return out_m, out_v, out_w

    def _write_back_evicted(self, evict_ids: torch.Tensor, evict_pos_cpu: torch.Tensor, tables: dict) -> None:
        n_evict = int(evict_ids.numel())
        if n_evict == 0:
            return
        if self._defer_writeback:
            evict_pos_gpu = evict_pos_cpu.to(device=self._cache_m[next(iter(self._cache_m))].device)
            pending_rows_m: dict[str, torch.Tensor] = {}
            pending_rows_v: dict[str, torch.Tensor] = {}
            pending_rows_w: dict[str, torch.Tensor] = {}
            for key in self._pin_m:
                dim = self._cache_m[key].size(1)
                pending_rows_m[key] = torch.empty((n_evict, dim), dtype=torch.float32, pin_memory=True)
                pending_rows_v[key] = torch.empty((n_evict, dim), dtype=torch.float32, pin_memory=True)
                pending_rows_w[key] = torch.empty((n_evict, dim), dtype=torch.float32, pin_memory=True)
            _compute_stream = torch.cuda.current_stream()
            with torch.cuda.stream(self._d2h_stream):
                self._d2h_stream.wait_stream(_compute_stream)
                for key in self._pin_m:
                    evict_m = self._cache_m[key].index_select(0, evict_pos_gpu)
                    evict_v = self._cache_v[key].index_select(0, evict_pos_gpu)
                    evict_w = self._cache_w[key].index_select(0, evict_pos_gpu)
                    pending_rows_m[key].copy_(evict_m, non_blocking=True)
                    pending_rows_v[key].copy_(evict_v, non_blocking=True)
                    pending_rows_w[key].copy_(evict_w, non_blocking=True)
            if self._overlap_cache:
                event = torch.cuda.Event()
                event.record(self._d2h_stream)
                self._pending_overlap.append({
                    "event": event,
                    "U": evict_ids.clone(),
                    "tables": tables,
                    "keys": tuple(self._pin_m.keys()),
                    "size": n_evict,
                    "rows_m": pending_rows_m,
                    "rows_v": pending_rows_v,
                    "rows_w": pending_rows_w,
                })
            else:
                self._d2h_event.record(self._d2h_stream)
                self._pending_U = evict_ids.clone()
                self._pending_tables = tables
                self._pending_keys = tuple(self._pin_m.keys())
                self._pending_size = n_evict
                self._pending_rows_m = pending_rows_m
                self._pending_rows_v = pending_rows_v
                self._pending_rows_w = pending_rows_w
            return

        for key in self._pin_m:
            evict_rows_m = self._cache_m[key].index_select(0, evict_pos_cpu.to(device=self._cache_m[key].device)).to(device="cpu")
            evict_rows_v = self._cache_v[key].index_select(0, evict_pos_cpu.to(device=self._cache_v[key].device)).to(device="cpu")
            evict_rows_w = self._cache_w[key].index_select(0, evict_pos_cpu.to(device=self._cache_w[key].device)).to(device="cpu")
            self._m[key][evict_ids] = evict_rows_m
            self._v[key][evict_ids] = evict_rows_v
            tables[key][evict_ids] = evict_rows_w
        if self._overlap_cache and self._last_seen_step.numel() > 0:
            self._last_seen_step[evict_ids] = self._t

    def flush_cache_to_master(self, tables: dict | None = None) -> None:
        if self._device != "cuda":
            return
        if tables is None:
            tables = self._tables_ref
        self.flush_pending_writes()
        if not self._overlap_cache or self._cache_U is None or self._cache_U.numel() == 0:
            return
        ids = self._cache_U
        for key in self._cache_m:
            self._m[key][ids] = self._cache_m[key].to(device="cpu")
            self._v[key][ids] = self._cache_v[key].to(device="cpu")
            tables[key][ids] = self._cache_w[key].to(device="cpu")
        if self._last_seen_step.numel() > 0:
            self._last_seen_step[ids] = self._t

    @staticmethod
    def _sorted_intersects(a: torch.Tensor | None, b: torch.Tensor | None) -> bool:
        if a is None or b is None or a.numel() == 0 or b.numel() == 0:
            return False
        pos = torch.searchsorted(a, b)
        in_range = pos < a.numel()
        if not bool(in_range.any()):
            return False
        pos_valid = pos[in_range]
        b_valid = b[in_range]
        return bool((a[pos_valid] == b_valid).any())

    def _apply_pending_entry(self, entry: dict) -> dict:
        U = entry["U"]
        U_size = int(entry["size"]) if int(entry["size"]) > 0 else U.numel()
        keys = entry["keys"]
        tables = entry["tables"]
        rows_m = entry["rows_m"]
        rows_v = entry["rows_v"]
        rows_w = entry["rows_w"]
        applied_update_norms: dict = {}
        for key in keys:
            rows_m_k = rows_m[key]
            rows_v_k = rows_v[key]
            rows_w_k = rows_w[key]
            applied_update_norms[key] = (rows_w_k - tables[key][U]).norm()
            self._m[key][U] = rows_m_k[:U_size]
            self._v[key][U] = rows_v_k[:U_size]
            tables[key][U] = rows_w_k[:U_size]
        return applied_update_norms

    def flush_pending_writes(self, required_ids: torch.Tensor | None = None, force: bool = True) -> dict:
        """Commit the previous step's deferred D2H results into the CPU master tables.

        Waits for the D2H DMA (which was fired asynchronously at the end of the
        previous step) to complete, then scatter-writes the updated rows into the
        full-vocab ``_m``, ``_v``, and master weight tables.

        In overlap-cache mode this supports hazard-aware flushes:
        - force=False: only block when ``required_ids`` intersects pending rows,
          otherwise opportunistically retire entries that already completed.
        - force=True : flush all pending entries (used for checkpoint/eval safety).
        """
        t_wait_s = 0.0
        if self._overlap_cache and self._device == "cuda":
            if not self._pending_overlap:
                return {"wait_s": 0.0}
            applied_update_norms: dict = {}
            keep_entries: list[dict] = []
            for entry in self._pending_overlap:
                needs_now = self._sorted_intersects(entry["U"], required_ids)
                ready = bool(entry["event"].query())
                should_apply = force or needs_now or ready
                if should_apply:
                    if not ready:
                        t_sync0 = time.time()
                        entry["event"].synchronize()
                        t_wait_s += (time.time() - t_sync0)
                    applied_update_norms.update(self._apply_pending_entry(entry))
                else:
                    keep_entries.append(entry)

            # Keep queue bounded even when caller never requests intersecting rows.
            while len(keep_entries) > self._pending_overlap_limit:
                entry = keep_entries.pop(0)
                if not bool(entry["event"].query()):
                    t_sync0 = time.time()
                    entry["event"].synchronize()
                    t_wait_s += (time.time() - t_sync0)
                applied_update_norms.update(self._apply_pending_entry(entry))

            self._pending_overlap = keep_entries
            applied_update_norms["wait_s"] = t_wait_s
            return applied_update_norms

        if (
            self._device != "cuda"
            or not hasattr(self, '_pending_U')
            or self._pending_U is None
            or self._pending_keys is None
            or len(self._pending_keys) == 0
        ):
            return {"wait_s": 0.0}
        # Block CPU until the D2H DMA stream has finished copying data into _pin_m/v/w.
        t_sync0 = time.time()
        self._d2h_event.synchronize()
        t_wait_s += (time.time() - t_sync0)
        U = self._pending_U
        U_size = int(self._pending_size) if self._pending_size > 0 else U.numel()
        applied_update_norms: dict = {}
        for key in self._pending_keys:
            rows_m = self._pending_rows_m[key] if self._pending_rows_m is not None else self._pin_m[key][:U_size]
            rows_v = self._pending_rows_v[key] if self._pending_rows_v is not None else self._pin_v[key][:U_size]
            rows_w = self._pending_rows_w[key] if self._pending_rows_w is not None else self._pin_w[key][:U_size]
            applied_update_norms[key] = (rows_w - self._pending_tables[key][U]).norm()
            self._m[key][U] = rows_m
            self._v[key][U] = rows_v
        if self._pending_tables is not None:
            for key in self._pending_keys:
                rows_w = self._pending_rows_w[key] if self._pending_rows_w is not None else self._pin_w[key][:U_size]
                self._pending_tables[key][U] = rows_w
        self._pending_U = None
        self._pending_tables = None
        self._pending_keys = None
        self._pending_size = 0
        self._pending_rows_m = None
        self._pending_rows_v = None
        self._pending_rows_w = None
        applied_update_norms["wait_s"] = t_wait_s
        return applied_update_norms

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
        return_update_norms: bool = False,
    ) -> dict:
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
        update_norms: dict = {}

        use_gpu = (
            self._device == "cuda"
            and prefetched_m is not None
            and prefetched_v is not None
            and prefetched_w is not None
        )
        updated_keys: list[str] = []

        

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
                # Guard m/v before D2H: if prefetched_m/v had NaN from a prior corrupted
                # step, those NaN propagate into m_rows/v_rows and would be written to
                # CPU master state, poisoning all future prefetches for those rows.
                # nan_to_num here breaks the recirculation loop.
                m_rows = m_rows.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
                v_rows = v_rows.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

                m_hat = m_rows / bc1
                v_hat = v_rows / bc2
                # clamp_min before sqrt: subnormal f32 values after bf16→f32 cast and
                # repeated global decay can produce negative v_hat on some GPU archs,
                # causing sqrt to return NaN.  eps (=1e-10) is the floor anyway.
                v_hat = v_hat.clamp_min_(self.eps)
                update = m_hat / (v_hat.sqrt_() + self.eps)
                # Standard AdamW: weight decay on the pre-update weight, then subtract step.
                w_updated = prefetched_w[key] * (1.0 - lr * self.weight_decay) - lr * update
                if return_update_norms:
                    update_norms[key] = (w_updated - prefetched_w[key]).norm()
                # Final NaN guard on weights — should never trigger after the fixes above,
                # but kept as a last-resort safety net to prevent master weight corruption.
                w_updated = w_updated.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

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
                if self._defer_writeback and not self._overlap_cache:
                    _compute_stream = torch.cuda.current_stream()
                    U_size = U_step.numel()
                    with torch.cuda.stream(self._d2h_stream):
                        self._d2h_stream.wait_stream(_compute_stream)
                        self._pin_m[key][:U_size].copy_(m_rows, non_blocking=True)
                        self._pin_v[key][:U_size].copy_(v_rows, non_blocking=True)
                        self._pin_w[key][:U_size].copy_(w_updated, non_blocking=True)
                    updated_keys.append(key)
                else:
                    if not self._overlap_cache:
                        # Sparse-row Adam semantics: only rows in U_step are updated.
                        # Missing rows keep their previous m/v state unchanged.
                        self._m[key][U_step] = m_rows.to(device="cpu")
                        self._v[key][U_step] = v_rows.to(device="cpu")
                        tables[key][U_step] = w_updated.to(device="cpu")

                if self._overlap_cache and self._cache_U is not None:
                    self._cache_m[key].copy_(m_rows)
                    self._cache_v[key].copy_(v_rows)
                    self._cache_w[key].copy_(w_updated)
                    if self._last_seen_step.numel() > 0:
                        self._last_seen_step[U_step] = t

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

                # Sparse-row Adam semantics: only rows in U_step are updated.
                # Missing rows keep their previous m/v state unchanged.
                self._m[key][U_step] = m_rows
                self._v[key][U_step] = v_rows

                m_hat = m_rows / bc1
                v_hat = v_rows / bc2
                v_hat = v_hat.clamp_min_(self.eps)

                update = m_hat / (v_hat.sqrt_() + self.eps)
                w_rows = tables[key][U_step]
                w_updated = w_rows * (1.0 - lr * self.weight_decay) - lr * update
                if return_update_norms:
                    update_norms[key] = (w_updated - w_rows).norm()
                # Standard AdamW: weight decay on the pre-update weight, then subtract step.
                tables[key][U_step] = w_updated

        # After the per-key loop: record the D2H event and store pending state.
        # flush_pending_writes() at the start of the *next* step will synchronize
        # on this event before reading from _m/_v/tables.
        if use_gpu and self._defer_writeback and not self._overlap_cache:
            self._d2h_event.record(self._d2h_stream)
            if len(updated_keys) > 0:
                self._pending_U = U_step
                self._pending_tables = tables
                self._pending_keys = tuple(updated_keys)
            else:
                self._pending_U = None
                self._pending_tables = None
                self._pending_keys = None
        return update_norms

    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        # Flush any in-flight D2H writes and overlap-cache rows so state is consistent.
        self.flush_cache_to_master(self._tables_ref)
        state = {
            "t": self._t,
            "m": {k: v.clone() for k, v in self._m.items()},
            "v": {k: v.clone() for k, v in self._v.items()},
        }
        if self._overlap_cache and self._last_seen_step.numel() > 0:
            state["last_seen_step"] = self._last_seen_step.clone()
        return state

    def load_state_dict(self, state: dict) -> None:
        self._t = int(state["t"])
        for k in self._m:
            if k in state["m"]:
                self._m[k].copy_(state["m"][k])
            if k in state["v"]:
                self._v[k].copy_(state["v"][k])
        if self._overlap_cache and self._last_seen_step.numel() > 0 and "last_seen_step" in state:
            self._last_seen_step.copy_(state["last_seen_step"])
        if self._device == "cuda":
            self._cache_U = None
            self._cache_m = {}
            self._cache_v = {}
            self._cache_w = {}
            self._pending_overlap = []
            self.last_prefetch_stats = {}
            self._pending_U = None
            self._pending_tables = None
            self._pending_keys = None
            self._pending_size = 0
            self._pending_rows_m = None
            self._pending_rows_v = None
            self._pending_rows_w = None
