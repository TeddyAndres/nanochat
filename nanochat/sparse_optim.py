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
                # ---- GPU arithmetic path ----
                # grad is already GPU bf16 — upcast in-place on GPU, no H2D.
                # prefetched_m/v/w are GPU f32 row slices fetched before the forward pass.
                g = grad.detach().to(dtype=torch.float32)

                m_rows = beta1 * prefetched_m[key] + (1.0 - beta1) * g
                v_rows = beta2 * prefetched_v[key] + (1.0 - beta2) * g.square()

                m_hat = m_rows / bc1
                v_hat = v_rows / bc2
                update = m_hat / (v_hat.sqrt_() + self.eps)
                w_updated = (1.0 - lr * self.weight_decay) * prefetched_w[key] - lr * update

                # D2H writeback: synchronous (~few ms for |U_step| rows).
                # .to("cpu") blocks until the preceding GPU Adam kernels AND the DMA
                # complete, so no explicit synchronize() is needed before this call.
                self._m[key][U_step] = m_rows.to("cpu")
                self._v[key][U_step] = v_rows.to("cpu")
                tables[key][U_step] = w_updated.to("cpu")

            else:
                # ---- CPU arithmetic path (original) ----
                # Move grad to CPU float32 (blocking — GPU has already finished backward)
                g = grad.detach().to(dtype=torch.float32, device="cpu")

                m_rows = self._m[key][U_step]
                v_rows = self._v[key][U_step]

                m_rows = beta1 * m_rows + (1.0 - beta1) * g
                v_rows = beta2 * v_rows + (1.0 - beta2) * g.square()

                self._m[key][U_step] = m_rows
                self._v[key][U_step] = v_rows

                m_hat = m_rows / bc1
                v_hat = v_rows / bc2

                update = m_hat / (v_hat.sqrt_() + self.eps)
                w_rows = tables[key][U_step]
                tables[key][U_step] = (1.0 - lr * self.weight_decay) * w_rows - lr * update

    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
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
