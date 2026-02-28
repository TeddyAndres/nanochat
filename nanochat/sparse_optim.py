"""
Vocabulary-row-sparse optimizer for sparse-mode training.

VocabRowAdamW maintains per-row Adam m/v state for every vocab table
(wte, value_embeds, lm_head) entirely on CPU.  Each optimizer step it
receives the accumulated gradient for only the |U_step| active rows
(moved from GPU to CPU), and updates exactly those rows in the master
weight tensors — no sparse COO tensors, no SparseAdam, no custom
autograd Functions.
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
    """

    def __init__(
        self,
        tables: dict,
        initial_lrs: dict,
        betas: tuple = (0.8, 0.95),
        eps: float = 1e-10,
        weight_decay: float = 0.0,
    ) -> None:
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self._initial_lrs: dict = dict(initial_lrs)
        self._t: int = 0

        self._m: dict = {k: torch.zeros_like(w, dtype=torch.float32) for k, w in tables.items()}
        self._v: dict = {k: torch.zeros_like(w, dtype=torch.float32) for k, w in tables.items()}

    @torch.no_grad()
    def step(
        self,
        U_step: torch.Tensor,
        grads: dict,
        tables: dict,
        lr_multiplier: float = 1.0,
    ) -> None:
        """Apply one Adam update to the U_step rows of every table.

        U_step       : CPU sorted LongTensor of active global token indices.
        grads        : {key: GPU bf16 or CPU tensor} shape (|U_step|, dim).
        tables       : {key: master_weight_cpu_f32} — same references as at __init__.
        lr_multiplier: LR schedule multiplier (lrm); effective lr = initial_lrs[key] * lr_multiplier.
        """
        beta1, beta2 = self.betas
        self._t += 1
        t = self._t
        bc1 = 1.0 - beta1 ** t
        bc2 = 1.0 - beta2 ** t

        for key, grad in grads.items():
            if grad is None:
                continue

            lr = self._initial_lrs[key] * lr_multiplier

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
