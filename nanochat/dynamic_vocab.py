import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn


COLD_LOGIT_BIAS_CLAMP_MIN = -6.0
COLD_LOGIT_BIAS_CLAMP_MAX = 6.0


@dataclass
class DynamicVocabStep:
    active_ids_cpu: torch.Tensor
    active_vocab: Optional[dict]
    optimizer_state: Optional[dict]
    unique_count: int
    live_count: int = 0
    bytes_h2d: int = 0
    h2d_ms: float = 0.0
    bytes_d2h: int = 0
    d2h_ms: float = 0.0
    optimizer_ms: float = 0.0
    grad_accum_queue_ms: float = 0.0
    grad_accum_queue_count: int = 0
    grad_accum_flush_ms: float = 0.0
    grad_accum_stage_ms: float = 0.0
    grad_accum_apply_ms: float = 0.0
    grad_accum_restore_ms: float = 0.0
    grad_accum_writeback_ms: float = 0.0
    grad_accum_resident_count: int = 0
    d2h_launch_ms: float = 0.0
    d2h_sync_ms: float = 0.0
    cpu_writeback_ms: float = 0.0
    active_param_bytes: int = 0
    active_grad_bytes: int = 0
    active_optimizer_bytes: int = 0
    u_capacity: int = 0
    stage_count: int = 0
    writeback_count: int = 0
    cold_bias_clamped_count: int = 0
    cold_bias_abs_max: float = 0.0
    hot_activation_counts_cpu: Optional[torch.Tensor] = None
    active_slot_ids_cpu: Optional[torch.Tensor] = None
    active_mask_cpu: Optional[torch.Tensor] = None
    slot_to_global_cpu: Optional[torch.Tensor] = None
    stage_ids_cpu: Optional[torch.Tensor] = None
    stage_slot_ids_cpu: Optional[torch.Tensor] = None
    writeback_ids_cpu: Optional[torch.Tensor] = None
    writeback_slot_ids_cpu: Optional[torch.Tensor] = None
    grad_accum_ids_cpu: Optional[torch.Tensor] = None
    grad_accum_steps: int = 1
    grad_accum_micro_step: int = 0
    is_grad_accum_boundary: bool = True
    is_last_step: bool = False
    fixed_u_mode: bool = False


class DynamicVocabRuntime:
    """CPU-master / GPU-active runtime for vocab-dimension tables.

    First-pass constraints:
    - single GPU only
    - grad_accum_steps == 1 for the non-manifest first-pass sparse path
    - vocab tables are offloaded to CPU masters between optimizer steps
    - optimizer moments for vocab tables live on CPU and are staged to GPU only for active rows
    """

    def __init__(
        self,
        model,
        device,
        embedding_lr,
        unembedding_lr,
        first_hot_unembedding_lr=None,
        hot_unembedding_ramp_activations=0,
        hot_unembedding_ramp_start_lr=None,
        fixed_u_max=None,
        grad_accum_u_max=None,
        cold_bias_reference_tokens=2**19,
        value_embedding_lr=None,
        adam_betas=(0.8, 0.95),
        eps=1e-10,
        weight_decay=0.0,
    ):
        self.model = model
        self.device = torch.device(device)
        self.use_cuda = self.device.type == "cuda"
        self.beta1, self.beta2 = adam_betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.cold_bias_reference_tokens = float(cold_bias_reference_tokens)
        self.first_hot_unembedding_lr = None if first_hot_unembedding_lr is None or first_hot_unembedding_lr <= 0.0 else float(first_hot_unembedding_lr)
        self.hot_unembedding_ramp_activations = max(0, int(hot_unembedding_ramp_activations))
        self.hot_unembedding_ramp_start_lr = None if hot_unembedding_ramp_start_lr is None or hot_unembedding_ramp_start_lr <= 0.0 else float(hot_unembedding_ramp_start_lr)
        self.fixed_u_max = 0 if fixed_u_max is None else int(fixed_u_max)
        self.fixed_u_mode = self.fixed_u_max > 0
        default_grad_accum_u_max = self.fixed_u_max if self.fixed_u_mode else 0
        self.grad_accum_u_max = default_grad_accum_u_max if grad_accum_u_max is None else int(grad_accum_u_max)
        if self.grad_accum_u_max < 0:
            raise ValueError(f"grad_accum_u_max must be non-negative, got {self.grad_accum_u_max}")
        if self.fixed_u_mode and self.grad_accum_u_max < self.fixed_u_max:
            raise ValueError(
                f"grad_accum_u_max must be at least fixed_u_max in fixed-U sparse mode, got grad_accum_u_max={self.grad_accum_u_max}, fixed_u_max={self.fixed_u_max}"
            )
        value_embedding_lr = embedding_lr if value_embedding_lr is None else value_embedding_lr

        self.table_specs = {
            "wte": {
                "param": model.transformer.wte.weight,
                "lr": embedding_lr,
            },
            "lm_head": {
                "param": model.lm_head.weight,
                "lr": unembedding_lr,
            },
        }
        for layer_name, embed in model.value_embeds.items():
            self.table_specs[f"value_embeds.{layer_name}"] = {
                "param": embed.weight,
                "lr": value_embedding_lr,
            }

        self.state = {}
        self.runtime_step = 0
        self.last_seen_step_cpu = torch.full((model.config.vocab_size,), -1, dtype=torch.long)
        self.hot_activation_count_cpu = torch.zeros((model.config.vocab_size,), dtype=torch.long)
        self._cpu_receive_buffers = {}
        for spec in self.table_specs.values():
            param = spec["param"]
            param.data = param.data.detach().to("cpu")
            if param.grad is not None:
                param.grad = None
            self.state[param] = {
                "step": 0,
                "exp_avg": torch.zeros_like(param.data, device="cpu"),
                "exp_avg_sq": torch.zeros_like(param.data, device="cpu"),
            }

        self.fixed_params = {}
        self.fixed_optimizer_state = {}
        self.fixed_active_vocab = None
        self.fixed_logit_mask = None
        self.fixed_cold_logit_bias = None
        self.fixed_slot_to_global_cpu = None
        self.fixed_active_mask_cpu = None
        self._fixed_live_state = False
        self._grad_accum_ids_cpu = None
        self._grad_accum_global_to_local_cpu = None
        self._grad_accum_buffers = None
        self._grad_accum_count = 0
        self._grad_accum_live = False
        self._grad_accum_stage_count = 0
        self._grad_accum_non_live_chunks = []
        self._grad_accum_cached_union_mask_cpu = None
        self._grad_accum_pending_transfers = []
        self._grad_accum_transfer_stream = torch.cuda.Stream(device=self.device) if self.use_cuda else None
        self._cpu_writeback_executor = ThreadPoolExecutor(max_workers=1)
        self._pending_cpu_writeback_future = None
        self._pending_cpu_writeback_mask_cpu = None
        if self.fixed_u_mode:
            self.fixed_slot_to_global_cpu = torch.full((self.fixed_u_max,), -1, dtype=torch.long)
            self.fixed_active_mask_cpu = torch.zeros(self.fixed_u_max, dtype=torch.bool)
            for name, spec in self.table_specs.items():
                param = spec["param"]
                shape = (self.fixed_u_max,) + tuple(param.shape[1:])
                self.fixed_params[name] = nn.Parameter(
                    torch.zeros(shape, device=self.device, dtype=param.dtype),
                    requires_grad=True,
                )
                self.fixed_optimizer_state[name] = {
                    "exp_avg": torch.zeros(shape, device=self.device, dtype=param.dtype),
                    "exp_avg_sq": torch.zeros(shape, device=self.device, dtype=param.dtype),
                }
            self.fixed_logit_mask = torch.zeros(self.fixed_u_max, dtype=torch.bool, device=self.device)
            self.fixed_cold_logit_bias = torch.zeros(self.fixed_u_max, dtype=torch.float32, device=self.device)
            self.fixed_active_vocab = {
                "wte": self.fixed_params["wte"],
                "lm_head": self.fixed_params["lm_head"],
                "value_embeds": {
                    name.split(".", 1)[1]: self.fixed_params[name]
                    for name in self.fixed_params
                    if name.startswith("value_embeds.")
                },
                "logit_mask": self.fixed_logit_mask,
                "cold_logit_bias": self.fixed_cold_logit_bias,
            }

    def state_dict(self):
        serialized = {}
        for name, spec in self.table_specs.items():
            param = spec["param"]
            state = self.state[param]
            serialized[name] = {
                "step": state["step"],
                "exp_avg": state["exp_avg"],
                "exp_avg_sq": state["exp_avg_sq"],
                "lr": spec["lr"],
            }
        return {
            "version": 3,
            "runtime_step": self.runtime_step,
            "last_seen_step_cpu": self.last_seen_step_cpu,
            "hot_activation_count_cpu": self.hot_activation_count_cpu,
            "tables": serialized,
        }

    def load_state_dict(self, state_dict):
        self.runtime_step = int(state_dict.get("runtime_step", 0))
        if "last_seen_step_cpu" in state_dict:
            self.last_seen_step_cpu.copy_(state_dict["last_seen_step_cpu"].to(device="cpu", dtype=torch.long))
        else:
            self.last_seen_step_cpu.fill_(-1)
        if "hot_activation_count_cpu" in state_dict:
            self.hot_activation_count_cpu.copy_(state_dict["hot_activation_count_cpu"].to(device="cpu", dtype=torch.long))
        else:
            self.hot_activation_count_cpu.zero_()
        tables = state_dict.get("tables", {})
        for name, table_state in tables.items():
            spec = self.table_specs[name]
            param = spec["param"]
            state = self.state[param]
            state["step"] = int(table_state["step"])
            state["exp_avg"].copy_(table_state["exp_avg"].to("cpu"))
            state["exp_avg_sq"].copy_(table_state["exp_avg_sq"].to("cpu"))

    def _capture_cold_steps_cpu(self, active_ids_cpu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        active_ids_cpu = active_ids_cpu.detach().to(device="cpu", dtype=torch.long)
        if active_ids_cpu.numel() == 0:
            return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
        last_seen = self.last_seen_step_cpu.index_select(0, active_ids_cpu)
        hot_activation_counts_cpu = self.hot_activation_count_cpu.index_select(0, active_ids_cpu)
        cold_steps_cpu = (self.runtime_step - last_seen - 1).clamp_min_(0)
        cold_steps_cpu.masked_fill_(last_seen < 0, 0)
        self.last_seen_step_cpu.index_fill_(0, active_ids_cpu, self.runtime_step)
        self.hot_activation_count_cpu.index_copy_(0, active_ids_cpu, hot_activation_counts_cpu + 1)
        return cold_steps_cpu, hot_activation_counts_cpu

    def _get_lm_head_row_lr(self, base_lr: float, hot_activation_counts_cpu: Optional[torch.Tensor], device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        if hot_activation_counts_cpu is None:
            return None
        hot_activation_counts = hot_activation_counts_cpu.detach().to(device=device, dtype=torch.long)
        row_lr = None
        if self.hot_unembedding_ramp_activations > 0 and self.hot_unembedding_ramp_start_lr is not None:
            eligible = hot_activation_counts < self.hot_unembedding_ramp_activations
            if eligible.any():
                start_lr = min(base_lr, self.hot_unembedding_ramp_start_lr)
                row_lr = torch.full((hot_activation_counts.numel(),), float(base_lr), dtype=dtype, device=device)
                if self.hot_unembedding_ramp_activations == 1:
                    row_lr.masked_fill_(eligible, start_lr)
                else:
                    hot_progress = hot_activation_counts[eligible].to(dtype=torch.float32) / float(self.hot_unembedding_ramp_activations - 1)
                    ramp_lr = start_lr + (base_lr - start_lr) * hot_progress
                    row_lr[eligible] = ramp_lr.to(dtype=dtype, device=device)
                return row_lr
        if self.first_hot_unembedding_lr is not None:
            first_hot_mask = hot_activation_counts == 0
            if first_hot_mask.any():
                first_hot_lr = min(base_lr, self.first_hot_unembedding_lr)
                row_lr = torch.full((hot_activation_counts.numel(),), float(base_lr), dtype=dtype, device=device)
                row_lr.masked_fill_(first_hot_mask, first_hot_lr)
                return row_lr
        return None

    def _get_dense_cold_steps_cpu(self) -> torch.Tensor:
        cold_steps_cpu = (self.runtime_step - self.last_seen_step_cpu - 1).clamp_min(0)
        return cold_steps_cpu.masked_fill(self.last_seen_step_cpu < 0, 0)

    def _compute_cold_logit_bias_cpu(
        self,
        cold_steps_cpu: torch.Tensor,
        cold_bias_scale: float = 0.0,
        cold_bias_tokens_per_step: Optional[int] = None,
    ) -> torch.Tensor:
        cold_bias, _, _ = self._compute_cold_logit_bias_with_stats_cpu(
            cold_steps_cpu,
            cold_bias_scale=cold_bias_scale,
            cold_bias_tokens_per_step=cold_bias_tokens_per_step,
        )
        return cold_bias

    def _compute_cold_logit_bias_with_stats_cpu(
        self,
        cold_steps_cpu: torch.Tensor,
        cold_bias_scale: float = 0.0,
        cold_bias_tokens_per_step: Optional[int] = None,
    ) -> tuple[torch.Tensor, int, float]:
        cold_steps_cpu = cold_steps_cpu.detach().to(device="cpu", dtype=torch.float32)
        if cold_steps_cpu.numel() == 0:
            return torch.empty(0, dtype=torch.float32), 0, 0.0
        if cold_bias_scale <= 0.0 or cold_bias_tokens_per_step is None or cold_bias_tokens_per_step <= 0:
            return torch.zeros_like(cold_steps_cpu), 0, 0.0
        reference_tokens = max(self.cold_bias_reference_tokens, 1.0)
        cold_tokens = cold_steps_cpu * float(cold_bias_tokens_per_step)
        cold_bias = float(cold_bias_scale) * torch.log1p(cold_tokens / reference_tokens)
        clamped_mask = (cold_bias < COLD_LOGIT_BIAS_CLAMP_MIN) | (cold_bias > COLD_LOGIT_BIAS_CLAMP_MAX)
        cold_bias.clamp_(min=COLD_LOGIT_BIAS_CLAMP_MIN, max=COLD_LOGIT_BIAS_CLAMP_MAX)
        cold_bias_abs_max = float(cold_bias.abs().max().item()) if cold_bias.numel() > 0 else 0.0
        return cold_bias, int(clamped_mask.sum().item()), cold_bias_abs_max

    def get_dense_cold_logit_bias(
        self,
        cold_bias_scale: float = 0.0,
        cold_bias_tokens_per_step: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        if cold_bias_scale <= 0.0 or cold_bias_tokens_per_step is None or cold_bias_tokens_per_step <= 0:
            return None
        cold_steps_cpu = self._get_dense_cold_steps_cpu()
        cold_bias_cpu = self._compute_cold_logit_bias_cpu(
            cold_steps_cpu,
            cold_bias_scale=cold_bias_scale,
            cold_bias_tokens_per_step=cold_bias_tokens_per_step,
        )
        return cold_bias_cpu.to(self.device, non_blocking=self.use_cuda)

    @contextmanager
    def materialize_dense_params(self):
        """Temporarily move full vocab tables to the model device for dense eval/inference."""
        original_data = {}
        try:
            self._flush_pending_cpu_writeback()
            if self._grad_accum_live:
                raise RuntimeError("Cannot materialize dense params while sparse accumulated gradients are pending")
            self.flush_active_to_cpu()
            for spec in self.table_specs.values():
                param = spec["param"]
                original_data[param] = param.data
                param.data = param.data.to(self.device, non_blocking=self.use_cuda)
            if self.use_cuda:
                torch.cuda.synchronize(self.device)
            yield self.model
        finally:
            for spec in self.table_specs.values():
                param = spec["param"]
                param.data = original_data[param]
            if self.use_cuda:
                torch.cuda.synchronize(self.device)
                torch.cuda.empty_cache()

    def _clear_fixed_grads(self):
        if not self.fixed_u_mode:
            return
        for param in self.fixed_params.values():
            param.grad = None

    def _flush_pending_cpu_writeback(self, required_ids_cpu: Optional[torch.Tensor] = None) -> None:
        future = self._pending_cpu_writeback_future
        if future is None:
            return
        if required_ids_cpu is not None and self._pending_cpu_writeback_mask_cpu is not None:
            required_ids_cpu = required_ids_cpu.detach().to(device="cpu", dtype=torch.long)
            if required_ids_cpu.numel() > 0 and not self._pending_cpu_writeback_mask_cpu[required_ids_cpu].any() and not future.done():
                return
        future.result()
        self._pending_cpu_writeback_future = None
        self._pending_cpu_writeback_mask_cpu = None

    def _zero_fixed_grad_slots_(self, slot_ids_cpu: Optional[torch.Tensor]) -> None:
        if not self.fixed_u_mode or slot_ids_cpu is None:
            return
        slot_ids_cpu = slot_ids_cpu.detach().to(device="cpu", dtype=torch.long)
        if slot_ids_cpu.numel() == 0:
            return
        slot_ids_device = slot_ids_cpu.to(self.device)
        for param in self.fixed_params.values():
            if param.grad is not None:
                param.grad.index_fill_(0, slot_ids_device, 0)

    def _invalidate_fixed_live_state(self):
        if not self.fixed_u_mode:
            return
        self._fixed_live_state = False
        if self.fixed_slot_to_global_cpu is not None:
            self.fixed_slot_to_global_cpu.fill_(-1)
        if self.fixed_active_mask_cpu is not None:
            self.fixed_active_mask_cpu.zero_()
        if self.fixed_logit_mask is not None:
            self.fixed_logit_mask.zero_()
        self._clear_fixed_grads()

    def _clear_grad_accum_window(self):
        self._grad_accum_pending_transfers = []
        self._grad_accum_ids_cpu = None
        self._grad_accum_global_to_local_cpu = None
        self._grad_accum_count = 0
        self._grad_accum_live = False
        self._grad_accum_non_live_chunks = []
        self._grad_accum_cached_union_mask_cpu = None

    def _ensure_grad_accum_buffers(self, capacity: int) -> None:
        needs_new = self._grad_accum_buffers is None
        if not needs_new:
            assert self._grad_accum_buffers is not None
            for name, spec in self.table_specs.items():
                param = spec["param"]
                buffer = self._grad_accum_buffers.get(name)
                expected_rank = param.dim()
                if (
                    buffer is None or
                    buffer.dtype != param.dtype or
                    buffer.dim() != expected_rank or
                    buffer.size(0) < capacity or
                    any(buffer.size(dim) != param.size(dim) for dim in range(1, expected_rank))
                ):
                    needs_new = True
                    break
        if needs_new:
            self._grad_accum_buffers = {}
            for name, spec in self.table_specs.items():
                param = spec["param"]
                shape = (capacity,) + tuple(param.shape[1:])
                self._grad_accum_buffers[name] = torch.zeros(shape, dtype=param.dtype, device=self.device)

    def _flush_pending_grad_accum_transfers(self, wait: bool) -> None:
        if not self._grad_accum_pending_transfers:
            return
        if self._grad_accum_buffers is None:
            raise RuntimeError("Sparse grad accumulation buffers are missing while transfers are pending")

        remaining = []
        for pending in self._grad_accum_pending_transfers:
            ready_event = pending.get("ready_event")
            if self.use_cuda and ready_event is not None:
                if not wait and not ready_event.query():
                    remaining.append(pending)
                    continue
                ready_event.synchronize()

            accum_row_ids_cpu = pending["accum_row_ids_cpu"]
            for name, grad_rows_cpu in pending["buffers"].items():
                self._grad_accum_buffers[name].index_add_(0, accum_row_ids_cpu, grad_rows_cpu)

        self._grad_accum_pending_transfers = remaining

    def _queue_grad_accum_transfer_(self, accum_row_ids_cpu: torch.Tensor, active_slot_ids_device: torch.Tensor, grad_map: dict[str, torch.Tensor]) -> None:
        if not grad_map:
            return

        if self._grad_accum_buffers is None:
            raise RuntimeError("Sparse grad accumulation buffers are missing while queueing transfers")

        accum_row_ids_cpu = accum_row_ids_cpu.detach().to(device="cpu", dtype=torch.long)
        if not self.use_cuda:
            for name, grad in grad_map.items():
                grad_rows_cpu = grad.detach().index_select(0, active_slot_ids_device).to(device="cpu", dtype=self._grad_accum_buffers[name].dtype)
                self._grad_accum_buffers[name].index_add_(0, accum_row_ids_cpu, grad_rows_cpu)
            return

        assert self._grad_accum_transfer_stream is not None

        pending = {
            "accum_row_ids_cpu": accum_row_ids_cpu,
            "buffers": {},
            "source_refs": [active_slot_ids_device],
        }
        current_stream = torch.cuda.current_stream(self.device)
        with torch.cuda.stream(self._grad_accum_transfer_stream):
            self._grad_accum_transfer_stream.wait_stream(current_stream)
            for name, grad in grad_map.items():
                grad_detached = grad.detach()
                grad_rows = grad_detached.index_select(0, active_slot_ids_device)
                grad_rows_cpu = torch.empty(
                    tuple(grad_rows.shape),
                    dtype=self._grad_accum_buffers[name].dtype,
                    pin_memory=True,
                )
                grad_rows_cpu.copy_(grad_rows, non_blocking=True)
                pending["buffers"][name] = grad_rows_cpu
                pending["source_refs"].append(grad_detached)
                pending["source_refs"].append(grad_rows)
            ready_event = torch.cuda.Event()
            ready_event.record(self._grad_accum_transfer_stream)
        pending["ready_event"] = ready_event
        self._grad_accum_pending_transfers.append(pending)

    def _start_grad_accum_window(self, grad_accum_ids_cpu: torch.Tensor):
        grad_accum_ids_cpu = grad_accum_ids_cpu.detach().to(device="cpu", dtype=torch.long)
        grad_accum_count = int(grad_accum_ids_cpu.numel())
        capacity = self.grad_accum_u_max if self.grad_accum_u_max > 0 else grad_accum_count
        if grad_accum_count > capacity:
            raise ValueError(
                f"Sparse grad accumulation overflow: need {grad_accum_count} rows, capacity is {capacity}"
            )
        self._grad_accum_ids_cpu = grad_accum_ids_cpu
        self._grad_accum_count = grad_accum_count
        global_to_local = torch.full((self.model.config.vocab_size,), -1, dtype=torch.long)
        if grad_accum_count > 0:
            global_to_local[grad_accum_ids_cpu] = torch.arange(grad_accum_count, dtype=torch.long)
        self._grad_accum_global_to_local_cpu = global_to_local
        self._ensure_grad_accum_buffers(capacity)
        assert self._grad_accum_buffers is not None
        for buffer in self._grad_accum_buffers.values():
            if grad_accum_count > 0:
                buffer[:grad_accum_count].zero_()
        self._grad_accum_live = True
        self._grad_accum_stage_count = 0
        self._grad_accum_non_live_chunks = []
        self._grad_accum_cached_union_mask_cpu = torch.zeros(grad_accum_count, dtype=torch.bool)

    def _cache_grad_accum_leaving_rows_(self, accum_row_ids_cpu: torch.Tensor, leaving_slot_ids_cpu: torch.Tensor) -> None:
        if accum_row_ids_cpu.numel() == 0:
            return
        if self._grad_accum_cached_union_mask_cpu is None:
            raise RuntimeError("Sparse grad accumulation cache mask is missing while caching leaving rows")

        accum_row_ids_cpu = accum_row_ids_cpu.detach().to(device="cpu", dtype=torch.long)
        leaving_slot_ids_cpu = leaving_slot_ids_cpu.detach().to(device="cpu", dtype=torch.long)
        new_mask = ~self._grad_accum_cached_union_mask_cpu[accum_row_ids_cpu]
        if not new_mask.any():
            return

        cache_row_ids_cpu = accum_row_ids_cpu[new_mask]
        cache_slot_ids_cpu = leaving_slot_ids_cpu[new_mask]
        self._grad_accum_cached_union_mask_cpu[cache_row_ids_cpu] = True

        cache_slot_ids_device = cache_slot_ids_cpu.to(self.device)
        assert self._grad_accum_ids_cpu is not None
        chunk = {
            "union_row_ids_cpu": cache_row_ids_cpu,
            "union_row_ids_device": cache_row_ids_cpu.to(self.device),
            "global_ids_cpu": self._grad_accum_ids_cpu.index_select(0, cache_row_ids_cpu),
            "tables": {},
        }
        for name in self.table_specs:
            chunk["tables"][name] = {
                "param": nn.Parameter(
                    self.fixed_params[name].detach().index_select(0, cache_slot_ids_device).clone(),
                    requires_grad=True,
                ),
                "exp_avg": self.fixed_optimizer_state[name]["exp_avg"].detach().index_select(0, cache_slot_ids_device).clone(),
                "exp_avg_sq": self.fixed_optimizer_state[name]["exp_avg_sq"].detach().index_select(0, cache_slot_ids_device).clone(),
            }
        self._grad_accum_non_live_chunks.append(chunk)

    def _writeback_fixed_rows_(self, global_ids_cpu: torch.Tensor, slot_ids_cpu: torch.Tensor):
        self._flush_pending_cpu_writeback(global_ids_cpu)
        global_ids_cpu = global_ids_cpu.detach().to(device="cpu", dtype=torch.long)
        slot_ids_cpu = slot_ids_cpu.detach().to(device="cpu", dtype=torch.long)
        if global_ids_cpu.numel() == 0:
            return
        slot_ids_device = slot_ids_cpu.to(self.device)
        cpu_rows = {}
        cpu_exp_avg = {}
        cpu_exp_avg_sq = {}
        for name, spec in self.table_specs.items():
            active_param = self.fixed_params[name]
            active_state = self.fixed_optimizer_state[name]
            rows = active_param.detach().index_select(0, slot_ids_device)
            exp_avg = active_state["exp_avg"].detach().index_select(0, slot_ids_device)
            exp_avg_sq = active_state["exp_avg_sq"].detach().index_select(0, slot_ids_device)
            row_buffer = self._get_cpu_receive_buffer(f"rows:{name}", tuple(rows.shape), rows.dtype)
            exp_avg_buffer = self._get_cpu_receive_buffer(f"exp_avg:{name}", tuple(exp_avg.shape), exp_avg.dtype)
            exp_avg_sq_buffer = self._get_cpu_receive_buffer(f"exp_avg_sq:{name}", tuple(exp_avg_sq.shape), exp_avg_sq.dtype)
            row_buffer.copy_(rows, non_blocking=self.use_cuda)
            exp_avg_buffer.copy_(exp_avg, non_blocking=self.use_cuda)
            exp_avg_sq_buffer.copy_(exp_avg_sq, non_blocking=self.use_cuda)
            cpu_rows[name] = row_buffer
            cpu_exp_avg[name] = exp_avg_buffer
            cpu_exp_avg_sq[name] = exp_avg_sq_buffer
        if self.use_cuda:
            torch.cuda.synchronize(self.device)
        for name, spec in self.table_specs.items():
            param = spec["param"]
            state = self.state[param]
            param.index_copy_(0, global_ids_cpu, cpu_rows[name])
            state["exp_avg"].index_copy_(0, global_ids_cpu, cpu_exp_avg[name])
            state["exp_avg_sq"].index_copy_(0, global_ids_cpu, cpu_exp_avg_sq[name])

    @torch.no_grad()
    def flush_active_to_cpu(self):
        self._flush_pending_cpu_writeback()
        if self._grad_accum_live:
            raise RuntimeError("Cannot flush active sparse rows while sparse accumulated gradients are pending")
        if not self.fixed_u_mode or not self._fixed_live_state:
            return
        assert self.fixed_slot_to_global_cpu is not None
        active_slot_ids_cpu = torch.nonzero(self.fixed_slot_to_global_cpu >= 0, as_tuple=False).flatten()
        if active_slot_ids_cpu.numel() == 0:
            return
        active_ids_cpu = self.fixed_slot_to_global_cpu[active_slot_ids_cpu]
        self._writeback_fixed_rows_(active_ids_cpu, active_slot_ids_cpu)

    def _stage_rows_to_gpu(self, cpu_tensor_map):
        gpu_tensor_map = {}
        for name, tensor in cpu_tensor_map.items():
            staged = tensor.pin_memory() if self.use_cuda else tensor
            gpu_tensor_map[name] = staged.to(self.device, non_blocking=self.use_cuda)
        return gpu_tensor_map

    def _get_cpu_receive_buffer(self, name: str, shape: tuple[int, ...], dtype: torch.dtype):
        buffer = self._cpu_receive_buffers.get(name)
        is_inference_buffer = bool(buffer is not None and getattr(buffer, "is_inference", lambda: False)())
        needs_new = (
            buffer is None or
            is_inference_buffer or
            buffer.dtype != dtype or
            buffer.dim() != len(shape) or
            any(buffer.size(dim) < shape_dim for dim, shape_dim in enumerate(shape))
        )
        if needs_new:
            with torch.inference_mode(False):
                buffer = torch.empty(shape, dtype=dtype, pin_memory=self.use_cuda)
            self._cpu_receive_buffers[name] = buffer
        assert buffer is not None
        slices = tuple(slice(0, dim) for dim in shape)
        return buffer[slices]

    def _prepare_dynamic_step(
        self,
        active_ids_cpu: torch.Tensor,
        cold_bias_scale: float = 0.0,
        cold_bias_tokens_per_step: Optional[int] = None,
    ) -> DynamicVocabStep:
        self._flush_pending_cpu_writeback()
        active_ids_cpu = active_ids_cpu.detach().to(device="cpu", dtype=torch.long)
        cold_steps_cpu, hot_activation_counts_cpu = self._capture_cold_steps_cpu(active_ids_cpu)
        cold_logit_bias_cpu, cold_bias_clamped_count, cold_bias_abs_max = self._compute_cold_logit_bias_with_stats_cpu(
            cold_steps_cpu,
            cold_bias_scale=cold_bias_scale,
            cold_bias_tokens_per_step=cold_bias_tokens_per_step,
        )
        cpu_rows = {}
        cpu_exp_avg = {}
        cpu_exp_avg_sq = {}
        for name, spec in self.table_specs.items():
            param = spec["param"]
            state = self.state[param]
            rows = param.index_select(0, active_ids_cpu)
            exp_avg = state["exp_avg"].index_select(0, active_ids_cpu)
            exp_avg_sq = state["exp_avg_sq"].index_select(0, active_ids_cpu)
            cpu_rows[name] = rows
            cpu_exp_avg[name] = exp_avg
            cpu_exp_avg_sq[name] = exp_avg_sq

        gpu_rows = self._stage_rows_to_gpu(cpu_rows)
        gpu_exp_avg = self._stage_rows_to_gpu(cpu_exp_avg)
        gpu_exp_avg_sq = self._stage_rows_to_gpu(cpu_exp_avg_sq)

        active_vocab = {
            "wte": nn.Parameter(gpu_rows["wte"], requires_grad=True),
            "lm_head": nn.Parameter(gpu_rows["lm_head"], requires_grad=True),
            "value_embeds": {
                name.split(".", 1)[1]: nn.Parameter(gpu_rows[name], requires_grad=True)
                for name in gpu_rows
                if name.startswith("value_embeds.")
            },
            "cold_logit_bias": cold_logit_bias_cpu.to(self.device, non_blocking=self.use_cuda),
        }
        optimizer_state = {
            name: {
                "exp_avg": gpu_exp_avg[name],
                "exp_avg_sq": gpu_exp_avg_sq[name],
            }
            for name in gpu_rows
        }
        return DynamicVocabStep(
            active_ids_cpu=active_ids_cpu,
            active_vocab=active_vocab,
            optimizer_state=optimizer_state,
            unique_count=active_ids_cpu.numel(),
            live_count=active_ids_cpu.numel(),
            u_capacity=active_ids_cpu.numel(),
            stage_count=active_ids_cpu.numel(),
            cold_bias_clamped_count=cold_bias_clamped_count,
            cold_bias_abs_max=cold_bias_abs_max,
            hot_activation_counts_cpu=hot_activation_counts_cpu,
        )

    def _prepare_fixed_step(
        self,
        step_meta: dict,
        cold_bias_scale: float = 0.0,
        cold_bias_tokens_per_step: Optional[int] = None,
    ) -> DynamicVocabStep:
        assert self.fixed_u_mode, "Fixed-U step requested without fixed_u_max runtime configuration"
        active_ids_cpu = step_meta["active_ids_cpu"].detach().to(device="cpu", dtype=torch.long)
        grad_accum_ids_cpu = step_meta.get("grad_accum_ids_cpu", active_ids_cpu).detach().to(device="cpu", dtype=torch.long)
        grad_accum_steps = int(step_meta.get("grad_accum_steps", 1))
        grad_accum_micro_step = int(step_meta.get("grad_accum_micro_step", 0))
        is_grad_accum_boundary = bool(step_meta.get("is_grad_accum_boundary", True))
        active_slot_ids_cpu = step_meta["active_slot_ids_cpu"].detach().to(device="cpu", dtype=torch.long)
        cold_steps_cpu, hot_activation_counts_cpu = self._capture_cold_steps_cpu(active_ids_cpu)
        cold_logit_bias_cpu, cold_bias_clamped_count, cold_bias_abs_max = self._compute_cold_logit_bias_with_stats_cpu(
            cold_steps_cpu,
            cold_bias_scale=cold_bias_scale,
            cold_bias_tokens_per_step=cold_bias_tokens_per_step,
        )
        active_mask_cpu = step_meta["active_mask_cpu"].detach().to(device="cpu", dtype=torch.bool)
        slot_to_global_cpu = step_meta["slot_to_global_cpu"].detach().to(device="cpu", dtype=torch.long)
        stage_ids_cpu = step_meta["stage_ids_cpu"].detach().to(device="cpu", dtype=torch.long)
        stage_slot_ids_cpu = step_meta["stage_slot_ids_cpu"].detach().to(device="cpu", dtype=torch.long)
        writeback_ids_cpu = step_meta["writeback_ids_cpu"].detach().to(device="cpu", dtype=torch.long)
        writeback_slot_ids_cpu = step_meta["writeback_slot_ids_cpu"].detach().to(device="cpu", dtype=torch.long)
        is_last_step = bool(step_meta["is_last_step"])

        assert self.fixed_slot_to_global_cpu is not None
        assert self.fixed_active_mask_cpu is not None
        assert self.fixed_logit_mask is not None
        assert self.fixed_active_vocab is not None
        assert self.fixed_cold_logit_bias is not None

        if not self._fixed_live_state:
            stage_ids_cpu = active_ids_cpu
            stage_slot_ids_cpu = active_slot_ids_cpu

        self._flush_pending_cpu_writeback(stage_ids_cpu)

        preserve_resident_grads = (
            grad_accum_steps > 1 and
            self._grad_accum_live and
            self._grad_accum_ids_cpu is not None and
            torch.equal(self._grad_accum_ids_cpu, grad_accum_ids_cpu)
        )
        if preserve_resident_grads:
            self._zero_fixed_grad_slots_(stage_slot_ids_cpu)
        else:
            self._clear_fixed_grads()
        if stage_ids_cpu.numel() > 0:
            stage_slot_ids_device = stage_slot_ids_cpu.to(self.device)
            for name, spec in self.table_specs.items():
                param = spec["param"]
                state = self.state[param]
                rows = param.index_select(0, stage_ids_cpu)
                exp_avg = state["exp_avg"].index_select(0, stage_ids_cpu)
                exp_avg_sq = state["exp_avg_sq"].index_select(0, stage_ids_cpu)
                rows_gpu = rows.pin_memory().to(self.device, non_blocking=self.use_cuda) if self.use_cuda else rows.to(self.device)
                exp_avg_gpu = exp_avg.pin_memory().to(self.device, non_blocking=self.use_cuda) if self.use_cuda else exp_avg.to(self.device)
                exp_avg_sq_gpu = exp_avg_sq.pin_memory().to(self.device, non_blocking=self.use_cuda) if self.use_cuda else exp_avg_sq.to(self.device)
                self.fixed_params[name].data.index_copy_(0, stage_slot_ids_device, rows_gpu)
                self.fixed_optimizer_state[name]["exp_avg"].index_copy_(0, stage_slot_ids_device, exp_avg_gpu)
                self.fixed_optimizer_state[name]["exp_avg_sq"].index_copy_(0, stage_slot_ids_device, exp_avg_sq_gpu)

        self.fixed_slot_to_global_cpu.copy_(slot_to_global_cpu)
        self.fixed_active_mask_cpu.copy_(active_mask_cpu)
        self.fixed_logit_mask.copy_(active_mask_cpu.to(self.device, non_blocking=self.use_cuda))
        self.fixed_cold_logit_bias.zero_()
        if active_slot_ids_cpu.numel() > 0:
            active_slot_ids_device = active_slot_ids_cpu.to(self.device)
            self.fixed_cold_logit_bias.index_copy_(0, active_slot_ids_device, cold_logit_bias_cpu.to(self.device, non_blocking=self.use_cuda))
        self._fixed_live_state = True

        step_active_vocab = {
            **self.fixed_active_vocab,
            "value_embeds": self.fixed_active_vocab["value_embeds"],
        }
        step_optimizer_state = {
            **self.fixed_optimizer_state,
        }

        return DynamicVocabStep(
            active_ids_cpu=active_ids_cpu,
            active_vocab=step_active_vocab,
            optimizer_state=step_optimizer_state,
            unique_count=active_ids_cpu.numel(),
            live_count=active_ids_cpu.numel(),
            u_capacity=self.fixed_u_max,
            stage_count=stage_ids_cpu.numel(),
            active_slot_ids_cpu=active_slot_ids_cpu,
            active_mask_cpu=active_mask_cpu,
            slot_to_global_cpu=slot_to_global_cpu,
            stage_ids_cpu=stage_ids_cpu,
            stage_slot_ids_cpu=stage_slot_ids_cpu,
            writeback_ids_cpu=writeback_ids_cpu,
            writeback_slot_ids_cpu=writeback_slot_ids_cpu,
            grad_accum_ids_cpu=grad_accum_ids_cpu,
            grad_accum_steps=grad_accum_steps,
            grad_accum_micro_step=grad_accum_micro_step,
            is_grad_accum_boundary=is_grad_accum_boundary,
            is_last_step=is_last_step,
            fixed_u_mode=True,
            cold_bias_clamped_count=cold_bias_clamped_count,
            cold_bias_abs_max=cold_bias_abs_max,
            hot_activation_counts_cpu=hot_activation_counts_cpu,
        )

    def prepare_step(
        self,
        active_ids_cpu,
        cold_bias_scale: float = 0.0,
        cold_bias_tokens_per_step: Optional[int] = None,
    ) -> DynamicVocabStep:
        if isinstance(active_ids_cpu, dict):
            return self._prepare_fixed_step(
                active_ids_cpu,
                cold_bias_scale=cold_bias_scale,
                cold_bias_tokens_per_step=cold_bias_tokens_per_step,
            )
        return self._prepare_dynamic_step(
            active_ids_cpu,
            cold_bias_scale=cold_bias_scale,
            cold_bias_tokens_per_step=cold_bias_tokens_per_step,
        )

    def _adamw_update_(self, param_name: str, active_param: nn.Parameter, active_state: dict, slot_ids_cpu: Optional[torch.Tensor] = None, hot_activation_counts_cpu: Optional[torch.Tensor] = None) -> None:
        grad = active_param.grad
        if grad is None:
            return
        spec = self.table_specs[param_name]
        state = self.state[spec["param"]]
        state["step"] += 1
        self._adamw_update_with_step_(
            param_name,
            active_param,
            active_state,
            state["step"],
            slot_ids_cpu=slot_ids_cpu,
            hot_activation_counts_cpu=hot_activation_counts_cpu,
        )

    def _adamw_update_with_step_(self, param_name: str, active_param: nn.Parameter, active_state: dict, step_value: int, slot_ids_cpu: Optional[torch.Tensor] = None, hot_activation_counts_cpu: Optional[torch.Tensor] = None) -> None:
        grad = active_param.grad
        if grad is None:
            return
        spec = self.table_specs[param_name]
        exp_avg = active_state["exp_avg"]
        exp_avg_sq = active_state["exp_avg_sq"]
        row_lr = None
        if param_name == "lm_head":
            row_lr = self._get_lm_head_row_lr(float(spec["lr"]), hot_activation_counts_cpu, active_param.device, active_param.dtype)
        if slot_ids_cpu is None:
            if self.weight_decay != 0.0:
                active_param.mul_(1 - spec["lr"] * self.weight_decay)
            exp_avg.lerp_(grad, 1 - self.beta1)
            exp_avg_sq.lerp_(grad.square(), 1 - self.beta2)
            bias1 = 1 - self.beta1 ** step_value
            bias2 = 1 - self.beta2 ** step_value
            denom = (exp_avg_sq / bias2).sqrt().add_(self.eps)
            if row_lr is not None:
                row_lr = row_lr.view((-1,) + (1,) * (active_param.dim() - 1))
                active_param.add_((exp_avg / denom) * row_lr, alpha=-1.0 / bias1)
            else:
                step_size = spec["lr"] / bias1
                active_param.addcdiv_(exp_avg, denom, value=-step_size)
            return

        slot_ids_cpu = slot_ids_cpu.detach().to(device="cpu", dtype=torch.long)
        if slot_ids_cpu.numel() == 0:
            return
        slot_ids = slot_ids_cpu.to(active_param.device)
        param_rows = active_param.index_select(0, slot_ids)
        grad_rows = grad.index_select(0, slot_ids)
        exp_avg_rows = exp_avg.index_select(0, slot_ids)
        exp_avg_sq_rows = exp_avg_sq.index_select(0, slot_ids)
        if self.weight_decay != 0.0:
            param_rows.mul_(1 - spec["lr"] * self.weight_decay)
        exp_avg_rows.lerp_(grad_rows, 1 - self.beta1)
        exp_avg_sq_rows.lerp_(grad_rows.square(), 1 - self.beta2)
        bias1 = 1 - self.beta1 ** step_value
        bias2 = 1 - self.beta2 ** step_value
        denom = (exp_avg_sq_rows / bias2).sqrt().add_(self.eps)
        if row_lr is not None:
            row_lr = row_lr.view((-1,) + (1,) * (param_rows.dim() - 1))
            param_rows.add_((exp_avg_rows / denom) * row_lr, alpha=-1.0 / bias1)
        else:
            step_size = spec["lr"] / bias1
            param_rows.addcdiv_(exp_avg_rows, denom, value=-step_size)
        active_param.index_copy_(0, slot_ids, param_rows)
        exp_avg.index_copy_(0, slot_ids, exp_avg_rows)
        exp_avg_sq.index_copy_(0, slot_ids, exp_avg_sq_rows)

    @torch.no_grad()
    def accumulate_gradients(self, step_ctx: DynamicVocabStep) -> DynamicVocabStep:
        t_start = time.perf_counter()
        assert step_ctx.fixed_u_mode, "Sparse grad accumulation currently supports fixed-U mode only"
        assert step_ctx.active_slot_ids_cpu is not None
        assert step_ctx.active_ids_cpu is not None
        if step_ctx.grad_accum_ids_cpu is None:
            raise ValueError("Sparse grad accumulation requires grad_accum_ids_cpu metadata")

        if (not self._grad_accum_live) or self._grad_accum_ids_cpu is None or not torch.equal(self._grad_accum_ids_cpu, step_ctx.grad_accum_ids_cpu):
            self._start_grad_accum_window(step_ctx.grad_accum_ids_cpu)

        assert self._grad_accum_global_to_local_cpu is not None
        assert self._grad_accum_buffers is not None
        t_flush_start = time.perf_counter()
        self._flush_pending_grad_accum_transfers(wait=False)
        flush_ms = (time.perf_counter() - t_flush_start) * 1000.0

        t_queue_start = time.perf_counter()
        queued_count = 0
        if not step_ctx.is_grad_accum_boundary:
            leaving_ids_cpu = step_ctx.writeback_ids_cpu
            leaving_slot_ids_cpu = step_ctx.writeback_slot_ids_cpu
            if leaving_ids_cpu is None:
                leaving_ids_cpu = torch.empty(0, dtype=torch.long)
            if leaving_slot_ids_cpu is None:
                leaving_slot_ids_cpu = torch.empty(0, dtype=torch.long)
            leaving_ids_cpu = leaving_ids_cpu.detach().to(device="cpu", dtype=torch.long)
            leaving_slot_ids_cpu = leaving_slot_ids_cpu.detach().to(device="cpu", dtype=torch.long)
            if leaving_ids_cpu.numel() > 0:
                accum_row_ids_cpu = self._grad_accum_global_to_local_cpu[leaving_ids_cpu]
                if (accum_row_ids_cpu < 0).any():
                    raise ValueError("Sparse grad accumulation map is missing leaving vocab rows")
                leaving_slot_ids_device = leaving_slot_ids_cpu.to(self.device)
                accum_row_ids_device = accum_row_ids_cpu.to(self.device)
                self._cache_grad_accum_leaving_rows_(accum_row_ids_cpu, leaving_slot_ids_cpu)
                for name in self.table_specs:
                    if name == "wte":
                        grad = self.fixed_params["wte"].grad
                    elif name == "lm_head":
                        grad = self.fixed_params["lm_head"].grad
                    else:
                        grad = self.fixed_params[name].grad
                    if grad is None:
                        continue
                    grad_rows = grad.detach().index_select(0, leaving_slot_ids_device)
                    self._grad_accum_buffers[name].index_add_(0, accum_row_ids_device, grad_rows.to(dtype=self._grad_accum_buffers[name].dtype))
                self._zero_fixed_grad_slots_(leaving_slot_ids_cpu)
                queued_count = int(leaving_ids_cpu.numel())
        queue_ms = (time.perf_counter() - t_queue_start) * 1000.0

        step_ctx.active_vocab = None
        step_ctx.optimizer_state = None
        self._grad_accum_stage_count += int(step_ctx.stage_count)
        step_ctx.unique_count = int(step_ctx.grad_accum_ids_cpu.numel())
        step_ctx.live_count = int(step_ctx.active_ids_cpu.numel())
        step_ctx.grad_accum_flush_ms = flush_ms
        step_ctx.grad_accum_queue_ms = queue_ms
        step_ctx.grad_accum_queue_count = queued_count
        step_ctx.grad_accum_resident_count = max(int(step_ctx.active_ids_cpu.numel()) - queued_count, 0)
        step_ctx.optimizer_ms = (time.perf_counter() - t_start) * 1000.0
        return step_ctx

    @torch.no_grad()
    def apply_accumulated_gradients(self) -> DynamicVocabStep:
        if not self._grad_accum_live or self._grad_accum_ids_cpu is None or self._grad_accum_buffers is None:
            raise ValueError("No sparse accumulated gradients are pending")

        t_flush_start = time.perf_counter()
        self._flush_pending_grad_accum_transfers(wait=True)
        flush_ms = (time.perf_counter() - t_flush_start) * 1000.0

        grad_accum_ids_cpu = self._grad_accum_ids_cpu
        grad_accum_count = self._grad_accum_count
        live_count = int(grad_accum_ids_cpu.numel())
        live_slot_ids_cpu = torch.empty(0, dtype=torch.long)
        live_union_row_ids_cpu = torch.empty(0, dtype=torch.long)
        live_union_mask_cpu = torch.zeros(grad_accum_count, dtype=torch.bool)
        if self.fixed_u_mode and self._fixed_live_state:
            assert self.fixed_slot_to_global_cpu is not None
            live_slot_ids_cpu = torch.nonzero(self.fixed_slot_to_global_cpu >= 0, as_tuple=False).flatten()
            live_count = int(live_slot_ids_cpu.numel())
            if live_slot_ids_cpu.numel() > 0:
                assert self._grad_accum_global_to_local_cpu is not None
                live_global_ids_cpu = self.fixed_slot_to_global_cpu[live_slot_ids_cpu]
                live_union_row_ids_cpu = self._grad_accum_global_to_local_cpu[live_global_ids_cpu]
                if (live_union_row_ids_cpu < 0).any():
                    raise ValueError("Sparse grad accumulation map is missing live fixed-U rows")
                live_union_mask_cpu[live_union_row_ids_cpu] = True

        non_live_union_row_ids_cpu = torch.nonzero(~live_union_mask_cpu, as_tuple=False).flatten()
        non_live_grad_accum_ids_cpu = grad_accum_ids_cpu.index_select(0, non_live_union_row_ids_cpu) if non_live_union_row_ids_cpu.numel() > 0 else torch.empty(0, dtype=torch.long)
        cached_non_live_mask_cpu = torch.zeros(non_live_union_row_ids_cpu.numel(), dtype=torch.bool)
        if self._grad_accum_cached_union_mask_cpu is not None and non_live_union_row_ids_cpu.numel() > 0:
            cached_non_live_mask_cpu = self._grad_accum_cached_union_mask_cpu[non_live_union_row_ids_cpu]
        staged_non_live_union_row_ids_cpu = non_live_union_row_ids_cpu[~cached_non_live_mask_cpu]
        staged_non_live_grad_accum_ids_cpu = grad_accum_ids_cpu.index_select(0, staged_non_live_union_row_ids_cpu) if staged_non_live_union_row_ids_cpu.numel() > 0 else torch.empty(0, dtype=torch.long)

        t_stage_start = time.perf_counter()
        non_live_params = {}
        non_live_optimizer_state = {}
        live_slot_ids_device = live_slot_ids_cpu.to(self.device) if live_slot_ids_cpu.numel() > 0 else None
        staged_non_live_union_row_ids_device = staged_non_live_union_row_ids_cpu.to(self.device) if staged_non_live_union_row_ids_cpu.numel() > 0 else None
        for name, spec in self.table_specs.items():
            param = spec["param"]
            state = self.state[param]
            if staged_non_live_grad_accum_ids_cpu.numel() > 0 and staged_non_live_union_row_ids_device is not None:
                rows = param.index_select(0, staged_non_live_grad_accum_ids_cpu)
                exp_avg = state["exp_avg"].index_select(0, staged_non_live_grad_accum_ids_cpu)
                exp_avg_sq = state["exp_avg_sq"].index_select(0, staged_non_live_grad_accum_ids_cpu)
                rows_gpu = rows.pin_memory().to(self.device, non_blocking=self.use_cuda) if self.use_cuda else rows.to(self.device)
                exp_avg_gpu = exp_avg.pin_memory().to(self.device, non_blocking=self.use_cuda) if self.use_cuda else exp_avg.to(self.device)
                exp_avg_sq_gpu = exp_avg_sq.pin_memory().to(self.device, non_blocking=self.use_cuda) if self.use_cuda else exp_avg_sq.to(self.device)
                non_live_params[name] = nn.Parameter(rows_gpu, requires_grad=True)
                non_live_optimizer_state[name] = {
                    "exp_avg": exp_avg_gpu,
                    "exp_avg_sq": exp_avg_sq_gpu,
                }
        stage_ms = (time.perf_counter() - t_stage_start) * 1000.0

        t_apply_start = time.perf_counter()
        for name in self.table_specs:
            spec = self.table_specs[name]
            state = self.state[spec["param"]]
            has_live_grad = live_slot_ids_cpu.numel() > 0 and self.fixed_params[name].grad is not None
            has_cached_non_live_grad = bool(self._grad_accum_non_live_chunks)
            has_staged_non_live_grad = name in non_live_params and staged_non_live_union_row_ids_device is not None
            if not has_live_grad and not has_cached_non_live_grad and not has_staged_non_live_grad:
                continue
            state["step"] += 1
            step_value = state["step"]
            if has_live_grad:
                self._adamw_update_with_step_(
                    name,
                    self.fixed_params[name],
                    self.fixed_optimizer_state[name],
                    step_value,
                    slot_ids_cpu=live_slot_ids_cpu,
                )
            if has_cached_non_live_grad:
                for chunk in self._grad_accum_non_live_chunks:
                    chunk_param = chunk["tables"][name]["param"]
                    chunk_param.grad = self._grad_accum_buffers[name].index_select(0, chunk["union_row_ids_device"])
                    self._adamw_update_with_step_(
                        name,
                        chunk_param,
                        {
                            "exp_avg": chunk["tables"][name]["exp_avg"],
                            "exp_avg_sq": chunk["tables"][name]["exp_avg_sq"],
                        },
                        step_value,
                    )
            if has_staged_non_live_grad:
                non_live_params[name].grad = self._grad_accum_buffers[name].index_select(0, staged_non_live_union_row_ids_device)
                self._adamw_update_with_step_(
                    name,
                    non_live_params[name],
                    non_live_optimizer_state[name],
                    step_value,
                )
        apply_ms = (time.perf_counter() - t_apply_start) * 1000.0

        restore_ms = 0.0

        t_writeback_start = time.perf_counter()
        d2h_launch_ms = 0.0
        d2h_sync_ms = 0.0
        cpu_writeback_ms = 0.0
        writeback_ids_cpu = non_live_grad_accum_ids_cpu
        if writeback_ids_cpu.numel() > 0:
            self._flush_pending_cpu_writeback()
            t_d2h_launch_start = time.perf_counter()
            writeback_segments = []
            for name, spec in self.table_specs.items():
                param = spec["param"]
                state = self.state[param]
                for chunk_idx, chunk in enumerate(self._grad_accum_non_live_chunks):
                    row_source = chunk["tables"][name]["param"].detach()
                    exp_avg_source = chunk["tables"][name]["exp_avg"].detach()
                    exp_avg_sq_source = chunk["tables"][name]["exp_avg_sq"].detach()
                    row_buffer = self._get_cpu_receive_buffer(f"rows:accum:{name}:chunk{chunk_idx}", tuple(row_source.shape), row_source.dtype)
                    exp_avg_buffer = self._get_cpu_receive_buffer(f"exp_avg:accum:{name}:chunk{chunk_idx}", tuple(exp_avg_source.shape), exp_avg_source.dtype)
                    exp_avg_sq_buffer = self._get_cpu_receive_buffer(f"exp_avg_sq:accum:{name}:chunk{chunk_idx}", tuple(exp_avg_sq_source.shape), exp_avg_sq_source.dtype)
                    row_buffer.copy_(row_source, non_blocking=self.use_cuda)
                    exp_avg_buffer.copy_(exp_avg_source, non_blocking=self.use_cuda)
                    exp_avg_sq_buffer.copy_(exp_avg_sq_source, non_blocking=self.use_cuda)
                    writeback_segments.append((name, param, state, chunk["global_ids_cpu"], row_buffer, exp_avg_buffer, exp_avg_sq_buffer))
                if name in non_live_params:
                    row_source = non_live_params[name].detach()
                    exp_avg_source = non_live_optimizer_state[name]["exp_avg"].detach()
                    exp_avg_sq_source = non_live_optimizer_state[name]["exp_avg_sq"].detach()
                    row_buffer = self._get_cpu_receive_buffer(f"rows:accum:{name}:fallback", tuple(row_source.shape), row_source.dtype)
                    exp_avg_buffer = self._get_cpu_receive_buffer(f"exp_avg:accum:{name}:fallback", tuple(exp_avg_source.shape), exp_avg_source.dtype)
                    exp_avg_sq_buffer = self._get_cpu_receive_buffer(f"exp_avg_sq:accum:{name}:fallback", tuple(exp_avg_sq_source.shape), exp_avg_sq_source.dtype)
                    row_buffer.copy_(row_source, non_blocking=self.use_cuda)
                    exp_avg_buffer.copy_(exp_avg_source, non_blocking=self.use_cuda)
                    exp_avg_sq_buffer.copy_(exp_avg_sq_source, non_blocking=self.use_cuda)
                    writeback_segments.append((name, param, state, staged_non_live_grad_accum_ids_cpu, row_buffer, exp_avg_buffer, exp_avg_sq_buffer))
            d2h_launch_ms = (time.perf_counter() - t_d2h_launch_start) * 1000.0
            ready_event = None
            if self.use_cuda:
                ready_event = torch.cuda.Event()
                ready_event.record(torch.cuda.current_stream(self.device))

            pending_mask_cpu = torch.zeros(self.model.config.vocab_size, dtype=torch.bool)
            pending_mask_cpu[writeback_ids_cpu] = True

            def _write_segments_when_ready(event, segments) -> None:
                if event is not None:
                    event.synchronize()
                with torch.no_grad():
                    for _, param, state, segment_ids_cpu, row_buffer, exp_avg_buffer, exp_avg_sq_buffer in segments:
                        param.index_copy_(0, segment_ids_cpu, row_buffer)
                        state["exp_avg"].index_copy_(0, segment_ids_cpu, exp_avg_buffer)
                        state["exp_avg_sq"].index_copy_(0, segment_ids_cpu, exp_avg_sq_buffer)

            self._pending_cpu_writeback_future = self._cpu_writeback_executor.submit(_write_segments_when_ready, ready_event, writeback_segments)
            self._pending_cpu_writeback_mask_cpu = pending_mask_cpu
        writeback_ms = (time.perf_counter() - t_writeback_start) * 1000.0

        metrics = DynamicVocabStep(
            active_ids_cpu=grad_accum_ids_cpu,
            active_vocab=None,
            optimizer_state=None,
            unique_count=int(grad_accum_ids_cpu.numel()),
            live_count=live_count,
            u_capacity=self.grad_accum_u_max if self.grad_accum_u_max > 0 else int(grad_accum_ids_cpu.numel()),
            stage_count=int(self._grad_accum_stage_count),
            writeback_count=int(writeback_ids_cpu.numel()),
            grad_accum_flush_ms=flush_ms,
            grad_accum_queue_count=int(writeback_ids_cpu.numel()),
            grad_accum_stage_ms=stage_ms,
            grad_accum_apply_ms=apply_ms,
            grad_accum_restore_ms=restore_ms,
            grad_accum_writeback_ms=writeback_ms,
            grad_accum_resident_count=live_count,
            d2h_launch_ms=d2h_launch_ms,
            d2h_sync_ms=d2h_sync_ms,
            cpu_writeback_ms=cpu_writeback_ms,
            optimizer_ms=apply_ms,
            fixed_u_mode=self.fixed_u_mode,
        )
        self._clear_fixed_grads()
        self._clear_grad_accum_window()
        self.runtime_step += 1
        return metrics

    @torch.no_grad()
    def step(self, step_ctx: DynamicVocabStep) -> DynamicVocabStep:
        assert step_ctx.active_vocab is not None
        assert step_ctx.optimizer_state is not None
        if step_ctx.fixed_u_mode:
            assert step_ctx.active_slot_ids_cpu is not None
            self._adamw_update_("wte", self.fixed_params["wte"], self.fixed_optimizer_state["wte"], slot_ids_cpu=step_ctx.active_slot_ids_cpu)
            self._adamw_update_(
                "lm_head",
                self.fixed_params["lm_head"],
                self.fixed_optimizer_state["lm_head"],
                slot_ids_cpu=step_ctx.active_slot_ids_cpu,
                hot_activation_counts_cpu=step_ctx.hot_activation_counts_cpu,
            )
            for layer_name in step_ctx.active_vocab["value_embeds"]:
                param_name = f"value_embeds.{layer_name}"
                self._adamw_update_(
                    param_name,
                    self.fixed_params[param_name],
                    self.fixed_optimizer_state[param_name],
                    slot_ids_cpu=step_ctx.active_slot_ids_cpu,
                )

            writeback_ids_cpu = step_ctx.active_ids_cpu if step_ctx.is_last_step else step_ctx.writeback_ids_cpu
            writeback_slot_ids_cpu = step_ctx.active_slot_ids_cpu if step_ctx.is_last_step else step_ctx.writeback_slot_ids_cpu
            if writeback_ids_cpu is None:
                writeback_ids_cpu = torch.empty(0, dtype=torch.long)
            if writeback_slot_ids_cpu is None:
                writeback_slot_ids_cpu = torch.empty(0, dtype=torch.long)
            writeback_ids_cpu = writeback_ids_cpu.detach().to(device="cpu", dtype=torch.long)
            writeback_slot_ids_cpu = writeback_slot_ids_cpu.detach().to(device="cpu", dtype=torch.long)
            step_ctx.writeback_count = writeback_ids_cpu.numel()

            self._writeback_fixed_rows_(writeback_ids_cpu, writeback_slot_ids_cpu)
            self._clear_fixed_grads()
            step_ctx.active_vocab = None
            step_ctx.optimizer_state = None
            self.runtime_step += 1
            return step_ctx

        self._adamw_update_("wte", step_ctx.active_vocab["wte"], step_ctx.optimizer_state["wte"])
        self._adamw_update_(
            "lm_head",
            step_ctx.active_vocab["lm_head"],
            step_ctx.optimizer_state["lm_head"],
            hot_activation_counts_cpu=step_ctx.hot_activation_counts_cpu,
        )
        for layer_name, active_param in step_ctx.active_vocab["value_embeds"].items():
            param_name = f"value_embeds.{layer_name}"
            self._adamw_update_(param_name, active_param, step_ctx.optimizer_state[param_name])
        step_ctx.writeback_count = step_ctx.active_ids_cpu.numel()

        cpu_rows = {}
        cpu_exp_avg = {}
        cpu_exp_avg_sq = {}
        for name, spec in self.table_specs.items():
            if name == "wte":
                active_param = step_ctx.active_vocab["wte"]
            elif name == "lm_head":
                active_param = step_ctx.active_vocab["lm_head"]
            else:
                layer_name = name.split(".", 1)[1]
                active_param = step_ctx.active_vocab["value_embeds"][layer_name]
            active_state = step_ctx.optimizer_state[name]
            row_buffer = self._get_cpu_receive_buffer(f"rows:{name}", tuple(active_param.shape), active_param.dtype)
            exp_avg_buffer = self._get_cpu_receive_buffer(f"exp_avg:{name}", tuple(active_state["exp_avg"].shape), active_state["exp_avg"].dtype)
            exp_avg_sq_buffer = self._get_cpu_receive_buffer(f"exp_avg_sq:{name}", tuple(active_state["exp_avg_sq"].shape), active_state["exp_avg_sq"].dtype)
            row_buffer.copy_(active_param.detach(), non_blocking=self.use_cuda)
            exp_avg_buffer.copy_(active_state["exp_avg"].detach(), non_blocking=self.use_cuda)
            exp_avg_sq_buffer.copy_(active_state["exp_avg_sq"].detach(), non_blocking=self.use_cuda)
            rows = row_buffer
            exp_avg = exp_avg_buffer
            exp_avg_sq = exp_avg_sq_buffer
            cpu_rows[name] = rows
            cpu_exp_avg[name] = exp_avg
            cpu_exp_avg_sq[name] = exp_avg_sq
        if self.use_cuda:
            torch.cuda.synchronize(self.device)
        for name, spec in self.table_specs.items():
            param = spec["param"]
            state = self.state[param]
            param.index_copy_(0, step_ctx.active_ids_cpu, cpu_rows[name])
            state["exp_avg"].index_copy_(0, step_ctx.active_ids_cpu, cpu_exp_avg[name])
            state["exp_avg_sq"].index_copy_(0, step_ctx.active_ids_cpu, cpu_exp_avg_sq[name])
        step_ctx.active_vocab = None
        step_ctx.optimizer_state = None
        self.runtime_step += 1
        return step_ctx
