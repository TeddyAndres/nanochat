import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


COLD_LOGIT_BIAS_CLAMP_MIN = -3.0
COLD_LOGIT_BIAS_CLAMP_MAX = 3.0


def round_capacity_up(value: int | None, multiple: int) -> int | None:
    if value is None:
        return None
    value = int(value)
    multiple = int(multiple)
    if multiple <= 1 or value <= 0:
        return value
    return ((value + multiple - 1) // multiple) * multiple


@dataclass
class DynamicVocabStep:
    active_ids_cpu: torch.Tensor
    active_vocab: Optional[dict]
    optimizer_state: Optional[dict]
    unique_count: int
    union_inputs: Optional[torch.Tensor] = None
    union_targets: Optional[torch.Tensor] = None
    live_count: int = 0
    step_u_count: int = 0
    bytes_h2d: int = 0
    h2d_ms: float = 0.0
    bytes_d2h: int = 0
    d2h_ms: float = 0.0
    optimizer_ms: float = 0.0
    grad_accum_start_ms: float = 0.0
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
    prep_writeback_wait_ms: float = 0.0
    prep_prefetch_wait_ms: float = 0.0
    prep_cpu_gather_ms: float = 0.0
    prep_h2d_enqueue_ms: float = 0.0
    prep_h2d_tensor_count: int = 0
    prep_h2d_bytes: int = 0
    prep_prefetch_hit: int = 0
    d2h_segment_count: int = 0
    d2h_row_count: int = 0
    d2h_bytes: int = 0
    active_param_bytes: int = 0
    active_grad_bytes: int = 0
    active_optimizer_bytes: int = 0
    u_capacity: int = 0
    stage_count: int = 0
    writeback_count: int = 0
    cloud_residual_capacity: int = 0
    warm_budget_target: int = 0
    cold_budget_target: int = 0
    warm_candidate_count: int = 0
    cloud_plan_ms: float = 0.0
    cloud_hidden_query_ms: float = 0.0
    cloud_selection_ms: float = 0.0
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
    lm_head_active_ids_cpu: Optional[torch.Tensor] = None
    lm_head_active_slot_ids_cpu: Optional[torch.Tensor] = None
    warm_ids_cpu: Optional[torch.Tensor] = None
    warm_slot_ids_cpu: Optional[torch.Tensor] = None
    cold_ids_cpu: Optional[torch.Tensor] = None
    cold_slot_ids_cpu: Optional[torch.Tensor] = None
    cloud_stage_ids_cpu: Optional[torch.Tensor] = None
    cloud_stage_slot_ids_cpu: Optional[torch.Tensor] = None
    cloud_writeback_ids_cpu: Optional[torch.Tensor] = None
    cloud_writeback_slot_ids_cpu: Optional[torch.Tensor] = None
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
        lm_head_u_max=None,
        grad_accum_u_max=None,
        capacity_round_multiple=1,
        cold_bias_reference_tokens=2**19,
        value_embedding_lr=None,
        unembedding_warm_lr=None,
        unembedding_cold_lr=None,
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
        self.capacity_round_multiple = max(1, int(capacity_round_multiple))
        self.disable_fixed_overlap_reuse = os.getenv("NANOCHAT_DISABLE_FIXED_OVERLAP_REUSE", "0") == "1"
        fixed_u_max_value = 0 if fixed_u_max is None else round_capacity_up(int(fixed_u_max), self.capacity_round_multiple)
        assert fixed_u_max_value is not None
        self.fixed_u_max = int(fixed_u_max_value)
        self.fixed_u_mode = self.fixed_u_max > 0
        requested_lm_head_u_max = self.fixed_u_max if lm_head_u_max is None else round_capacity_up(int(lm_head_u_max), self.capacity_round_multiple)
        assert requested_lm_head_u_max is not None
        self.lm_head_u_max = max(self.fixed_u_max, requested_lm_head_u_max)
        if self.lm_head_u_max < 0:
            raise ValueError(f"lm_head_u_max must be non-negative, got {self.lm_head_u_max}")
        default_grad_accum_u_max = self.fixed_u_max if self.fixed_u_mode else 0
        grad_accum_u_max_value = default_grad_accum_u_max if grad_accum_u_max is None else round_capacity_up(int(grad_accum_u_max), self.capacity_round_multiple)
        assert grad_accum_u_max_value is not None
        self.grad_accum_u_max = int(grad_accum_u_max_value)
        if self.grad_accum_u_max < 0:
            raise ValueError(f"grad_accum_u_max must be non-negative, got {self.grad_accum_u_max}")
        if self.fixed_u_mode and self.grad_accum_u_max < self.fixed_u_max:
            raise ValueError(
                f"grad_accum_u_max must be at least fixed_u_max in fixed-U sparse mode, got grad_accum_u_max={self.grad_accum_u_max}, fixed_u_max={self.fixed_u_max}"
            )
        if self.fixed_u_mode and self.lm_head_u_max < self.grad_accum_u_max:
            self.lm_head_u_max = self.grad_accum_u_max
        self.fixed_input_u_max = max(self.fixed_u_max, self.grad_accum_u_max)
        value_embedding_lr = embedding_lr if value_embedding_lr is None else value_embedding_lr

        self.table_specs = {
            "wte": {
                "param": model.transformer.wte.weight,
                "lr": embedding_lr,
            },
            "lm_head": {
                "param": model.lm_head.weight,
                "lr": unembedding_lr,
                "warm_lr": unembedding_lr if unembedding_warm_lr is None else float(unembedding_warm_lr),
                "cold_lr": unembedding_lr if unembedding_cold_lr is None else float(unembedding_cold_lr),
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
        self.token_event_step_count_cpu = torch.zeros((model.config.vocab_size,), dtype=torch.long)
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
        self.fixed_input_slot_to_global_cpu = None
        self.fixed_lm_head_slot_to_global_cpu = None
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
        self._grad_accum_hot_activation_counts_cpu = None
        self._grad_accum_window_cold_steps_cpu = None
        self._grad_accum_window_cold_logit_bias_cpu = None
        self._grad_accum_window_cold_bias_clamped_count = 0
        self._grad_accum_window_cold_bias_abs_max = 0.0
        self._grad_accum_pending_transfers = []
        self._grad_accum_transfer_stream = torch.cuda.Stream(device=self.device) if self.use_cuda else None
        self._cpu_writeback_executor = ThreadPoolExecutor(max_workers=max(2, min(8, len(self.table_specs))))
        self._pending_cpu_writeback_futures = None
        self._pending_cpu_writeback_future = None
        self._pending_cpu_writeback_mask_cpu = None
        self._stage_prefetch_executor = ThreadPoolExecutor(max_workers=1)
        self._pending_stage_prefetch_future = None
        self._pending_stage_prefetch_key = None
        self._gpu_stage_buffers = {}
        self._last_hidden_query_ms = 0.0
        self.global_token_count_cpu = torch.zeros((model.config.vocab_size,), dtype=torch.long)
        if self.fixed_u_mode:
            self.fixed_slot_to_global_cpu = torch.full((self.fixed_u_max,), -1, dtype=torch.long)
            self.fixed_input_slot_to_global_cpu = torch.full((self.fixed_input_u_max,), -1, dtype=torch.long)
            self.fixed_lm_head_slot_to_global_cpu = torch.full((self.lm_head_u_max,), -1, dtype=torch.long)
            self.fixed_active_mask_cpu = torch.zeros(self.fixed_u_max, dtype=torch.bool)
            for name, spec in self.table_specs.items():
                param = spec["param"]
                row_count = self.lm_head_u_max if name == "lm_head" else self.fixed_input_u_max
                shape = (row_count,) + tuple(param.shape[1:])
                self.fixed_params[name] = nn.Parameter(
                    torch.zeros(shape, device=self.device, dtype=param.dtype),
                    requires_grad=True,
                )
                self.fixed_optimizer_state[name] = {
                    "exp_avg": torch.zeros(shape, device=self.device, dtype=param.dtype),
                    "exp_avg_sq": torch.zeros(shape, device=self.device, dtype=param.dtype),
                }
            self.fixed_logit_mask = torch.zeros(self.lm_head_u_max, dtype=torch.bool, device=self.device)
            self.fixed_cold_logit_bias = torch.zeros(self.lm_head_u_max, dtype=torch.float32, device=self.device)
            self.fixed_active_vocab = {
                "wte": self.fixed_params["wte"],
                "lm_head": self.fixed_params["lm_head"],
                "value_embeds": {
                    name.split(".", 1)[1]: self.fixed_params[name]
                    for name in self.fixed_params
                    if name.startswith("value_embeds.")
                },
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
            "version": 4,
            "runtime_step": self.runtime_step,
            "last_seen_step_cpu": self.last_seen_step_cpu,
            "hot_activation_count_cpu": self.hot_activation_count_cpu,
            "token_event_step_count_cpu": self.token_event_step_count_cpu,
            "global_token_count_cpu": self.global_token_count_cpu,
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
        if "token_event_step_count_cpu" in state_dict:
            self.token_event_step_count_cpu.copy_(state_dict["token_event_step_count_cpu"].to(device="cpu", dtype=torch.long))
        else:
            self.token_event_step_count_cpu.zero_()
        if "global_token_count_cpu" in state_dict:
            self.global_token_count_cpu.copy_(state_dict["global_token_count_cpu"].to(device="cpu", dtype=torch.long))
        else:
            self.global_token_count_cpu.zero_()
        tables = state_dict.get("tables", {})
        for name, table_state in tables.items():
            spec = self.table_specs[name]
            param = spec["param"]
            state = self.state[param]
            state["step"] = int(table_state["step"])
            state["exp_avg"].copy_(table_state["exp_avg"].to("cpu"))
            state["exp_avg_sq"].copy_(table_state["exp_avg_sq"].to("cpu"))

    def _peek_cold_steps_and_hot_counts_cpu(self, active_ids_cpu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        active_ids_cpu = active_ids_cpu.detach().to(device="cpu", dtype=torch.long)
        if active_ids_cpu.numel() == 0:
            return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
        last_seen = self.last_seen_step_cpu.index_select(0, active_ids_cpu)
        hot_activation_counts_cpu = self.hot_activation_count_cpu.index_select(0, active_ids_cpu)
        cold_steps_cpu = (self.runtime_step - last_seen - 1).clamp_min_(0)
        cold_steps_cpu.masked_fill_(last_seen < 0, 0)
        return cold_steps_cpu, hot_activation_counts_cpu

    def _capture_cold_steps_cpu(self, active_ids_cpu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        active_ids_cpu = active_ids_cpu.detach().to(device="cpu", dtype=torch.long)
        cold_steps_cpu, hot_activation_counts_cpu = self._peek_cold_steps_and_hot_counts_cpu(active_ids_cpu)
        if active_ids_cpu.numel() == 0:
            return cold_steps_cpu, hot_activation_counts_cpu
        self.last_seen_step_cpu.index_fill_(0, active_ids_cpu, self.runtime_step)
        self.hot_activation_count_cpu.index_copy_(0, active_ids_cpu, hot_activation_counts_cpu + 1)
        return cold_steps_cpu, hot_activation_counts_cpu

    def _get_token_event_step_values_cpu(self, global_ids_cpu: torch.Tensor) -> torch.Tensor:
        global_ids_cpu = global_ids_cpu.detach().to(device="cpu", dtype=torch.long)
        if global_ids_cpu.numel() == 0:
            return torch.empty(0, dtype=torch.long)
        return self.token_event_step_count_cpu.index_select(0, global_ids_cpu) + 1

    def _increment_token_event_step_counts_(self, global_ids_cpu: torch.Tensor) -> None:
        global_ids_cpu = global_ids_cpu.detach().to(device="cpu", dtype=torch.long)
        if global_ids_cpu.numel() == 0:
            return
        unique_ids_cpu = torch.unique(global_ids_cpu, sorted=True)
        current_counts = self.token_event_step_count_cpu.index_select(0, unique_ids_cpu)
        self.token_event_step_count_cpu.index_copy_(0, unique_ids_cpu, current_counts + 1)

    def _mark_table_updated_(self, param_name: str, updated_tables: set[str]) -> None:
        if param_name in updated_tables:
            return
        spec = self.table_specs[param_name]
        self.state[spec["param"]]["step"] += 1
        updated_tables.add(param_name)

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

    def _empty_long_cpu(self) -> torch.Tensor:
        return torch.empty(0, dtype=torch.long)

    def _update_token_counts_from_local_batch(self, step_meta: dict) -> None:
        inputs_cpu_local = step_meta.get("inputs_cpu_local")
        if inputs_cpu_local is None:
            return
        slot_to_global_cpu = step_meta["slot_to_global_cpu"].detach().to(device="cpu", dtype=torch.long)
        local_ids = inputs_cpu_local.detach().to(device="cpu", dtype=torch.long).reshape(-1)
        global_ids = slot_to_global_cpu.index_select(0, local_ids)
        valid_ids = global_ids[global_ids >= 0]
        if valid_ids.numel() == 0:
            return
        counts = torch.ones_like(valid_ids, dtype=self.global_token_count_cpu.dtype)
        self.global_token_count_cpu.index_add_(0, valid_ids, counts)
        self.token_event_step_count_cpu.index_add_(0, valid_ids, counts)

    def _rank_tokens_by_frequency(self, excluded_mask_cpu: torch.Tensor, limit: int) -> torch.Tensor:
        if limit <= 0:
            return self._empty_long_cpu()
        counts = self.global_token_count_cpu.clone()
        counts.masked_fill_(excluded_mask_cpu, -1)
        positive_count = int((counts >= 0).sum().item())
        if positive_count <= 0:
            return self._empty_long_cpu()
        topk_count = min(limit, positive_count)
        top_values, top_ids = torch.topk(counts, k=topk_count)
        valid_mask = top_values >= 0
        return top_ids[valid_mask]

    def _get_resident_cloud_ids_cpu(self) -> torch.Tensor:
        if not self.fixed_u_mode or not self._fixed_live_state:
            return self._empty_long_cpu()
        if self.fixed_lm_head_slot_to_global_cpu is None:
            return self._empty_long_cpu()
        cloud_slot_mask_cpu = self.fixed_lm_head_slot_to_global_cpu >= 0
        if self.fixed_active_mask_cpu is not None:
            cloud_slot_mask_cpu = cloud_slot_mask_cpu.clone()
            cloud_slot_mask_cpu[:self.fixed_u_max] &= ~self.fixed_active_mask_cpu
        cloud_slot_ids_cpu = torch.nonzero(cloud_slot_mask_cpu, as_tuple=False).flatten()
        if cloud_slot_ids_cpu.numel() == 0:
            return self._empty_long_cpu()
        return self.fixed_lm_head_slot_to_global_cpu.index_select(0, cloud_slot_ids_cpu)

    def _select_router_positions(self, query_count: int, max_positions: int) -> torch.Tensor:
        if query_count <= 0:
            return self._empty_long_cpu()
        if max_positions <= 0 or query_count <= max_positions:
            return torch.arange(query_count, dtype=torch.long)
        # Evenly subsample positions to keep CPU routing bounded without biasing to the prefix.
        return torch.linspace(0, query_count - 1, steps=max_positions, dtype=torch.float32).round().to(dtype=torch.long)

    def _select_hidden_query_positions(self, seq_len: int, subsample_strategy: str, num_samples: int) -> torch.Tensor:
        if seq_len <= 0 or num_samples <= 0:
            return self._empty_long_cpu()
        if subsample_strategy == "last":
            window_start = max(0, seq_len - min(512, seq_len))
            positions = torch.arange(window_start, seq_len, dtype=torch.long)
            if positions.numel() <= num_samples:
                return positions
            step = max(1, positions.numel() // num_samples)
            return positions[::step][:num_samples]
        if subsample_strategy != "uniform":
            raise ValueError(f"Unsupported hidden query subsample strategy '{subsample_strategy}'")
        step = max(1, seq_len // num_samples)
        positions = torch.arange(step // 2, seq_len, step, dtype=torch.long)
        if positions.numel() == 0:
            positions = torch.tensor([seq_len - 1], dtype=torch.long)
        return positions[:num_samples]

    def _build_hidden_query_active_vocab(
        self,
        active_ids_cpu: torch.Tensor,
        active_slot_ids_cpu: torch.Tensor,
    ) -> dict:
        active_ids_cpu = active_ids_cpu.detach().to(device="cpu", dtype=torch.long)
        active_slot_ids_cpu = active_slot_ids_cpu.detach().to(device="cpu", dtype=torch.long)
        active_slot_ids_device = active_slot_ids_cpu.to(self.device)
        active_vocab = {
            "wte": torch.zeros_like(self.fixed_params["wte"], device=self.device),
            "value_embeds": {},
        }
        if active_ids_cpu.numel() == 0:
            for name in self.fixed_params:
                if name.startswith("value_embeds."):
                    layer_name = name.split(".", 1)[1]
                    active_vocab["value_embeds"][layer_name] = torch.zeros_like(self.fixed_params[name], device=self.device)
            return active_vocab

        for name, spec in self.table_specs.items():
            if name == "lm_head":
                continue
            param = spec["param"]
            rows = param.index_select(0, active_ids_cpu)
            rows_gpu = rows.pin_memory().to(self.device, non_blocking=self.use_cuda) if self.use_cuda else rows.to(self.device)
            if name == "wte":
                active_vocab["wte"].index_copy_(0, active_slot_ids_device, rows_gpu)
            else:
                layer_name = name.split(".", 1)[1]
                value_rows = torch.zeros_like(self.fixed_params[name], device=self.device)
                value_rows.index_copy_(0, active_slot_ids_device, rows_gpu)
                active_vocab["value_embeds"][layer_name] = value_rows
        return active_vocab

    def get_subsampled_hidden_queries(
        self,
        preview_tokens: torch.Tensor,
        active_ids_cpu: torch.Tensor,
        active_slot_ids_cpu: torch.Tensor,
        subsample_strategy: str = "uniform",
        num_samples: int = 32,
        max_prefix_len: int = 2048,
    ) -> torch.Tensor:
        preview_tokens = preview_tokens.detach().to(device="cpu", dtype=torch.long)
        hidden_query_t0 = time.perf_counter()
        if preview_tokens.dim() == 1:
            preview_tokens = preview_tokens.unsqueeze(0)
        elif preview_tokens.dim() != 2:
            raise ValueError(f"preview_tokens must have rank 1 or 2, got shape {tuple(preview_tokens.shape)}")
        positions = self._select_hidden_query_positions(preview_tokens.shape[1], subsample_strategy, num_samples)
        if positions.numel() == 0:
            self._last_hidden_query_ms = (time.perf_counter() - hidden_query_t0) * 1000.0
            return torch.empty((0, self.model.config.n_embd), dtype=torch.float32)

        active_vocab = self._build_hidden_query_active_vocab(active_ids_cpu, active_slot_ids_cpu)
        with torch.inference_mode():
            seq_len = int(preview_tokens.shape[1])
            use_single_forward = max_prefix_len <= 0 or seq_len <= max_prefix_len
            if use_single_forward:
                preview_device = preview_tokens.to(self.device, non_blocking=self.use_cuda)
                hidden = self.model.forward_features(preview_device, active_vocab=active_vocab)
                selected_hidden = hidden.index_select(1, positions.to(device=hidden.device))
                queries = selected_hidden.mean(dim=0).to(device="cpu", dtype=torch.float32)
                self._last_hidden_query_ms = (time.perf_counter() - hidden_query_t0) * 1000.0
                return queries

            suffix_start = max(0, seq_len - max_prefix_len)
            if int(positions.min().item()) >= suffix_start:
                truncated_preview = preview_tokens[:, suffix_start:]
                truncated_positions = positions - suffix_start
                preview_device = truncated_preview.to(self.device, non_blocking=self.use_cuda)
                hidden = self.model.forward_features(preview_device, active_vocab=active_vocab)
                selected_hidden = hidden.index_select(1, truncated_positions.to(device=hidden.device))
                queries = selected_hidden.mean(dim=0).to(device="cpu", dtype=torch.float32)
                self._last_hidden_query_ms = (time.perf_counter() - hidden_query_t0) * 1000.0
                return queries

            # Exact fallback for restrictive max_prefix_len settings that would otherwise change causal context.
            queries = []
            for pos in positions.tolist():
                prefix = preview_tokens[:, : pos + 1]
                if max_prefix_len > 0 and prefix.shape[1] > max_prefix_len:
                    prefix = prefix[:, -max_prefix_len:]
                prefix_device = prefix.to(self.device, non_blocking=self.use_cuda)
                hidden = self.model.forward_features(prefix_device, active_vocab=active_vocab)
                h_pos = hidden[:, -1, :]
                h_pos = h_pos.mean(dim=0) if h_pos.shape[0] > 1 else h_pos.squeeze(0)
                queries.append(h_pos.detach().to(device="cpu", dtype=torch.float32))
            self._last_hidden_query_ms = (time.perf_counter() - hidden_query_t0) * 1000.0
            return torch.stack(queries) if queries else torch.empty((0, self.model.config.n_embd), dtype=torch.float32)

    def _build_preview_queries_cpu(
        self,
        next_batch_preview: torch.Tensor,
        hidden_states_next: Optional[torch.Tensor] = None,
        max_positions: int = 0,
    ) -> torch.Tensor:
        if hidden_states_next is not None:
            hidden_states_next = hidden_states_next.detach().to(device="cpu", dtype=torch.float32)
            if hidden_states_next.dim() == 2:
                queries = hidden_states_next
            elif hidden_states_next.dim() == 3:
                queries = hidden_states_next.mean(dim=0)
            else:
                raise ValueError(f"hidden_states_next must have rank 2 or 3, got shape {tuple(hidden_states_next.shape)}")
        else:
            preview_tokens = next_batch_preview.detach().to(device="cpu", dtype=torch.long)
            if preview_tokens.dim() == 1:
                preview_tokens = preview_tokens.unsqueeze(0)
            elif preview_tokens.dim() != 2:
                raise ValueError(f"next_batch_preview must have rank 1 or 2, got shape {tuple(preview_tokens.shape)}")
            token_embeds = self.table_specs["wte"]["param"].index_select(0, preview_tokens.reshape(-1)).to(dtype=torch.float32)
            token_embeds = token_embeds.view(preview_tokens.size(0), preview_tokens.size(1), -1)
            queries = token_embeds.mean(dim=0)
            if queries.size(0) > 1:
                padded = F.pad(queries.transpose(0, 1).unsqueeze(0), (1, 1), mode="replicate")
                queries = F.avg_pool1d(padded, kernel_size=3, stride=1).squeeze(0).transpose(0, 1)

        selected_positions = self._select_router_positions(int(queries.size(0)), max_positions)
        if selected_positions.numel() == 0:
            return torch.empty((0, self.table_specs["wte"]["param"].shape[1]), dtype=torch.float32)
        return queries.index_select(0, selected_positions)

    def select_warm_cloud(
        self,
        next_batch_preview: torch.Tensor,
        current_step_u_ids: torch.Tensor,
        *,
        max_warm: int,
        topk_per_position: int = 500,
        freq_boost_power: float = 0.0,
        hidden_states_next: Optional[torch.Tensor] = None,
        router_weights: Optional[torch.Tensor] = None,
        shortlist_limit: int = 20000,
        max_positions: int = 0,
        preview_active_ids_cpu: Optional[torch.Tensor] = None,
        preview_active_slot_ids_cpu: Optional[torch.Tensor] = None,
        hidden_query_strategy: str = "uniform",
        hidden_query_max_prefix_len: int = 2048,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current_step_u_ids = current_step_u_ids.detach().to(device="cpu", dtype=torch.long)
        if max_warm <= 0:
            empty = self._empty_long_cpu()
            return empty, empty

        excluded_mask_cpu = torch.zeros(self.model.config.vocab_size, dtype=torch.bool)
        excluded_mask_cpu[current_step_u_ids] = True

        if shortlist_limit > 0:
            effective_shortlist_limit = max(int(shortlist_limit), int(max_warm))
            candidate_ids = self._rank_tokens_by_frequency(excluded_mask_cpu, limit=effective_shortlist_limit)
        else:
            candidate_ids = torch.nonzero(~excluded_mask_cpu, as_tuple=False).flatten()
        if candidate_ids.numel() == 0:
            empty = self._empty_long_cpu()
            return empty, empty

        queries = self._build_preview_queries_cpu(
            next_batch_preview,
            hidden_states_next=hidden_states_next,
            max_positions=max_positions,
        )
        if queries.numel() == 0:
            empty = self._empty_long_cpu()
            return empty, empty

        if router_weights is not None:
            router_weights = router_weights.detach().to(device="cpu", dtype=torch.float32)
            candidate_matrix = router_weights.index_select(1, candidate_ids).transpose(0, 1).contiguous()
        else:
            candidate_matrix = self.table_specs["lm_head"]["param"].index_select(0, candidate_ids).to(dtype=torch.float32)

        queries = F.normalize(queries, dim=-1)
        candidate_matrix = F.normalize(candidate_matrix, dim=-1)
        warm_scores = torch.zeros(candidate_ids.numel(), dtype=torch.float32)
        topk = min(int(topk_per_position), int(candidate_ids.numel()))
        if topk <= 0:
            empty = self._empty_long_cpu()
            return empty, empty

        logits = queries @ candidate_matrix.transpose(0, 1)
        top_values, top_indices = torch.topk(logits, k=topk, dim=1)
        position_scores = torch.softmax(top_values, dim=1)
        warm_scores.scatter_add_(0, top_indices.reshape(-1), position_scores.reshape(-1))

        if freq_boost_power > 0.0:
            freq = self.global_token_count_cpu.index_select(0, candidate_ids).to(dtype=torch.float32)
            freq = freq.pow(float(freq_boost_power))
            warm_scores += torch.log1p(freq) * 0.1

        warm_scores[excluded_mask_cpu.index_select(0, candidate_ids)] = -float("inf")
        ranked_indices = torch.argsort(warm_scores, descending=True)
        valid_ranked = ranked_indices[warm_scores.index_select(0, ranked_indices) > -float("inf")]
        ranked_candidate_ids = candidate_ids.index_select(0, valid_ranked)
        warm_ids = ranked_candidate_ids[: min(max_warm, ranked_candidate_ids.numel())]
        return warm_ids, ranked_candidate_ids

    def plan_next_lm_head_cloud(
        self,
        step_meta: dict,
        *,
        warm_proportion: float,
        router_candidate_pool_size: int,
        router_topk: int,
        source_token_limit: int,
        hidden_query_strategy: str = "uniform",
        hidden_query_max_prefix_len: int = 2048,
    ) -> dict:
        plan_t0 = time.perf_counter()
        planned_step_meta = dict(step_meta)
        planned_step_meta["warm_ids_cpu"] = self._empty_long_cpu()
        planned_step_meta["cold_ids_cpu"] = self._empty_long_cpu()
        planned_step_meta["warm_candidate_ids_cpu"] = self._empty_long_cpu()
        planned_step_meta["lm_head_u_max"] = int(self.lm_head_u_max)
        planned_step_meta["cloud_residual_capacity"] = 0
        planned_step_meta["warm_budget_target"] = 0
        planned_step_meta["cold_budget_target"] = 0
        planned_step_meta["warm_candidate_count"] = 0
        planned_step_meta["cloud_plan_ms"] = 0.0
        planned_step_meta["cloud_hidden_query_ms"] = 0.0
        planned_step_meta["cloud_selection_ms"] = 0.0

        self._update_token_counts_from_local_batch(planned_step_meta)

        if not self.fixed_u_mode or self.lm_head_u_max <= self.fixed_u_max:
            planned_step_meta["cloud_plan_ms"] = (time.perf_counter() - plan_t0) * 1000.0
            return planned_step_meta

        step_ids_cpu = planned_step_meta["active_ids_cpu"].detach().to(device="cpu", dtype=torch.long)
        residual_capacity = max(self.lm_head_u_max - int(step_ids_cpu.numel()), 0)
        planned_step_meta["cloud_residual_capacity"] = int(residual_capacity)
        if residual_capacity <= 0:
            planned_step_meta["cloud_plan_ms"] = (time.perf_counter() - plan_t0) * 1000.0
            return planned_step_meta

        warm_budget = min(residual_capacity, max(int(round(residual_capacity * float(warm_proportion))), 0))
        cold_budget = residual_capacity - warm_budget
        planned_step_meta["warm_budget_target"] = int(warm_budget)
        planned_step_meta["cold_budget_target"] = int(cold_budget)

        excluded_mask_cpu = torch.zeros(self.model.config.vocab_size, dtype=torch.bool)
        excluded_mask_cpu[step_ids_cpu] = True
        prev_cloud_ids_cpu = self._get_resident_cloud_ids_cpu()
        prev_cloud_mask_cpu = torch.zeros(self.model.config.vocab_size, dtype=torch.bool)
        if prev_cloud_ids_cpu.numel() > 0:
            prev_cloud_mask_cpu[prev_cloud_ids_cpu] = True

        warm_candidate_ids_cpu = self._empty_long_cpu()
        warm_selection_t0 = time.perf_counter()
        self._last_hidden_query_ms = 0.0
        if warm_budget > 0 and router_candidate_pool_size > 0 and router_topk > 0:
            inputs_cpu_local = planned_step_meta.get("inputs_cpu_local")
            if inputs_cpu_local is not None:
                warm_ids_cpu, warm_candidate_ids_cpu = self.select_warm_cloud(
                    inputs_cpu_local.detach().to(device="cpu", dtype=torch.long),
                    step_ids_cpu,
                    max_warm=warm_budget,
                    topk_per_position=router_topk,
                    shortlist_limit=router_candidate_pool_size,
                    max_positions=source_token_limit,
                    preview_active_ids_cpu=step_ids_cpu,
                    preview_active_slot_ids_cpu=planned_step_meta["active_slot_ids_cpu"],
                    hidden_query_strategy=hidden_query_strategy,
                    hidden_query_max_prefix_len=hidden_query_max_prefix_len,
                )
            else:
                warm_ids_cpu = self._empty_long_cpu()
        else:
            warm_ids_cpu = self._empty_long_cpu()
        if warm_candidate_ids_cpu.numel() > 0:
            ranked_prev_warm_mask_cpu = prev_cloud_mask_cpu.index_select(0, warm_candidate_ids_cpu)
            retained_warm_ids_cpu = warm_candidate_ids_cpu[ranked_prev_warm_mask_cpu][:warm_budget]
            if retained_warm_ids_cpu.numel() > 0:
                new_warm_ids_cpu = warm_candidate_ids_cpu[~ranked_prev_warm_mask_cpu]
                remaining_budget = warm_budget - int(retained_warm_ids_cpu.numel())
                warm_ids_cpu = torch.cat((retained_warm_ids_cpu, new_warm_ids_cpu[:remaining_budget])) if remaining_budget > 0 else retained_warm_ids_cpu
        if warm_ids_cpu.numel() < warm_budget:
            warm_fill_excluded_cpu = excluded_mask_cpu.clone()
            if warm_candidate_ids_cpu.numel() > 0:
                warm_fill_excluded_cpu[warm_candidate_ids_cpu] = True
            if warm_ids_cpu.numel() > 0:
                warm_fill_excluded_cpu[warm_ids_cpu] = True
            warm_fill_ids_cpu = self._rank_tokens_by_frequency(
                warm_fill_excluded_cpu,
                limit=warm_budget - int(warm_ids_cpu.numel()),
            )
            if warm_fill_ids_cpu.numel() > 0:
                warm_ids_cpu = torch.cat((warm_ids_cpu, warm_fill_ids_cpu)) if warm_ids_cpu.numel() > 0 else warm_fill_ids_cpu
                warm_candidate_ids_cpu = (
                    torch.cat((warm_candidate_ids_cpu, warm_fill_ids_cpu))
                    if warm_candidate_ids_cpu.numel() > 0 else warm_fill_ids_cpu
                )
        planned_step_meta["cloud_hidden_query_ms"] = float(self._last_hidden_query_ms)
        planned_step_meta["cloud_selection_ms"] = (time.perf_counter() - warm_selection_t0) * 1000.0

        if warm_candidate_ids_cpu.numel() > 0:
            excluded_mask_cpu[warm_candidate_ids_cpu] = True
        cold_ids_cpu = self._empty_long_cpu()
        if cold_budget > 0:
            cold_candidate_pool = max(int(cold_budget) * 4, int(cold_budget))
            cold_candidate_ids_cpu = self._rank_tokens_by_frequency(excluded_mask_cpu, cold_candidate_pool)
            if cold_candidate_ids_cpu.numel() > 0:
                ranked_prev_cold_mask_cpu = prev_cloud_mask_cpu.index_select(0, cold_candidate_ids_cpu)
                retained_cold_ids_cpu = cold_candidate_ids_cpu[ranked_prev_cold_mask_cpu][:cold_budget]
                if retained_cold_ids_cpu.numel() > 0:
                    new_cold_ids_cpu = cold_candidate_ids_cpu[~ranked_prev_cold_mask_cpu]
                    remaining_budget = cold_budget - int(retained_cold_ids_cpu.numel())
                    cold_ids_cpu = torch.cat((retained_cold_ids_cpu, new_cold_ids_cpu[:remaining_budget])) if remaining_budget > 0 else retained_cold_ids_cpu
                else:
                    cold_ids_cpu = cold_candidate_ids_cpu[:cold_budget]
            if cold_ids_cpu.numel() < cold_budget:
                cold_fill_excluded_cpu = excluded_mask_cpu.clone()
                if cold_ids_cpu.numel() > 0:
                    cold_fill_excluded_cpu[cold_ids_cpu] = True
                cold_fill_ids_cpu = self._rank_tokens_by_frequency(
                    cold_fill_excluded_cpu,
                    cold_budget - int(cold_ids_cpu.numel()),
                )
                if cold_fill_ids_cpu.numel() > 0:
                    cold_ids_cpu = torch.cat((cold_ids_cpu, cold_fill_ids_cpu)) if cold_ids_cpu.numel() > 0 else cold_fill_ids_cpu

        planned_step_meta["warm_ids_cpu"] = warm_ids_cpu
        planned_step_meta["cold_ids_cpu"] = cold_ids_cpu
        planned_step_meta["warm_candidate_ids_cpu"] = warm_candidate_ids_cpu
        planned_step_meta["warm_candidate_count"] = int(warm_candidate_ids_cpu.numel())
        planned_step_meta["cloud_plan_ms"] = (time.perf_counter() - plan_t0) * 1000.0
        return planned_step_meta

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
        futures = self._pending_cpu_writeback_futures
        if futures is None and self._pending_cpu_writeback_future is not None:
            futures = [self._pending_cpu_writeback_future]
        if not futures:
            return
        if required_ids_cpu is not None and self._pending_cpu_writeback_mask_cpu is not None:
            required_ids_cpu = required_ids_cpu.detach().to(device="cpu", dtype=torch.long)
            if required_ids_cpu.numel() > 0 and not self._pending_cpu_writeback_mask_cpu[required_ids_cpu].any() and not all(future.done() for future in futures):
                return
        for future in futures:
            future.result()
        self._pending_cpu_writeback_futures = None
        self._pending_cpu_writeback_future = None
        self._pending_cpu_writeback_mask_cpu = None

    def _append_pending_cpu_writeback_futures(self, futures, ids_cpu: torch.Tensor) -> None:
        if not futures:
            return
        ids_cpu = ids_cpu.detach().to(device="cpu", dtype=torch.long)
        if self._pending_cpu_writeback_futures is None:
            self._pending_cpu_writeback_futures = []
        self._pending_cpu_writeback_futures.extend(futures)
        self._pending_cpu_writeback_future = self._pending_cpu_writeback_futures[0]
        if self._pending_cpu_writeback_mask_cpu is None:
            pending_mask_cpu = torch.zeros(self.model.config.vocab_size, dtype=torch.bool)
            if ids_cpu.numel() > 0:
                pending_mask_cpu[ids_cpu] = True
            self._pending_cpu_writeback_mask_cpu = pending_mask_cpu
        elif ids_cpu.numel() > 0:
            self._pending_cpu_writeback_mask_cpu[ids_cpu] = True

    def _has_active_pending_cpu_writeback(self) -> bool:
        futures = self._pending_cpu_writeback_futures
        if futures is None and self._pending_cpu_writeback_future is not None:
            futures = [self._pending_cpu_writeback_future]
        if not futures:
            return False
        return not all(future.done() for future in futures)

    def _zero_fixed_grad_slots_(self, slot_ids_cpu: Optional[torch.Tensor], table_names: Optional[tuple[str, ...]] = None) -> None:
        if not self.fixed_u_mode or slot_ids_cpu is None:
            return
        slot_ids_cpu = slot_ids_cpu.detach().to(device="cpu", dtype=torch.long)
        if slot_ids_cpu.numel() == 0:
            return
        if table_names is None:
            table_names = tuple(self.fixed_params.keys())
        slot_ids_device = slot_ids_cpu.to(self.device)
        for name in table_names:
            param = self.fixed_params[name]
            if param.grad is not None:
                param.grad.index_fill_(0, slot_ids_device, 0)

    def _union_input_table_names(self) -> tuple[str, ...]:
        return tuple(name for name in self.table_specs if name != "lm_head")

    def _invalidate_fixed_live_state(self):
        if not self.fixed_u_mode:
            return
        self._fixed_live_state = False
        if self.fixed_slot_to_global_cpu is not None:
            self.fixed_slot_to_global_cpu.fill_(-1)
        if self.fixed_input_slot_to_global_cpu is not None:
            self.fixed_input_slot_to_global_cpu.fill_(-1)
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
        self._grad_accum_hot_activation_counts_cpu = None
        self._grad_accum_window_cold_steps_cpu = None
        self._grad_accum_window_cold_logit_bias_cpu = None
        self._grad_accum_window_cold_bias_clamped_count = 0
        self._grad_accum_window_cold_bias_abs_max = 0.0

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
        self._grad_accum_hot_activation_counts_cpu = (
            self.hot_activation_count_cpu.index_select(0, grad_accum_ids_cpu)
            if grad_accum_count > 0 else torch.empty(0, dtype=torch.long)
        )
        self._grad_accum_window_cold_steps_cpu = None
        self._grad_accum_window_cold_logit_bias_cpu = None
        self._grad_accum_window_cold_bias_clamped_count = 0
        self._grad_accum_window_cold_bias_abs_max = 0.0

    def _cache_grad_accum_leaving_rows_(
        self,
        accum_row_ids_cpu: torch.Tensor,
        leaving_slot_ids_cpu: torch.Tensor,
        table_names: Optional[tuple[str, ...]] = None,
    ) -> None:
        if accum_row_ids_cpu.numel() == 0:
            return
        if self._grad_accum_cached_union_mask_cpu is None:
            raise RuntimeError("Sparse grad accumulation cache mask is missing while caching leaving rows")
        if table_names is None:
            table_names = tuple(self.table_specs.keys())

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
        for name in table_names:
            chunk["tables"][name] = {
                "param": nn.Parameter(
                    self.fixed_params[name].detach().index_select(0, cache_slot_ids_device).clone(),
                    requires_grad=True,
                ),
                "exp_avg": self.fixed_optimizer_state[name]["exp_avg"].detach().index_select(0, cache_slot_ids_device).clone(),
                "exp_avg_sq": self.fixed_optimizer_state[name]["exp_avg_sq"].detach().index_select(0, cache_slot_ids_device).clone(),
            }
        self._grad_accum_non_live_chunks.append(chunk)

    @torch.no_grad()
    def _writeback_fixed_rows_(
        self,
        global_ids_cpu: torch.Tensor,
        slot_ids_cpu: torch.Tensor,
        table_names: Optional[tuple[str, ...]] = None,
    ):
        self._flush_pending_cpu_writeback(global_ids_cpu)
        global_ids_cpu = global_ids_cpu.detach().to(device="cpu", dtype=torch.long)
        slot_ids_cpu = slot_ids_cpu.detach().to(device="cpu", dtype=torch.long)
        if global_ids_cpu.numel() == 0:
            return
        if table_names is None:
            table_names = tuple(self.table_specs.keys())
        slot_ids_device = slot_ids_cpu.to(self.device)
        cpu_rows = {}
        cpu_exp_avg = {}
        cpu_exp_avg_sq = {}
        for name in table_names:
            spec = self.table_specs[name]
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
        for name in table_names:
            spec = self.table_specs[name]
            param = spec["param"]
            state = self.state[param]
            param.index_copy_(0, global_ids_cpu, cpu_rows[name])
            state["exp_avg"].index_copy_(0, global_ids_cpu, cpu_exp_avg[name])
            state["exp_avg_sq"].index_copy_(0, global_ids_cpu, cpu_exp_avg_sq[name])

    @torch.no_grad()
    def _writeback_fixed_lm_head_rows_(self, global_ids_cpu: torch.Tensor, slot_ids_cpu: torch.Tensor):
        self._flush_pending_cpu_writeback(global_ids_cpu)
        global_ids_cpu = global_ids_cpu.detach().to(device="cpu", dtype=torch.long)
        slot_ids_cpu = slot_ids_cpu.detach().to(device="cpu", dtype=torch.long)
        if global_ids_cpu.numel() == 0:
            return
        slot_ids_device = slot_ids_cpu.to(self.device)
        active_param = self.fixed_params["lm_head"]
        active_state = self.fixed_optimizer_state["lm_head"]
        rows = active_param.detach().index_select(0, slot_ids_device)
        exp_avg = active_state["exp_avg"].detach().index_select(0, slot_ids_device)
        exp_avg_sq = active_state["exp_avg_sq"].detach().index_select(0, slot_ids_device)
        row_buffer = self._get_cpu_receive_buffer(
            "rows:lm_head:cloud",
            tuple(rows.shape),
            rows.dtype,
            block_reuse_while_pending=True,
        )
        exp_avg_buffer = self._get_cpu_receive_buffer(
            "exp_avg:lm_head:cloud",
            tuple(exp_avg.shape),
            exp_avg.dtype,
            block_reuse_while_pending=True,
        )
        exp_avg_sq_buffer = self._get_cpu_receive_buffer(
            "exp_avg_sq:lm_head:cloud",
            tuple(exp_avg_sq.shape),
            exp_avg_sq.dtype,
            block_reuse_while_pending=True,
        )
        row_buffer.copy_(rows, non_blocking=self.use_cuda)
        exp_avg_buffer.copy_(exp_avg, non_blocking=self.use_cuda)
        exp_avg_sq_buffer.copy_(exp_avg_sq, non_blocking=self.use_cuda)
        if self.use_cuda:
            torch.cuda.synchronize(self.device)
        param = self.table_specs["lm_head"]["param"]
        state = self.state[param]
        param.index_copy_(0, global_ids_cpu, row_buffer)
        state["exp_avg"].index_copy_(0, global_ids_cpu, exp_avg_buffer)
        state["exp_avg_sq"].index_copy_(0, global_ids_cpu, exp_avg_sq_buffer)

    def _queue_fixed_rows_writeback_(self, global_ids_cpu: torch.Tensor, slot_ids_cpu: torch.Tensor) -> None:
        global_ids_cpu = global_ids_cpu.detach().to(device="cpu", dtype=torch.long)
        slot_ids_cpu = slot_ids_cpu.detach().to(device="cpu", dtype=torch.long)
        if global_ids_cpu.numel() == 0:
            return
        if not self.use_cuda:
            self._writeback_fixed_rows_(global_ids_cpu, slot_ids_cpu)
            return
        self._flush_pending_cpu_writeback(global_ids_cpu)
        slot_ids_device = slot_ids_cpu.to(self.device)
        writeback_segments = []
        for name, spec in self.table_specs.items():
            active_param = self.fixed_params[name]
            active_state = self.fixed_optimizer_state[name]
            rows = active_param.detach().index_select(0, slot_ids_device)
            exp_avg = active_state["exp_avg"].detach().index_select(0, slot_ids_device)
            exp_avg_sq = active_state["exp_avg_sq"].detach().index_select(0, slot_ids_device)
            row_buffer = self._get_cpu_receive_buffer(
                f"rows:{name}:fixed",
                tuple(rows.shape),
                rows.dtype,
                block_reuse_while_pending=True,
            )
            exp_avg_buffer = self._get_cpu_receive_buffer(
                f"exp_avg:{name}:fixed",
                tuple(exp_avg.shape),
                exp_avg.dtype,
                block_reuse_while_pending=True,
            )
            exp_avg_sq_buffer = self._get_cpu_receive_buffer(
                f"exp_avg_sq:{name}:fixed",
                tuple(exp_avg_sq.shape),
                exp_avg_sq.dtype,
                block_reuse_while_pending=True,
            )
            row_buffer.copy_(rows, non_blocking=True)
            exp_avg_buffer.copy_(exp_avg, non_blocking=True)
            exp_avg_sq_buffer.copy_(exp_avg_sq, non_blocking=True)
            param = spec["param"]
            state = self.state[param]
            writeback_segments.append((param, state, row_buffer, exp_avg_buffer, exp_avg_sq_buffer))
        ready_event = torch.cuda.Event()
        ready_event.record(torch.cuda.current_stream(self.device))
        def _write_rows_when_ready(event, ids_cpu, segments) -> None:
            event.synchronize()
            with torch.no_grad():
                for param, state, row_buffer, exp_avg_buffer, exp_avg_sq_buffer in segments:
                    param.index_copy_(0, ids_cpu, row_buffer)
                    state["exp_avg"].index_copy_(0, ids_cpu, exp_avg_buffer)
                    state["exp_avg_sq"].index_copy_(0, ids_cpu, exp_avg_sq_buffer)

        futures = [
            self._cpu_writeback_executor.submit(
                _write_rows_when_ready,
                ready_event,
                global_ids_cpu,
                [segment],
            )
            for segment in writeback_segments
        ]
        self._append_pending_cpu_writeback_futures(futures, global_ids_cpu)

    def _queue_fixed_lm_head_writeback_(self, global_ids_cpu: torch.Tensor, slot_ids_cpu: torch.Tensor) -> None:
        global_ids_cpu = global_ids_cpu.detach().to(device="cpu", dtype=torch.long)
        slot_ids_cpu = slot_ids_cpu.detach().to(device="cpu", dtype=torch.long)
        if global_ids_cpu.numel() == 0:
            return
        self._flush_pending_cpu_writeback(global_ids_cpu)
        slot_ids_device = slot_ids_cpu.to(self.device)
        active_param = self.fixed_params["lm_head"]
        active_state = self.fixed_optimizer_state["lm_head"]
        rows = active_param.detach().index_select(0, slot_ids_device)
        exp_avg = active_state["exp_avg"].detach().index_select(0, slot_ids_device)
        exp_avg_sq = active_state["exp_avg_sq"].detach().index_select(0, slot_ids_device)
        row_buffer = self._get_cpu_receive_buffer(
            "rows:lm_head:cloud",
            tuple(rows.shape),
            rows.dtype,
            block_reuse_while_pending=True,
        )
        exp_avg_buffer = self._get_cpu_receive_buffer(
            "exp_avg:lm_head:cloud",
            tuple(exp_avg.shape),
            exp_avg.dtype,
            block_reuse_while_pending=True,
        )
        exp_avg_sq_buffer = self._get_cpu_receive_buffer(
            "exp_avg_sq:lm_head:cloud",
            tuple(exp_avg_sq.shape),
            exp_avg_sq.dtype,
            block_reuse_while_pending=True,
        )
        row_buffer.copy_(rows, non_blocking=self.use_cuda)
        exp_avg_buffer.copy_(exp_avg, non_blocking=self.use_cuda)
        exp_avg_sq_buffer.copy_(exp_avg_sq, non_blocking=self.use_cuda)
        ready_event = None
        if self.use_cuda:
            ready_event = torch.cuda.Event()
            ready_event.record(torch.cuda.current_stream(self.device))
        param = self.table_specs["lm_head"]["param"]
        state = self.state[param]
        def _write_lm_head_rows_when_ready(event, ids_cpu, row_buf, exp_avg_buf, exp_avg_sq_buf) -> None:
            if event is not None:
                event.synchronize()
            with torch.no_grad():
                param.index_copy_(0, ids_cpu, row_buf)
                state["exp_avg"].index_copy_(0, ids_cpu, exp_avg_buf)
                state["exp_avg_sq"].index_copy_(0, ids_cpu, exp_avg_sq_buf)

        futures = [
            self._cpu_writeback_executor.submit(
                _write_lm_head_rows_when_ready,
                ready_event,
                global_ids_cpu,
                row_buffer,
                exp_avg_buffer,
                exp_avg_sq_buffer,
            )
        ]
        self._append_pending_cpu_writeback_futures(futures, global_ids_cpu)

    @torch.no_grad()
    def flush_active_to_cpu(self):
        self._flush_pending_cpu_writeback()
        if self._grad_accum_live:
            raise RuntimeError("Cannot flush active sparse rows while sparse accumulated gradients are pending")
        if not self.fixed_u_mode or not self._fixed_live_state:
            return
        assert self.fixed_input_slot_to_global_cpu is not None
        assert self.fixed_lm_head_slot_to_global_cpu is not None
        active_slot_ids_cpu = torch.nonzero(self.fixed_input_slot_to_global_cpu >= 0, as_tuple=False).flatten()
        if active_slot_ids_cpu.numel() == 0:
            active_ids_cpu = self._empty_long_cpu()
        else:
            active_ids_cpu = self.fixed_input_slot_to_global_cpu[active_slot_ids_cpu]
            self._writeback_fixed_rows_(
                active_ids_cpu,
                active_slot_ids_cpu,
                table_names=tuple(name for name in self.table_specs if name != "lm_head"),
            )
        lm_head_slot_ids_cpu = torch.nonzero(self.fixed_lm_head_slot_to_global_cpu >= 0, as_tuple=False).flatten()
        if lm_head_slot_ids_cpu.numel() > 0:
            lm_head_ids_cpu = self.fixed_lm_head_slot_to_global_cpu.index_select(0, lm_head_slot_ids_cpu)
            self._writeback_fixed_lm_head_rows_(lm_head_ids_cpu, lm_head_slot_ids_cpu)

    def _stage_rows_to_gpu(self, cpu_tensor_map):
        gpu_tensor_map = {}
        for name, tensor in cpu_tensor_map.items():
            gpu_tensor_map[name] = self._stage_cpu_tensor_to_device(name, tensor)
        return gpu_tensor_map

    def _get_gpu_stage_buffer(self, name: str, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        buffer = self._gpu_stage_buffers.get(name)
        needs_new = (
            buffer is None or
            buffer.dtype != dtype or
            buffer.dim() != len(shape) or
            any(buffer.size(dim) < shape_dim for dim, shape_dim in enumerate(shape))
        )
        if needs_new:
            buffer = torch.empty(shape, dtype=dtype, device=self.device)
            self._gpu_stage_buffers[name] = buffer
        assert buffer is not None
        slices = tuple(slice(0, dim) for dim in shape)
        return buffer[slices]

    def _stage_index_select_to_device(self, name: str, source_cpu: torch.Tensor, index_cpu: torch.Tensor) -> tuple[torch.Tensor, float, float, int]:
        source_cpu = source_cpu.detach()
        index_cpu = index_cpu.detach().to(device="cpu", dtype=torch.long)
        gather_t0 = time.perf_counter()
        if index_cpu.numel() == 0:
            shape = (0,) + tuple(source_cpu.shape[1:])
            if not self.use_cuda:
                gather_ms = (time.perf_counter() - gather_t0) * 1000.0
                return torch.empty(shape, dtype=source_cpu.dtype, device=self.device), gather_ms, 0.0, 0
            stage_buffer = self._get_cpu_receive_buffer(f"stage:{name}", shape, source_cpu.dtype)
            gather_ms = (time.perf_counter() - gather_t0) * 1000.0
            enqueue_t0 = time.perf_counter()
            gpu_buffer = self._get_gpu_stage_buffer(f"stage:{name}", shape, source_cpu.dtype)
            gpu_buffer.copy_(stage_buffer, non_blocking=True)
            enqueue_ms = (time.perf_counter() - enqueue_t0) * 1000.0
            return gpu_buffer, gather_ms, enqueue_ms, 0
        if not self.use_cuda:
            gathered = source_cpu.index_select(0, index_cpu).to(self.device)
            gather_ms = (time.perf_counter() - gather_t0) * 1000.0
            return gathered, gather_ms, 0.0, gathered.numel() * gathered.element_size()
        shape = (int(index_cpu.numel()),) + tuple(source_cpu.shape[1:])
        stage_buffer = self._get_cpu_receive_buffer(f"stage:{name}", shape, source_cpu.dtype)
        torch.index_select(source_cpu, 0, index_cpu, out=stage_buffer)
        gather_ms = (time.perf_counter() - gather_t0) * 1000.0
        enqueue_t0 = time.perf_counter()
        gpu_buffer = self._get_gpu_stage_buffer(f"stage:{name}", shape, source_cpu.dtype)
        gpu_buffer.copy_(stage_buffer, non_blocking=True)
        enqueue_ms = (time.perf_counter() - enqueue_t0) * 1000.0
        return gpu_buffer, gather_ms, enqueue_ms, stage_buffer.numel() * stage_buffer.element_size()

    def _stage_cpu_tensor_to_device(self, name: str, cpu_tensor: torch.Tensor) -> torch.Tensor:
        cpu_tensor = cpu_tensor.detach()
        if not self.use_cuda:
            return cpu_tensor.to(self.device)
        stage_buffer = self._get_cpu_receive_buffer(f"stage:{name}", tuple(cpu_tensor.shape), cpu_tensor.dtype)
        stage_buffer.copy_(cpu_tensor)
        return stage_buffer.to(self.device, non_blocking=True)

    def _stage_prefetched_cpu_tensor_to_device(self, name: str, cpu_tensor: torch.Tensor) -> tuple[torch.Tensor, float, int]:
        cpu_tensor = cpu_tensor.detach()
        if not self.use_cuda:
            gpu_tensor = cpu_tensor.to(self.device)
            return gpu_tensor, 0.0, gpu_tensor.numel() * gpu_tensor.element_size()
        enqueue_t0 = time.perf_counter()
        gpu_buffer = self._get_gpu_stage_buffer(f"stage:{name}", tuple(cpu_tensor.shape), cpu_tensor.dtype)
        gpu_buffer.copy_(cpu_tensor, non_blocking=True)
        enqueue_ms = (time.perf_counter() - enqueue_t0) * 1000.0
        return gpu_buffer, enqueue_ms, cpu_tensor.numel() * cpu_tensor.element_size()

    @torch.inference_mode()
    def _stage_mixed_prefetched_cpu_tensor_to_device(
        self,
        name: str,
        source_cpu: torch.Tensor,
        stage_ids_cpu: torch.Tensor,
        prefetched_cpu: torch.Tensor,
        prefetched_global_to_local_cpu: torch.Tensor,
    ) -> tuple[torch.Tensor, float, float, int]:
        source_cpu = source_cpu.detach()
        prefetched_cpu = prefetched_cpu.detach()
        gather_t0 = time.perf_counter()
        shape = (int(stage_ids_cpu.numel()),) + tuple(source_cpu.shape[1:])
        stage_buffer = self._get_cpu_receive_buffer(f"stage:{name}", shape, source_cpu.dtype)
        prefetched_local_ids_cpu = prefetched_global_to_local_cpu.index_select(0, stage_ids_cpu)
        prefetched_mask_cpu = prefetched_local_ids_cpu >= 0
        if prefetched_mask_cpu.any():
            prefetched_positions_cpu = torch.nonzero(prefetched_mask_cpu, as_tuple=False).flatten()
            prefetched_rows_cpu = prefetched_cpu.index_select(0, prefetched_local_ids_cpu[prefetched_mask_cpu])
            stage_buffer.index_copy_(0, prefetched_positions_cpu, prefetched_rows_cpu)
        missing_mask_cpu = ~prefetched_mask_cpu
        if missing_mask_cpu.any():
            missing_ids_cpu = stage_ids_cpu[missing_mask_cpu]
            missing_shape = (int(missing_ids_cpu.numel()),) + tuple(source_cpu.shape[1:])
            missing_buffer = self._get_cpu_receive_buffer(f"stage-miss:{name}", missing_shape, source_cpu.dtype)
            torch.index_select(source_cpu, 0, missing_ids_cpu, out=missing_buffer)
            missing_positions_cpu = torch.nonzero(missing_mask_cpu, as_tuple=False).flatten()
            stage_buffer.index_copy_(0, missing_positions_cpu, missing_buffer)
        gather_ms = (time.perf_counter() - gather_t0) * 1000.0
        if not self.use_cuda:
            gathered = stage_buffer.to(self.device)
            return gathered, gather_ms, 0.0, gathered.numel() * gathered.element_size()
        enqueue_t0 = time.perf_counter()
        gpu_buffer = self._get_gpu_stage_buffer(f"stage:{name}", shape, source_cpu.dtype)
        gpu_buffer.copy_(stage_buffer, non_blocking=True)
        enqueue_ms = (time.perf_counter() - enqueue_t0) * 1000.0
        return gpu_buffer, gather_ms, enqueue_ms, stage_buffer.numel() * stage_buffer.element_size()

    def _build_fixed_prefetch_requests(self, step_meta: dict) -> dict[str, torch.Tensor]:
        active_ids_cpu = step_meta["active_ids_cpu"].detach().to(device="cpu", dtype=torch.long)
        grad_accum_ids_cpu = step_meta.get("grad_accum_ids_cpu", active_ids_cpu).detach().to(device="cpu", dtype=torch.long)
        grad_accum_steps = int(step_meta.get("grad_accum_steps", 1))
        stage_ids_cpu = step_meta["stage_ids_cpu"].detach().to(device="cpu", dtype=torch.long)
        warm_ids_cpu = step_meta.get("warm_ids_cpu", self._empty_long_cpu()).detach().to(device="cpu", dtype=torch.long)
        cold_ids_cpu = step_meta.get("cold_ids_cpu", self._empty_long_cpu()).detach().to(device="cpu", dtype=torch.long)
        cloud_ids_cpu = torch.cat((warm_ids_cpu, cold_ids_cpu)) if warm_ids_cpu.numel() > 0 or cold_ids_cpu.numel() > 0 else self._empty_long_cpu()
        if grad_accum_steps > 1:
            if self._fixed_live_state and self.fixed_input_slot_to_global_cpu is not None and grad_accum_ids_cpu.numel() > 0:
                prev_input_slot_ids_cpu = torch.nonzero(self.fixed_input_slot_to_global_cpu >= 0, as_tuple=False).flatten()
                if prev_input_slot_ids_cpu.numel() > 0:
                    prev_input_ids_cpu = self.fixed_input_slot_to_global_cpu.index_select(0, prev_input_slot_ids_cpu)
                    prev_input_global_to_slot_cpu = torch.full((self.model.config.vocab_size,), -1, dtype=torch.long)
                    prev_input_global_to_slot_cpu[prev_input_ids_cpu] = prev_input_slot_ids_cpu
                    reused_old_slot_ids_cpu = prev_input_global_to_slot_cpu.index_select(0, grad_accum_ids_cpu)
                    non_lm_head_ids_cpu = grad_accum_ids_cpu if self.disable_fixed_overlap_reuse else grad_accum_ids_cpu[reused_old_slot_ids_cpu < 0]
                else:
                    non_lm_head_ids_cpu = grad_accum_ids_cpu
            else:
                non_lm_head_ids_cpu = grad_accum_ids_cpu
        else:
            non_lm_head_ids_cpu = stage_ids_cpu if self._fixed_live_state else active_ids_cpu
        lm_head_ids_cpu = self._empty_long_cpu()
        if grad_accum_steps > 1:
            lm_head_ids_cpu = grad_accum_ids_cpu
            if (not self.disable_fixed_overlap_reuse) and self._fixed_live_state and grad_accum_ids_cpu.numel() > 0:
                assert self.fixed_lm_head_slot_to_global_cpu is not None
                prev_lm_head_slot_mask_cpu = self.fixed_lm_head_slot_to_global_cpu >= 0
                prev_lm_head_slot_ids_cpu = torch.nonzero(prev_lm_head_slot_mask_cpu, as_tuple=False).flatten()
                if prev_lm_head_slot_ids_cpu.numel() > 0:
                    prev_lm_head_ids_cpu = self.fixed_lm_head_slot_to_global_cpu.index_select(0, prev_lm_head_slot_ids_cpu)
                    prev_lm_head_global_to_slot_cpu = torch.full((self.model.config.vocab_size,), -1, dtype=torch.long)
                    prev_lm_head_global_to_slot_cpu[prev_lm_head_ids_cpu] = prev_lm_head_slot_ids_cpu
                    reused_old_slot_ids_cpu = prev_lm_head_global_to_slot_cpu.index_select(0, grad_accum_ids_cpu)
                    lm_head_ids_cpu = grad_accum_ids_cpu[reused_old_slot_ids_cpu < 0]
        else:
            if self._fixed_live_state:
                lm_head_ids_cpu = torch.cat((stage_ids_cpu, cloud_ids_cpu)) if cloud_ids_cpu.numel() > 0 else stage_ids_cpu
            else:
                lm_head_ids_cpu = torch.cat((active_ids_cpu, cloud_ids_cpu)) if cloud_ids_cpu.numel() > 0 else active_ids_cpu

        requests = {}
        if non_lm_head_ids_cpu.numel() > 0:
            requests["__non_lm_head__"] = non_lm_head_ids_cpu
        if lm_head_ids_cpu.numel() > 0:
            requests["lm_head"] = lm_head_ids_cpu
        return requests

    @torch.inference_mode()
    def _prefetch_fixed_step_cpu_rows(self, requests: dict[str, torch.Tensor]) -> dict[str, dict]:
        required_ids = []
        for ids_cpu in requests.values():
            if ids_cpu.numel() > 0:
                required_ids.append(ids_cpu.detach().to(device="cpu", dtype=torch.long))
        if required_ids:
            required_ids_cpu = torch.unique(torch.cat(required_ids), sorted=False)
            self._flush_pending_cpu_writeback(required_ids_cpu)
        payload = {}
        non_lm_head_ids_cpu = requests.get("__non_lm_head__")
        for name, spec in self.table_specs.items():
            ids_cpu = requests.get(name)
            if name != "lm_head":
                ids_cpu = non_lm_head_ids_cpu
            if ids_cpu is None or ids_cpu.numel() == 0:
                continue
            ids_cpu = ids_cpu.detach().to(device="cpu", dtype=torch.long)
            param = spec["param"].detach()
            state = self.state[spec["param"]]
            exp_avg_src = state["exp_avg"].detach()
            exp_avg_sq_src = state["exp_avg_sq"].detach()
            rows = torch.empty((ids_cpu.numel(),) + tuple(param.shape[1:]), dtype=param.dtype, pin_memory=self.use_cuda)
            exp_avg = torch.empty((ids_cpu.numel(),) + tuple(param.shape[1:]), dtype=param.dtype, pin_memory=self.use_cuda)
            exp_avg_sq = torch.empty((ids_cpu.numel(),) + tuple(param.shape[1:]), dtype=param.dtype, pin_memory=self.use_cuda)
            torch.index_select(param, 0, ids_cpu, out=rows)
            torch.index_select(exp_avg_src, 0, ids_cpu, out=exp_avg)
            torch.index_select(exp_avg_sq_src, 0, ids_cpu, out=exp_avg_sq)
            global_to_local_cpu = torch.full((self.model.config.vocab_size,), -1, dtype=torch.long)
            global_to_local_cpu[ids_cpu] = torch.arange(ids_cpu.numel(), dtype=torch.long)
            payload[name] = {
                "ids_cpu": ids_cpu,
                "global_to_local_cpu": global_to_local_cpu,
                "param": rows,
                "exp_avg": exp_avg,
                "exp_avg_sq": exp_avg_sq,
            }
        return payload

    def prefetch_step(self, active_ids_cpu) -> None:
        if not (self.use_cuda and self.fixed_u_mode and isinstance(active_ids_cpu, dict)):
            return
        requests = self._build_fixed_prefetch_requests(active_ids_cpu)
        if not requests:
            self._pending_stage_prefetch_future = None
            self._pending_stage_prefetch_key = None
            return
        self._pending_stage_prefetch_key = id(active_ids_cpu)
        self._pending_stage_prefetch_future = self._stage_prefetch_executor.submit(self._prefetch_fixed_step_cpu_rows, requests)

    @torch.no_grad()
    def _apply_fixed_lm_head_cold_row_decay_(
        self,
        exempt_global_ids_cpu: torch.Tensor,
        cold_row_decay: float,
    ) -> None:
        if cold_row_decay <= 0.0:
            return
        vocab_size = int(self.model.config.vocab_size)
        if vocab_size <= 0:
            return
        exempt_global_ids_cpu = exempt_global_ids_cpu.detach().to(device="cpu", dtype=torch.long)
        if exempt_global_ids_cpu.numel() > 0:
            valid_mask = (exempt_global_ids_cpu >= 0) & (exempt_global_ids_cpu < vocab_size)
            exempt_global_ids_cpu = exempt_global_ids_cpu[valid_mask]
            if exempt_global_ids_cpu.numel() > 1:
                exempt_global_ids_cpu = torch.unique(exempt_global_ids_cpu)
        if exempt_global_ids_cpu.numel() >= vocab_size:
            return
        self._flush_pending_cpu_writeback()
        lm_head_param = self.table_specs["lm_head"]["param"]
        restore_rows = None
        if exempt_global_ids_cpu.numel() > 0:
            restore_rows = lm_head_param.index_select(0, exempt_global_ids_cpu).clone()
        lm_head_param.narrow(0, 0, vocab_size).mul_(1.0 - float(cold_row_decay))
        if restore_rows is not None:
            lm_head_param.index_copy_(0, exempt_global_ids_cpu, restore_rows)

    def _get_cpu_receive_buffer(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        block_reuse_while_pending: bool = False,
    ):
        buffer = self._cpu_receive_buffers.get(name)
        is_inference_buffer = bool(buffer is not None and getattr(buffer, "is_inference", lambda: False)())
        needs_new = (
            buffer is None or
            (block_reuse_while_pending and self._has_active_pending_cpu_writeback()) or
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
        cold_row_decay: float = 0.0,
        cold_bias_tokens_per_step: Optional[int] = None,
    ) -> DynamicVocabStep:
        if cold_row_decay > 0.0:
            raise ValueError("sparse cold-row decay currently requires fixed-U sparse mode")
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

        gpu_rows = {
            name: self._stage_cpu_tensor_to_device(f"dynamic:param:{name}", rows)
            for name, rows in cpu_rows.items()
        }
        gpu_exp_avg = {
            name: self._stage_cpu_tensor_to_device(f"dynamic:exp_avg:{name}", exp_avg)
            for name, exp_avg in cpu_exp_avg.items()
        }
        gpu_exp_avg_sq = {
            name: self._stage_cpu_tensor_to_device(f"dynamic:exp_avg_sq:{name}", exp_avg_sq)
            for name, exp_avg_sq in cpu_exp_avg_sq.items()
        }

        active_vocab = {
            "wte": nn.Parameter(gpu_rows["wte"], requires_grad=True),
            "lm_head": nn.Parameter(gpu_rows["lm_head"], requires_grad=True),
            "value_embeds": {
                name.split(".", 1)[1]: nn.Parameter(gpu_rows[name], requires_grad=True)
                for name in gpu_rows
                if name.startswith("value_embeds.")
            },
        }
        if cold_bias_scale > 0.0:
            active_vocab["cold_logit_bias"] = cold_logit_bias_cpu.to(self.device, non_blocking=self.use_cuda)
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
        cold_row_decay: float = 0.0,
        cold_bias_tokens_per_step: Optional[int] = None,
        prefetched_stage: Optional[dict[str, dict]] = None,
        prefetched_wait_ms: float = 0.0,
        prefetched_hit: int = 0,
    ) -> DynamicVocabStep:
        assert self.fixed_u_mode, "Fixed-U step requested without fixed_u_max runtime configuration"
        active_ids_cpu = step_meta["active_ids_cpu"].detach().to(device="cpu", dtype=torch.long)
        grad_accum_ids_cpu = step_meta.get("grad_accum_ids_cpu", active_ids_cpu).detach().to(device="cpu", dtype=torch.long)
        grad_accum_steps = int(step_meta.get("grad_accum_steps", 1))
        grad_accum_micro_step = int(step_meta.get("grad_accum_micro_step", 0))
        is_grad_accum_boundary = bool(step_meta.get("is_grad_accum_boundary", True))
        active_slot_ids_cpu = step_meta["active_slot_ids_cpu"].detach().to(device="cpu", dtype=torch.long)
        union_inputs_cpu_local = step_meta.get("inputs_union_cpu_local")
        union_targets_cpu_local = step_meta.get("targets_union_cpu_local")
        warm_ids_cpu = step_meta.get("warm_ids_cpu", self._empty_long_cpu()).detach().to(device="cpu", dtype=torch.long)
        cold_ids_cpu = step_meta.get("cold_ids_cpu", self._empty_long_cpu()).detach().to(device="cpu", dtype=torch.long)
        union_input_tables = grad_accum_steps > 1
        preserve_resident_grads = (
            grad_accum_steps > 1 and
            self._grad_accum_live and
            self._grad_accum_ids_cpu is not None and
            torch.equal(self._grad_accum_ids_cpu, grad_accum_ids_cpu)
        )
        if self.disable_fixed_overlap_reuse and grad_accum_steps > 1 and not preserve_resident_grads and self._fixed_live_state:
            self.flush_active_to_cpu()
            self._fixed_live_state = False
        if grad_accum_steps > 1 and not preserve_resident_grads:
            self._start_grad_accum_window(grad_accum_ids_cpu)
        active_union_row_ids_cpu = self._empty_long_cpu()
        union_cold_steps_cpu = self._empty_long_cpu().to(dtype=torch.long)
        cold_steps_cpu = self._empty_long_cpu().to(dtype=torch.long)
        hot_activation_counts_cpu = self._empty_long_cpu().to(dtype=torch.long)
        if grad_accum_steps > 1:
            assert self._grad_accum_global_to_local_cpu is not None
            if active_ids_cpu.numel() > 0:
                active_union_row_ids_cpu = self._grad_accum_global_to_local_cpu[active_ids_cpu]
                if (active_union_row_ids_cpu < 0).any():
                    raise ValueError("Sparse grad accumulation map is missing active vocab rows")
            if not preserve_resident_grads:
                union_cold_steps_cpu, _ = self._capture_cold_steps_cpu(grad_accum_ids_cpu)
                self._grad_accum_window_cold_steps_cpu = union_cold_steps_cpu.clone()
            else:
                if self._grad_accum_window_cold_steps_cpu is None:
                    raise RuntimeError("Sparse grad accumulation cold-step cache is missing for resident window")
                union_cold_steps_cpu = self._grad_accum_window_cold_steps_cpu
            if self._grad_accum_hot_activation_counts_cpu is None:
                raise RuntimeError("Sparse grad accumulation hot-count cache is missing")
            if active_union_row_ids_cpu.numel() > 0:
                hot_activation_counts_cpu = self._grad_accum_hot_activation_counts_cpu.index_select(0, active_union_row_ids_cpu)
                cold_steps_cpu = union_cold_steps_cpu.index_select(0, active_union_row_ids_cpu)
        else:
            cold_steps_cpu, hot_activation_counts_cpu = self._capture_cold_steps_cpu(active_ids_cpu)
        current_cold_logit_bias_cpu = self._empty_long_cpu().to(dtype=torch.float32)
        cold_bias_clamped_count = 0
        cold_bias_abs_max = 0.0
        if grad_accum_steps > 1:
            if not preserve_resident_grads:
                current_cold_logit_bias_cpu, cold_bias_clamped_count, cold_bias_abs_max = self._compute_cold_logit_bias_with_stats_cpu(
                    union_cold_steps_cpu,
                    cold_bias_scale=cold_bias_scale,
                    cold_bias_tokens_per_step=cold_bias_tokens_per_step,
                )
                self._grad_accum_window_cold_logit_bias_cpu = current_cold_logit_bias_cpu.clone()
                self._grad_accum_window_cold_bias_clamped_count = cold_bias_clamped_count
                self._grad_accum_window_cold_bias_abs_max = cold_bias_abs_max
            else:
                if self._grad_accum_window_cold_logit_bias_cpu is None:
                    raise RuntimeError("Sparse grad accumulation cold-bias cache is missing for resident window")
                current_cold_logit_bias_cpu = self._grad_accum_window_cold_logit_bias_cpu
                cold_bias_clamped_count = self._grad_accum_window_cold_bias_clamped_count
                cold_bias_abs_max = self._grad_accum_window_cold_bias_abs_max
        else:
            current_cold_logit_bias_cpu, cold_bias_clamped_count, cold_bias_abs_max = self._compute_cold_logit_bias_with_stats_cpu(
                cold_steps_cpu,
                cold_bias_scale=cold_bias_scale,
                cold_bias_tokens_per_step=cold_bias_tokens_per_step,
            )
        active_mask_cpu = step_meta["active_mask_cpu"].detach().to(device="cpu", dtype=torch.bool)
        slot_to_global_cpu = step_meta["slot_to_global_cpu"].detach().to(device="cpu", dtype=torch.long)
        manifest_fixed_u_max = int(slot_to_global_cpu.numel())
        if manifest_fixed_u_max > self.fixed_u_max:
            raise ValueError(
                f"Manifest fixed-U metadata exceeds runtime capacity: manifest_u_max={manifest_fixed_u_max}, runtime_fixed_u_max={self.fixed_u_max}"
            )
        if manifest_fixed_u_max < self.fixed_u_max:
            padded_slot_to_global_cpu = torch.full((self.fixed_u_max,), -1, dtype=torch.long)
            padded_slot_to_global_cpu[:manifest_fixed_u_max].copy_(slot_to_global_cpu)
            slot_to_global_cpu = padded_slot_to_global_cpu

            padded_active_mask_cpu = torch.zeros(self.fixed_u_max, dtype=torch.bool)
            padded_active_mask_cpu[:manifest_fixed_u_max].copy_(active_mask_cpu)
            active_mask_cpu = padded_active_mask_cpu
        stage_ids_cpu = step_meta["stage_ids_cpu"].detach().to(device="cpu", dtype=torch.long)
        stage_slot_ids_cpu = step_meta["stage_slot_ids_cpu"].detach().to(device="cpu", dtype=torch.long)
        writeback_ids_cpu = step_meta["writeback_ids_cpu"].detach().to(device="cpu", dtype=torch.long)
        writeback_slot_ids_cpu = step_meta["writeback_slot_ids_cpu"].detach().to(device="cpu", dtype=torch.long)
        is_last_step = bool(step_meta["is_last_step"])
        cloud_ids_cpu = torch.cat((warm_ids_cpu, cold_ids_cpu)) if warm_ids_cpu.numel() > 0 or cold_ids_cpu.numel() > 0 else self._empty_long_cpu()
        if cloud_ids_cpu.numel() > 0 and grad_accum_steps > 1:
            raise ValueError("lm_head cloud expansion currently supports grad_accum_steps == 1 only")
        inactive_lm_head_slot_ids_cpu = torch.nonzero(~active_mask_cpu, as_tuple=False).flatten()
        overflow_lm_head_slot_ids_cpu = torch.arange(self.fixed_u_max, self.lm_head_u_max, dtype=torch.long)
        available_cloud_slot_ids_cpu = torch.cat((inactive_lm_head_slot_ids_cpu, overflow_lm_head_slot_ids_cpu))
        if cloud_ids_cpu.numel() > available_cloud_slot_ids_cpu.numel():
            raise ValueError(
                f"lm_head cloud overflow: need {cloud_ids_cpu.numel()} cloud slots, only have {available_cloud_slot_ids_cpu.numel()} available for lm_head_u_max={self.lm_head_u_max}"
            )
        cloud_slot_ids_cpu = self._empty_long_cpu()
        cloud_stage_ids_cpu = self._empty_long_cpu()
        cloud_stage_slot_ids_cpu = self._empty_long_cpu()
        cloud_writeback_ids_cpu = self._empty_long_cpu()
        cloud_writeback_slot_ids_cpu = self._empty_long_cpu()
        if cloud_ids_cpu.numel() > 0:
            if self._fixed_live_state:
                assert self.fixed_lm_head_slot_to_global_cpu is not None
                assert self.fixed_active_mask_cpu is not None
                prev_cloud_slot_mask_cpu = self.fixed_lm_head_slot_to_global_cpu >= 0
                prev_cloud_slot_mask_cpu[:self.fixed_u_max] &= ~self.fixed_active_mask_cpu
                prev_cloud_slot_ids_cpu = torch.nonzero(prev_cloud_slot_mask_cpu, as_tuple=False).flatten()
                prev_cloud_ids_cpu = (
                    self.fixed_lm_head_slot_to_global_cpu.index_select(0, prev_cloud_slot_ids_cpu)
                    if prev_cloud_slot_ids_cpu.numel() > 0 else self._empty_long_cpu()
                )
                available_cloud_slot_mask_cpu = torch.zeros(self.lm_head_u_max, dtype=torch.bool)
                if available_cloud_slot_ids_cpu.numel() > 0:
                    available_cloud_slot_mask_cpu[available_cloud_slot_ids_cpu] = True
                assigned_cloud_slot_ids_cpu = torch.full((cloud_ids_cpu.numel(),), -1, dtype=torch.long)
                used_cloud_slot_mask_cpu = torch.zeros(self.lm_head_u_max, dtype=torch.bool)
                desired_cloud_positions = {int(global_id): idx for idx, global_id in enumerate(cloud_ids_cpu.tolist())}
                kept_prev_cloud_mask_cpu = torch.zeros(prev_cloud_ids_cpu.numel(), dtype=torch.bool)
                for prev_idx, (global_id, slot_id) in enumerate(zip(prev_cloud_ids_cpu.tolist(), prev_cloud_slot_ids_cpu.tolist())):
                    desired_idx = desired_cloud_positions.get(int(global_id))
                    if desired_idx is None or not bool(available_cloud_slot_mask_cpu[slot_id]):
                        continue
                    if assigned_cloud_slot_ids_cpu[desired_idx].item() >= 0:
                        continue
                    assigned_cloud_slot_ids_cpu[desired_idx] = int(slot_id)
                    used_cloud_slot_mask_cpu[slot_id] = True
                    kept_prev_cloud_mask_cpu[prev_idx] = True
                remaining_cloud_ids_cpu = cloud_ids_cpu[assigned_cloud_slot_ids_cpu < 0]
                remaining_cloud_slot_ids_cpu = available_cloud_slot_ids_cpu[~used_cloud_slot_mask_cpu[available_cloud_slot_ids_cpu]]
                if remaining_cloud_ids_cpu.numel() > 0:
                    assigned_cloud_slot_ids_cpu[assigned_cloud_slot_ids_cpu < 0] = remaining_cloud_slot_ids_cpu[:remaining_cloud_ids_cpu.numel()]
                    cloud_stage_ids_cpu = remaining_cloud_ids_cpu
                    cloud_stage_slot_ids_cpu = remaining_cloud_slot_ids_cpu[:remaining_cloud_ids_cpu.numel()]
                cloud_slot_ids_cpu = assigned_cloud_slot_ids_cpu
                if prev_cloud_ids_cpu.numel() > 0:
                    cloud_writeback_ids_cpu = prev_cloud_ids_cpu[~kept_prev_cloud_mask_cpu]
                    cloud_writeback_slot_ids_cpu = prev_cloud_slot_ids_cpu[~kept_prev_cloud_mask_cpu]
            else:
                cloud_slot_ids_cpu = available_cloud_slot_ids_cpu[:cloud_ids_cpu.numel()]
                cloud_stage_ids_cpu = cloud_ids_cpu
                cloud_stage_slot_ids_cpu = cloud_slot_ids_cpu
        warm_slot_ids_cpu = cloud_slot_ids_cpu[:warm_ids_cpu.numel()] if warm_ids_cpu.numel() > 0 else self._empty_long_cpu()
        cold_slot_ids_cpu = cloud_slot_ids_cpu[warm_ids_cpu.numel():] if cold_ids_cpu.numel() > 0 else self._empty_long_cpu()
        lm_head_active_ids_cpu = torch.cat((active_ids_cpu, cloud_ids_cpu)) if cloud_ids_cpu.numel() > 0 else active_ids_cpu.clone()
        lm_head_active_slot_ids_cpu = torch.cat((active_slot_ids_cpu, cloud_slot_ids_cpu)) if cloud_slot_ids_cpu.numel() > 0 else active_slot_ids_cpu.clone()
        next_step_lm_head_ids_cpu = grad_accum_ids_cpu if grad_accum_steps > 1 else lm_head_active_ids_cpu

        assert self.fixed_slot_to_global_cpu is not None
        assert self.fixed_lm_head_slot_to_global_cpu is not None
        assert self.fixed_active_mask_cpu is not None
        assert self.fixed_logit_mask is not None
        assert self.fixed_active_vocab is not None
        assert self.fixed_cold_logit_bias is not None

        if not self._fixed_live_state:
            stage_ids_cpu = active_ids_cpu
            stage_slot_ids_cpu = active_slot_ids_cpu
            cloud_stage_ids_cpu = cloud_ids_cpu
            cloud_stage_slot_ids_cpu = cloud_slot_ids_cpu

        grad_accum_stage_ids_cpu = grad_accum_ids_cpu
        grad_accum_stage_slot_ids_cpu = torch.arange(grad_accum_ids_cpu.numel(), dtype=torch.long)
        prep_writeback_wait_ms = 0.0
        prep_prefetch_wait_ms = float(prefetched_wait_ms)
        prep_cpu_gather_ms = 0.0
        prep_h2d_enqueue_ms = 0.0
        prep_h2d_tensor_count = 0
        prep_h2d_bytes = 0
        deferred_writeback_ids_cpu = self._empty_long_cpu()
        deferred_writeback_slot_ids_cpu = self._empty_long_cpu()
        deferred_lm_head_writeback_ids_cpu = self._empty_long_cpu()
        deferred_lm_head_writeback_slot_ids_cpu = self._empty_long_cpu()
        input_stage_ids_cpu = stage_ids_cpu
        input_stage_slot_ids_cpu = stage_slot_ids_cpu
        if union_input_tables:
            input_stage_ids_cpu = self._empty_long_cpu() if preserve_resident_grads else grad_accum_ids_cpu
            input_stage_slot_ids_cpu = self._empty_long_cpu() if preserve_resident_grads else torch.arange(grad_accum_ids_cpu.numel(), dtype=torch.long)
        if self.use_cuda and grad_accum_steps > 1 and self._fixed_live_state and not self.disable_fixed_overlap_reuse:
            assert self.fixed_input_slot_to_global_cpu is not None
            prev_input_slot_ids_cpu = torch.nonzero(self.fixed_input_slot_to_global_cpu >= 0, as_tuple=False).flatten()
            if prev_input_slot_ids_cpu.numel() > 0:
                prev_input_ids_cpu = self.fixed_input_slot_to_global_cpu.index_select(0, prev_input_slot_ids_cpu)
                prev_input_global_to_row_cpu = torch.full((self.model.config.vocab_size,), -1, dtype=torch.long)
                prev_input_global_to_row_cpu[prev_input_ids_cpu] = torch.arange(prev_input_ids_cpu.numel(), dtype=torch.long)
                kept_prev_input_rows_cpu = prev_input_global_to_row_cpu.index_select(0, grad_accum_ids_cpu)
                kept_prev_input_mask_cpu = torch.zeros(prev_input_slot_ids_cpu.numel(), dtype=torch.bool)
                valid_prev_input_mask_cpu = kept_prev_input_rows_cpu >= 0
                if valid_prev_input_mask_cpu.any():
                    kept_prev_input_mask_cpu[kept_prev_input_rows_cpu[valid_prev_input_mask_cpu]] = True
                deferred_writeback_ids_cpu = prev_input_ids_cpu[~kept_prev_input_mask_cpu]
                deferred_writeback_slot_ids_cpu = prev_input_slot_ids_cpu[~kept_prev_input_mask_cpu]
            prev_lm_head_slot_ids_cpu = torch.nonzero(self.fixed_lm_head_slot_to_global_cpu >= 0, as_tuple=False).flatten()
            if prev_lm_head_slot_ids_cpu.numel() > 0:
                prev_lm_head_ids_cpu = self.fixed_lm_head_slot_to_global_cpu.index_select(0, prev_lm_head_slot_ids_cpu)
                prev_lm_head_global_to_row_cpu = torch.full((self.model.config.vocab_size,), -1, dtype=torch.long)
                prev_lm_head_global_to_row_cpu[prev_lm_head_ids_cpu] = torch.arange(prev_lm_head_ids_cpu.numel(), dtype=torch.long)
                kept_prev_lm_head_rows_cpu = prev_lm_head_global_to_row_cpu.index_select(0, grad_accum_ids_cpu)
                kept_prev_lm_head_mask_cpu = torch.zeros(prev_lm_head_slot_ids_cpu.numel(), dtype=torch.bool)
                valid_prev_lm_head_mask_cpu = kept_prev_lm_head_rows_cpu >= 0
                if valid_prev_lm_head_mask_cpu.any():
                    kept_prev_lm_head_mask_cpu[kept_prev_lm_head_rows_cpu[valid_prev_lm_head_mask_cpu]] = True
                deferred_lm_head_writeback_ids_cpu = prev_lm_head_ids_cpu[~kept_prev_lm_head_mask_cpu]
                deferred_lm_head_writeback_slot_ids_cpu = prev_lm_head_slot_ids_cpu[~kept_prev_lm_head_mask_cpu]
        if grad_accum_steps > 1 and not preserve_resident_grads and grad_accum_ids_cpu.numel() > 0 and self._fixed_live_state and not self.disable_fixed_overlap_reuse:
            assert self.fixed_input_slot_to_global_cpu is not None
            prev_input_slot_mask_cpu = self.fixed_input_slot_to_global_cpu >= 0
            prev_input_slot_ids_cpu = torch.nonzero(prev_input_slot_mask_cpu, as_tuple=False).flatten()
            if prev_input_slot_ids_cpu.numel() > 0:
                prev_input_ids_cpu = self.fixed_input_slot_to_global_cpu.index_select(0, prev_input_slot_ids_cpu)
                prev_input_global_to_slot_cpu = torch.full((self.model.config.vocab_size,), -1, dtype=torch.long)
                prev_input_global_to_slot_cpu[prev_input_ids_cpu] = prev_input_slot_ids_cpu
                reused_old_input_slot_ids_cpu = prev_input_global_to_slot_cpu.index_select(0, grad_accum_ids_cpu)
                reuse_input_mask_cpu = reused_old_input_slot_ids_cpu >= 0
                if deferred_writeback_ids_cpu.numel() > 0:
                    self._queue_fixed_rows_writeback_(
                        deferred_writeback_ids_cpu,
                        deferred_writeback_slot_ids_cpu,
                    )
                    deferred_writeback_ids_cpu = self._empty_long_cpu()
                    deferred_writeback_slot_ids_cpu = self._empty_long_cpu()
                if reuse_input_mask_cpu.any():
                    reused_old_input_slot_ids_cpu = reused_old_input_slot_ids_cpu[reuse_input_mask_cpu]
                    reused_new_input_slot_ids_cpu = torch.nonzero(reuse_input_mask_cpu, as_tuple=False).flatten()
                    reused_old_input_slot_ids_device = reused_old_input_slot_ids_cpu.to(self.device)
                    reused_new_input_slot_ids_device = reused_new_input_slot_ids_cpu.to(self.device)
                    for name in self._union_input_table_names():
                        for source in (
                            self.fixed_params[name].data,
                            self.fixed_optimizer_state[name]["exp_avg"],
                            self.fixed_optimizer_state[name]["exp_avg_sq"],
                        ):
                            source_rows = source.index_select(0, reused_old_input_slot_ids_device)
                            source.index_copy_(0, reused_new_input_slot_ids_device, source_rows)
                    missing_input_mask_cpu = ~reuse_input_mask_cpu
                    input_stage_ids_cpu = grad_accum_ids_cpu[missing_input_mask_cpu]
                    input_stage_slot_ids_cpu = torch.nonzero(missing_input_mask_cpu, as_tuple=False).flatten()
                else:
                    input_stage_ids_cpu = grad_accum_ids_cpu
                    input_stage_slot_ids_cpu = torch.arange(grad_accum_ids_cpu.numel(), dtype=torch.long)
        if grad_accum_steps > 1 and not preserve_resident_grads and grad_accum_ids_cpu.numel() > 0 and self._fixed_live_state and not self.disable_fixed_overlap_reuse:
            prev_lm_head_slot_mask_cpu = self.fixed_lm_head_slot_to_global_cpu >= 0
            prev_lm_head_slot_ids_cpu = torch.nonzero(prev_lm_head_slot_mask_cpu, as_tuple=False).flatten()
            if prev_lm_head_slot_ids_cpu.numel() > 0:
                prev_lm_head_ids_cpu = self.fixed_lm_head_slot_to_global_cpu.index_select(0, prev_lm_head_slot_ids_cpu)
                prev_lm_head_global_to_slot_cpu = torch.full((self.model.config.vocab_size,), -1, dtype=torch.long)
                prev_lm_head_global_to_slot_cpu[prev_lm_head_ids_cpu] = prev_lm_head_slot_ids_cpu
                reused_old_slot_ids_cpu = prev_lm_head_global_to_slot_cpu.index_select(0, grad_accum_ids_cpu)
                reuse_mask_cpu = reused_old_slot_ids_cpu >= 0
                # Writeback leaving lm_head tokens BEFORE the GPU→GPU reuse copy overwrites their slots.
                # Mirrors the same pattern used in Block 2 for input tables.
                if deferred_lm_head_writeback_ids_cpu.numel() > 0:
                    self._queue_fixed_lm_head_writeback_(
                        deferred_lm_head_writeback_ids_cpu,
                        deferred_lm_head_writeback_slot_ids_cpu,
                    )
                    deferred_lm_head_writeback_ids_cpu = self._empty_long_cpu()
                    deferred_lm_head_writeback_slot_ids_cpu = self._empty_long_cpu()
                if reuse_mask_cpu.any():
                    reused_old_slot_ids_cpu = reused_old_slot_ids_cpu[reuse_mask_cpu]
                    reused_new_slot_ids_cpu = torch.nonzero(reuse_mask_cpu, as_tuple=False).flatten()
                    reused_old_slot_ids_device = reused_old_slot_ids_cpu.to(self.device)
                    reused_new_slot_ids_device = reused_new_slot_ids_cpu.to(self.device)
                    for source in (
                        self.fixed_params["lm_head"].data,
                        self.fixed_optimizer_state["lm_head"]["exp_avg"],
                        self.fixed_optimizer_state["lm_head"]["exp_avg_sq"],
                    ):
                        source_rows = source.index_select(0, reused_old_slot_ids_device)
                        source.index_copy_(0, reused_new_slot_ids_device, source_rows)
                    missing_mask_cpu = ~reuse_mask_cpu
                    grad_accum_stage_ids_cpu = grad_accum_ids_cpu[missing_mask_cpu]
                    grad_accum_stage_slot_ids_cpu = torch.nonzero(missing_mask_cpu, as_tuple=False).flatten()

        required_cpu_ids = []
        if stage_ids_cpu.numel() > 0:
            required_cpu_ids.append(stage_ids_cpu)
        if cloud_stage_ids_cpu.numel() > 0:
            required_cpu_ids.append(cloud_stage_ids_cpu)
        if grad_accum_steps > 1 and not preserve_resident_grads and input_stage_ids_cpu.numel() > 0:
            required_cpu_ids.append(input_stage_ids_cpu)
        if grad_accum_steps > 1 and not preserve_resident_grads and grad_accum_stage_ids_cpu.numel() > 0:
            required_cpu_ids.append(grad_accum_stage_ids_cpu)
        if deferred_writeback_ids_cpu.numel() > 0:
            self._queue_fixed_rows_writeback_(
                deferred_writeback_ids_cpu,
                deferred_writeback_slot_ids_cpu,
            )
        if deferred_lm_head_writeback_ids_cpu.numel() > 0:
            self._queue_fixed_lm_head_writeback_(
                deferred_lm_head_writeback_ids_cpu,
                deferred_lm_head_writeback_slot_ids_cpu,
            )
        flush_t0 = time.perf_counter()
        self._flush_pending_cpu_writeback(
            torch.cat(required_cpu_ids) if required_cpu_ids else None
        )
        prep_writeback_wait_ms += (time.perf_counter() - flush_t0) * 1000.0
        if cloud_writeback_ids_cpu.numel() > 0:
            self._queue_fixed_lm_head_writeback_(cloud_writeback_ids_cpu, cloud_writeback_slot_ids_cpu)
        if preserve_resident_grads:
            if not union_input_tables:
                self._zero_fixed_grad_slots_(
                    stage_slot_ids_cpu,
                    table_names=self._union_input_table_names(),
                )
        else:
            self._clear_fixed_grads()
        for name, spec in self.table_specs.items():
            param = spec["param"]
            state = self.state[param]
            stage_ids_for_name = input_stage_ids_cpu if union_input_tables and name != "lm_head" else stage_ids_cpu
            stage_slots_for_name = input_stage_slot_ids_cpu if union_input_tables and name != "lm_head" else stage_slot_ids_cpu
            if name == "lm_head":
                if grad_accum_steps > 1:
                    if preserve_resident_grads:
                        stage_ids_for_name = self._empty_long_cpu()
                        stage_slots_for_name = self._empty_long_cpu()
                    else:
                        stage_ids_for_name = grad_accum_stage_ids_cpu
                        stage_slots_for_name = grad_accum_stage_slot_ids_cpu
                elif cloud_stage_ids_cpu.numel() > 0:
                    stage_ids_for_name = torch.cat((stage_ids_cpu, cloud_stage_ids_cpu)) if stage_ids_cpu.numel() > 0 else cloud_stage_ids_cpu
                    stage_slots_for_name = torch.cat((stage_slot_ids_cpu, cloud_stage_slot_ids_cpu)) if stage_slot_ids_cpu.numel() > 0 else cloud_stage_slot_ids_cpu
            if stage_ids_for_name.numel() == 0:
                continue
            stage_slot_ids_device = stage_slots_for_name.to(self.device)
            stage_suffix = f"fixed:{grad_accum_micro_step}:{name}"
            prefetched_entry = None if prefetched_stage is None else prefetched_stage.get(name)
            if prefetched_entry is not None:
                if torch.equal(prefetched_entry["ids_cpu"], stage_ids_for_name):
                    rows_gpu, rows_enqueue_ms, rows_bytes = self._stage_prefetched_cpu_tensor_to_device(f"{stage_suffix}:param", prefetched_entry["param"])
                    exp_avg_gpu, exp_avg_enqueue_ms, exp_avg_bytes = self._stage_prefetched_cpu_tensor_to_device(f"{stage_suffix}:exp_avg", prefetched_entry["exp_avg"])
                    exp_avg_sq_gpu, exp_avg_sq_enqueue_ms, exp_avg_sq_bytes = self._stage_prefetched_cpu_tensor_to_device(f"{stage_suffix}:exp_avg_sq", prefetched_entry["exp_avg_sq"])
                    prep_h2d_enqueue_ms += rows_enqueue_ms + exp_avg_enqueue_ms + exp_avg_sq_enqueue_ms
                    prep_h2d_tensor_count += 3
                    prep_h2d_bytes += rows_bytes + exp_avg_bytes + exp_avg_sq_bytes
                else:
                    rows_gpu, rows_gather_ms, rows_enqueue_ms, rows_bytes = self._stage_mixed_prefetched_cpu_tensor_to_device(
                        f"{stage_suffix}:param",
                        param,
                        stage_ids_for_name,
                        prefetched_entry["param"],
                        prefetched_entry["global_to_local_cpu"],
                    )
                    exp_avg_gpu, exp_avg_gather_ms, exp_avg_enqueue_ms, exp_avg_bytes = self._stage_mixed_prefetched_cpu_tensor_to_device(
                        f"{stage_suffix}:exp_avg",
                        state["exp_avg"],
                        stage_ids_for_name,
                        prefetched_entry["exp_avg"],
                        prefetched_entry["global_to_local_cpu"],
                    )
                    exp_avg_sq_gpu, exp_avg_sq_gather_ms, exp_avg_sq_enqueue_ms, exp_avg_sq_bytes = self._stage_mixed_prefetched_cpu_tensor_to_device(
                        f"{stage_suffix}:exp_avg_sq",
                        state["exp_avg_sq"],
                        stage_ids_for_name,
                        prefetched_entry["exp_avg_sq"],
                        prefetched_entry["global_to_local_cpu"],
                    )
                    prep_cpu_gather_ms += rows_gather_ms + exp_avg_gather_ms + exp_avg_sq_gather_ms
                    prep_h2d_enqueue_ms += rows_enqueue_ms + exp_avg_enqueue_ms + exp_avg_sq_enqueue_ms
                    prep_h2d_tensor_count += 3
                    prep_h2d_bytes += rows_bytes + exp_avg_bytes + exp_avg_sq_bytes
            else:
                rows_gpu, rows_gather_ms, rows_enqueue_ms, rows_bytes = self._stage_index_select_to_device(f"{stage_suffix}:param", param, stage_ids_for_name)
                exp_avg_gpu, exp_avg_gather_ms, exp_avg_enqueue_ms, exp_avg_bytes = self._stage_index_select_to_device(f"{stage_suffix}:exp_avg", state["exp_avg"], stage_ids_for_name)
                exp_avg_sq_gpu, exp_avg_sq_gather_ms, exp_avg_sq_enqueue_ms, exp_avg_sq_bytes = self._stage_index_select_to_device(f"{stage_suffix}:exp_avg_sq", state["exp_avg_sq"], stage_ids_for_name)
                prep_cpu_gather_ms += rows_gather_ms + exp_avg_gather_ms + exp_avg_sq_gather_ms
                prep_h2d_enqueue_ms += rows_enqueue_ms + exp_avg_enqueue_ms + exp_avg_sq_enqueue_ms
                prep_h2d_tensor_count += 3
                prep_h2d_bytes += rows_bytes + exp_avg_bytes + exp_avg_sq_bytes
            self.fixed_params[name].data.index_copy_(0, stage_slot_ids_device, rows_gpu)
            self.fixed_optimizer_state[name]["exp_avg"].index_copy_(0, stage_slot_ids_device, exp_avg_gpu)
            self.fixed_optimizer_state[name]["exp_avg_sq"].index_copy_(0, stage_slot_ids_device, exp_avg_sq_gpu)

        if cold_row_decay > 0.0 and (grad_accum_steps == 1 or not preserve_resident_grads):
            self._apply_fixed_lm_head_cold_row_decay_(next_step_lm_head_ids_cpu, cold_row_decay)

        self.fixed_slot_to_global_cpu.copy_(slot_to_global_cpu)
        assert self.fixed_input_slot_to_global_cpu is not None
        self.fixed_input_slot_to_global_cpu.fill_(-1)
        if grad_accum_steps > 1:
            if grad_accum_ids_cpu.numel() > 0:
                self.fixed_input_slot_to_global_cpu[:grad_accum_ids_cpu.numel()].copy_(grad_accum_ids_cpu)
        else:
            self.fixed_input_slot_to_global_cpu[:self.fixed_u_max].copy_(slot_to_global_cpu)
        if grad_accum_steps > 1:
            if not preserve_resident_grads:
                self.fixed_lm_head_slot_to_global_cpu.fill_(-1)
                self.fixed_lm_head_slot_to_global_cpu[:grad_accum_ids_cpu.numel()].copy_(grad_accum_ids_cpu)
        else:
            self.fixed_lm_head_slot_to_global_cpu.fill_(-1)
            self.fixed_lm_head_slot_to_global_cpu[:self.fixed_u_max].copy_(slot_to_global_cpu)
        if grad_accum_steps == 1 and cloud_ids_cpu.numel() > 0:
            self.fixed_lm_head_slot_to_global_cpu.index_copy_(0, cloud_slot_ids_cpu, cloud_ids_cpu)
        self.fixed_active_mask_cpu.copy_(active_mask_cpu)
        use_logit_mask = (grad_accum_steps > 1) or (lm_head_active_ids_cpu.numel() < self.lm_head_u_max)
        if use_logit_mask:
            if grad_accum_steps > 1:
                if not preserve_resident_grads:
                    self.fixed_logit_mask.zero_()
                    if grad_accum_ids_cpu.numel() > 0:
                        self.fixed_logit_mask[:grad_accum_ids_cpu.numel()].fill_(True)
            else:
                self.fixed_logit_mask.zero_()
                self.fixed_logit_mask[:self.fixed_u_max].copy_(active_mask_cpu.to(self.device, non_blocking=self.use_cuda))
            if grad_accum_steps == 1 and cloud_slot_ids_cpu.numel() > 0:
                cloud_slot_ids_device = cloud_slot_ids_cpu.to(self.device)
                self.fixed_logit_mask.index_fill_(0, cloud_slot_ids_device, True)
        use_cold_logit_bias = cold_bias_scale > 0.0
        if use_cold_logit_bias:
            if grad_accum_steps > 1:
                self.fixed_cold_logit_bias.zero_()
                if current_cold_logit_bias_cpu.numel() > 0:
                    self.fixed_cold_logit_bias[:current_cold_logit_bias_cpu.numel()].copy_(
                        current_cold_logit_bias_cpu.to(self.device, non_blocking=self.use_cuda)
                    )
            else:
                self.fixed_cold_logit_bias.zero_()
                if active_slot_ids_cpu.numel() > 0:
                    bias_slot_ids_device = active_slot_ids_cpu.to(self.device)
                    self.fixed_cold_logit_bias.index_copy_(0, bias_slot_ids_device, current_cold_logit_bias_cpu.to(self.device, non_blocking=self.use_cuda))
        self._fixed_live_state = True

        union_inputs = None
        if union_inputs_cpu_local is not None and grad_accum_steps > 1:
            union_inputs = union_inputs_cpu_local.detach().to(self.device, non_blocking=self.use_cuda)
        union_targets = None
        if union_targets_cpu_local is not None and grad_accum_steps > 1:
            union_targets = union_targets_cpu_local.detach().to(self.device, non_blocking=self.use_cuda)

        step_active_vocab = {
            **self.fixed_active_vocab,
            "value_embeds": self.fixed_active_vocab["value_embeds"],
            "lm_head": self.fixed_active_vocab["lm_head"],
        }
        if grad_accum_steps > 1:
            union_count = int(grad_accum_ids_cpu.numel())
            wte_view = self.fixed_params["wte"][:union_count]
            lm_head_view = self.fixed_params["lm_head"][:union_count]
            if not self.use_cuda:
                wte_view.retain_grad()
                lm_head_view.retain_grad()
                if self.fixed_params["wte"].grad is not None:
                    wte_view.grad = self.fixed_params["wte"].grad[:union_count]
                if self.fixed_params["lm_head"].grad is not None:
                    lm_head_view.grad = self.fixed_params["lm_head"].grad[:union_count]
            value_embed_views = {}
            for layer_name in self.fixed_active_vocab["value_embeds"]:
                value_view = self.fixed_params[f"value_embeds.{layer_name}"][:union_count]
                if not self.use_cuda:
                    value_view.retain_grad()
                    base_grad = self.fixed_params[f"value_embeds.{layer_name}"].grad
                    if base_grad is not None:
                        value_view.grad = base_grad[:union_count]
                value_embed_views[layer_name] = value_view
            step_active_vocab["wte"] = wte_view
            step_active_vocab["value_embeds"] = value_embed_views
            step_active_vocab["lm_head"] = lm_head_view
            step_active_vocab.pop("logit_mask", None)
            if use_cold_logit_bias:
                step_active_vocab["cold_logit_bias"] = self.fixed_cold_logit_bias[:union_count]
            else:
                step_active_vocab.pop("cold_logit_bias", None)
        else:
            if use_logit_mask:
                step_active_vocab["logit_mask"] = self.fixed_logit_mask
            if use_cold_logit_bias:
                step_active_vocab["cold_logit_bias"] = self.fixed_cold_logit_bias
        step_optimizer_state = {
            **self.fixed_optimizer_state,
        }

        return DynamicVocabStep(
            active_ids_cpu=active_ids_cpu,
            active_vocab=step_active_vocab,
            optimizer_state=step_optimizer_state,
            union_inputs=union_inputs,
            union_targets=union_targets,
            unique_count=int(grad_accum_ids_cpu.numel()) if grad_accum_steps > 1 else lm_head_active_ids_cpu.numel(),
            live_count=int(grad_accum_ids_cpu.numel()) if grad_accum_steps > 1 else lm_head_active_ids_cpu.numel(),
            step_u_count=active_ids_cpu.numel(),
            u_capacity=self.lm_head_u_max,
            stage_count=(
                stage_ids_cpu.numel() +
                cloud_stage_ids_cpu.numel() +
                (grad_accum_stage_ids_cpu.numel() if grad_accum_steps > 1 and not preserve_resident_grads else 0)
            ),
            cloud_residual_capacity=int(step_meta.get("cloud_residual_capacity", 0)),
            warm_budget_target=int(step_meta.get("warm_budget_target", 0)),
            cold_budget_target=int(step_meta.get("cold_budget_target", 0)),
            warm_candidate_count=int(step_meta.get("warm_candidate_count", 0)),
            cloud_plan_ms=float(step_meta.get("cloud_plan_ms", 0.0)),
            cloud_hidden_query_ms=float(step_meta.get("cloud_hidden_query_ms", 0.0)),
            cloud_selection_ms=float(step_meta.get("cloud_selection_ms", 0.0)),
            active_slot_ids_cpu=active_slot_ids_cpu,
            active_mask_cpu=active_mask_cpu,
            slot_to_global_cpu=slot_to_global_cpu,
            stage_ids_cpu=stage_ids_cpu,
            stage_slot_ids_cpu=stage_slot_ids_cpu,
            writeback_ids_cpu=writeback_ids_cpu,
            writeback_slot_ids_cpu=writeback_slot_ids_cpu,
            grad_accum_ids_cpu=grad_accum_ids_cpu,
            lm_head_active_ids_cpu=grad_accum_ids_cpu if grad_accum_steps > 1 else lm_head_active_ids_cpu,
            lm_head_active_slot_ids_cpu=torch.arange(grad_accum_ids_cpu.numel(), dtype=torch.long) if grad_accum_steps > 1 else lm_head_active_slot_ids_cpu,
            warm_ids_cpu=warm_ids_cpu,
            warm_slot_ids_cpu=warm_slot_ids_cpu,
            cold_ids_cpu=cold_ids_cpu,
            cold_slot_ids_cpu=cold_slot_ids_cpu,
            cloud_stage_ids_cpu=cloud_stage_ids_cpu,
            cloud_stage_slot_ids_cpu=cloud_stage_slot_ids_cpu,
            cloud_writeback_ids_cpu=cloud_writeback_ids_cpu,
            cloud_writeback_slot_ids_cpu=cloud_writeback_slot_ids_cpu,
            grad_accum_steps=grad_accum_steps,
            grad_accum_micro_step=grad_accum_micro_step,
            is_grad_accum_boundary=is_grad_accum_boundary,
            is_last_step=is_last_step,
            fixed_u_mode=True,
            cold_bias_clamped_count=cold_bias_clamped_count,
            cold_bias_abs_max=cold_bias_abs_max,
            hot_activation_counts_cpu=hot_activation_counts_cpu,
            prep_writeback_wait_ms=prep_writeback_wait_ms,
            prep_prefetch_wait_ms=prep_prefetch_wait_ms,
            prep_cpu_gather_ms=prep_cpu_gather_ms,
            prep_h2d_enqueue_ms=prep_h2d_enqueue_ms,
            prep_h2d_tensor_count=prep_h2d_tensor_count,
            prep_h2d_bytes=prep_h2d_bytes,
            prep_prefetch_hit=prefetched_hit,
        )

    def prepare_step(
        self,
        active_ids_cpu,
        cold_bias_scale: float = 0.0,
        cold_row_decay: float = 0.0,
        cold_bias_tokens_per_step: Optional[int] = None,
    ) -> DynamicVocabStep:
        if isinstance(active_ids_cpu, dict):
            prefetched_stage = None
            prefetch_wait_ms = 0.0
            prefetch_hit = 0
            if self.use_cuda and self._pending_stage_prefetch_future is not None and self._pending_stage_prefetch_key == id(active_ids_cpu):
                prefetch_wait_t0 = time.perf_counter()
                prefetched_stage = self._pending_stage_prefetch_future.result()
                prefetch_wait_ms = (time.perf_counter() - prefetch_wait_t0) * 1000.0
                prefetch_hit = 1
                self._pending_stage_prefetch_future = None
                self._pending_stage_prefetch_key = None
            return self._prepare_fixed_step(
                active_ids_cpu,
                cold_bias_scale=cold_bias_scale,
                cold_row_decay=cold_row_decay,
                cold_bias_tokens_per_step=cold_bias_tokens_per_step,
                prefetched_stage=prefetched_stage,
                prefetched_wait_ms=prefetch_wait_ms,
                prefetched_hit=prefetch_hit,
            )
        return self._prepare_dynamic_step(
            active_ids_cpu,
            cold_bias_scale=cold_bias_scale,
            cold_row_decay=cold_row_decay,
            cold_bias_tokens_per_step=cold_bias_tokens_per_step,
        )

    def _adamw_update_(
        self,
        param_name: str,
        active_param: nn.Parameter,
        active_state: dict,
        step_value: int | torch.Tensor,
        slot_ids_cpu: Optional[torch.Tensor] = None,
        hot_activation_counts_cpu: Optional[torch.Tensor] = None,
        lr_override: Optional[float] = None,
    ) -> bool:
        grad = active_param.grad
        if grad is None:
            return False
        if slot_ids_cpu is not None:
            slot_ids_cpu = slot_ids_cpu.detach().to(device="cpu", dtype=torch.long)
            if slot_ids_cpu.numel() == 0:
                return False
        if isinstance(step_value, torch.Tensor) and step_value.numel() == 0:
            return False
        self._adamw_update_with_step_(
            param_name,
            active_param,
            active_state,
            step_value,
            slot_ids_cpu=slot_ids_cpu,
            hot_activation_counts_cpu=hot_activation_counts_cpu,
            lr_override=lr_override,
        )
        return True

    def _adamw_update_with_step_(
        self,
        param_name: str,
        active_param: nn.Parameter,
        active_state: dict,
        step_value: int | torch.Tensor,
        slot_ids_cpu: Optional[torch.Tensor] = None,
        hot_activation_counts_cpu: Optional[torch.Tensor] = None,
        lr_override: Optional[float] = None,
    ) -> None:
        grad = active_param.grad
        if grad is None:
            return
        spec = self.table_specs[param_name]
        exp_avg = active_state["exp_avg"]
        exp_avg_sq = active_state["exp_avg_sq"]
        base_lr = float(spec["lr"] if lr_override is None else lr_override)
        row_lr = None
        if param_name == "lm_head":
            row_lr = self._get_lm_head_row_lr(base_lr, hot_activation_counts_cpu, active_param.device, active_param.dtype)
        slot_ids = None
        if slot_ids_cpu is None:
            param_rows = active_param
            grad_rows = grad
            exp_avg_rows = exp_avg
            exp_avg_sq_rows = exp_avg_sq
        else:
            slot_ids_cpu = slot_ids_cpu.detach().to(device="cpu", dtype=torch.long)
            if slot_ids_cpu.numel() == 0:
                return
            slot_ids = slot_ids_cpu.to(active_param.device)
            param_rows = active_param.index_select(0, slot_ids)
            grad_rows = grad.index_select(0, slot_ids)
            exp_avg_rows = exp_avg.index_select(0, slot_ids)
            exp_avg_sq_rows = exp_avg_sq.index_select(0, slot_ids)
        if self.weight_decay != 0.0:
            param_rows.mul_(1 - base_lr * self.weight_decay)
        exp_avg_rows.lerp_(grad_rows, 1 - self.beta1)
        exp_avg_sq_rows.lerp_(grad_rows.square(), 1 - self.beta2)

        if isinstance(step_value, torch.Tensor):
            step_values = step_value.detach().to(device=param_rows.device, dtype=torch.float32)
            if step_values.dim() != 1 or step_values.numel() != param_rows.size(0):
                raise ValueError(
                    f"Sparse AdamW step values must be a 1D tensor with one entry per row, got shape {tuple(step_values.shape)} for {param_rows.size(0)} rows"
                )
            if step_values.numel() > 0 and torch.equal(step_values, step_values[:1].expand_as(step_values)):
                scalar_step_value = int(step_values[0].item())
                bias1 = 1 - self.beta1 ** scalar_step_value
                bias2 = 1 - self.beta2 ** scalar_step_value
                denom = (exp_avg_sq_rows / bias2).sqrt().add_(self.eps)
                if row_lr is not None:
                    row_lr = row_lr.view((-1,) + (1,) * (param_rows.dim() - 1))
                    param_rows.add_((exp_avg_rows / denom) * row_lr, alpha=-1.0 / bias1)
                else:
                    step_size = base_lr / bias1
                    param_rows.addcdiv_(exp_avg_rows, denom, value=-step_size)
            else:
                bias1 = 1 - torch.pow(torch.full_like(step_values, self.beta1), step_values)
                bias2 = 1 - torch.pow(torch.full_like(step_values, self.beta2), step_values)
                view_shape = (-1,) + (1,) * (param_rows.dim() - 1)
                denom = (exp_avg_sq_rows / bias2.view(view_shape)).sqrt().add_(self.eps)
                if row_lr is not None:
                    scale = row_lr / bias1.to(dtype=row_lr.dtype)
                else:
                    scale = torch.full_like(bias1, base_lr, dtype=torch.float32) / bias1
                param_rows.add_((exp_avg_rows / denom) * scale.to(dtype=param_rows.dtype).view(view_shape), alpha=-1.0)
        else:
            bias1 = 1 - self.beta1 ** step_value
            bias2 = 1 - self.beta2 ** step_value
            denom = (exp_avg_sq_rows / bias2).sqrt().add_(self.eps)
            if row_lr is not None:
                row_lr = row_lr.view((-1,) + (1,) * (param_rows.dim() - 1))
                param_rows.add_((exp_avg_rows / denom) * row_lr, alpha=-1.0 / bias1)
            else:
                step_size = base_lr / bias1
                param_rows.addcdiv_(exp_avg_rows, denom, value=-step_size)

        if slot_ids is not None:
            active_param.index_copy_(0, slot_ids, param_rows)
            exp_avg.index_copy_(0, slot_ids, exp_avg_rows)
            exp_avg_sq.index_copy_(0, slot_ids, exp_avg_sq_rows)

    def _materialize_manual_grad_accum_view_grads_(self, step_ctx: DynamicVocabStep) -> None:
        if self.use_cuda:
            return
        if step_ctx.active_vocab is None or step_ctx.grad_accum_ids_cpu is None:
            return
        union_count = int(step_ctx.grad_accum_ids_cpu.numel())
        if union_count <= 0:
            return

        def copy_view_grad(param_name: str, exposed_tensor: torch.Tensor) -> None:
            grad = getattr(exposed_tensor, "grad", None)
            if grad is None:
                return
            target_param = self.fixed_params[param_name]
            if target_param.grad is not None:
                return
            target_grad = torch.zeros_like(target_param)
            target_grad[:union_count].copy_(grad.to(dtype=target_grad.dtype))
            target_param.grad = target_grad

        copy_view_grad("wte", step_ctx.active_vocab["wte"])
        copy_view_grad("lm_head", step_ctx.active_vocab["lm_head"])
        for layer_name, exposed_tensor in step_ctx.active_vocab["value_embeds"].items():
            copy_view_grad(f"value_embeds.{layer_name}", exposed_tensor)

    @torch.no_grad()
    def accumulate_gradients(self, step_ctx: DynamicVocabStep) -> DynamicVocabStep:
        t_start = time.perf_counter()
        assert step_ctx.fixed_u_mode, "Sparse grad accumulation currently supports fixed-U mode only"
        assert step_ctx.active_slot_ids_cpu is not None
        assert step_ctx.active_ids_cpu is not None
        if step_ctx.grad_accum_ids_cpu is None:
            raise ValueError("Sparse grad accumulation requires grad_accum_ids_cpu metadata")
        self._materialize_manual_grad_accum_view_grads_(step_ctx)

        start_ms = 0.0
        if (not self._grad_accum_live) or self._grad_accum_ids_cpu is None or not torch.equal(self._grad_accum_ids_cpu, step_ctx.grad_accum_ids_cpu):
            t_window_start = time.perf_counter()
            self._start_grad_accum_window(step_ctx.grad_accum_ids_cpu)
            start_ms = (time.perf_counter() - t_window_start) * 1000.0

        assert self._grad_accum_global_to_local_cpu is not None
        assert self._grad_accum_buffers is not None
        t_flush_start = time.perf_counter()
        self._flush_pending_grad_accum_transfers(wait=False)
        flush_ms = (time.perf_counter() - t_flush_start) * 1000.0

        t_queue_start = time.perf_counter()
        queued_count = 0
        if not step_ctx.is_grad_accum_boundary:
            queued_count = 0
        queue_ms = (time.perf_counter() - t_queue_start) * 1000.0

        step_ctx.active_vocab = None
        step_ctx.optimizer_state = None
        self._grad_accum_stage_count += int(step_ctx.stage_count)
        step_ctx.unique_count = int(step_ctx.grad_accum_ids_cpu.numel())
        step_ctx.live_count = int(step_ctx.active_ids_cpu.numel())
        step_ctx.grad_accum_start_ms = start_ms
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
        grad_accum_hot_activation_counts_cpu = self._grad_accum_hot_activation_counts_cpu
        live_count = int(grad_accum_ids_cpu.numel())
        live_slot_ids_cpu = torch.empty(0, dtype=torch.long)
        live_global_ids_cpu = torch.empty(0, dtype=torch.long)
        live_union_row_ids_cpu = torch.empty(0, dtype=torch.long)
        live_union_mask_cpu = torch.zeros(grad_accum_count, dtype=torch.bool)
        live_lm_head_slot_ids_cpu = torch.empty(0, dtype=torch.long)
        live_lm_head_global_ids_cpu = torch.empty(0, dtype=torch.long)
        live_lm_head_union_row_ids_cpu = torch.empty(0, dtype=torch.long)
        if self.fixed_u_mode and self._fixed_live_state:
            assert self.fixed_input_slot_to_global_cpu is not None
            assert self.fixed_lm_head_slot_to_global_cpu is not None
            assert self._grad_accum_global_to_local_cpu is not None
            live_slot_ids_cpu = torch.nonzero(self.fixed_input_slot_to_global_cpu >= 0, as_tuple=False).flatten()
            live_count = int(live_slot_ids_cpu.numel())
            if live_slot_ids_cpu.numel() > 0:
                live_global_ids_cpu = self.fixed_input_slot_to_global_cpu[live_slot_ids_cpu]
                live_union_row_ids_cpu = self._grad_accum_global_to_local_cpu[live_global_ids_cpu]
                if (live_union_row_ids_cpu < 0).any():
                    raise ValueError("Sparse grad accumulation map is missing live fixed-U rows")
                live_union_mask_cpu[live_union_row_ids_cpu] = True
            live_lm_head_slot_ids_cpu = torch.nonzero(self.fixed_lm_head_slot_to_global_cpu >= 0, as_tuple=False).flatten()
            if live_lm_head_slot_ids_cpu.numel() > 0:
                live_lm_head_global_ids_cpu = self.fixed_lm_head_slot_to_global_cpu[live_lm_head_slot_ids_cpu]
                live_lm_head_union_row_ids_cpu = self._grad_accum_global_to_local_cpu[live_lm_head_global_ids_cpu]
                if (live_lm_head_union_row_ids_cpu < 0).any():
                    raise ValueError("Sparse grad accumulation map is missing live lm_head rows")
                live_union_mask_cpu[live_lm_head_union_row_ids_cpu] = True

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

        live_reentered_slot_ids_cpu = torch.empty(0, dtype=torch.long)
        live_reentered_union_row_ids_cpu = torch.empty(0, dtype=torch.long)
        if self._grad_accum_cached_union_mask_cpu is not None and live_union_row_ids_cpu.numel() > 0:
            live_reentered_mask_cpu = self._grad_accum_cached_union_mask_cpu[live_union_row_ids_cpu]
            if live_reentered_mask_cpu.any():
                live_reentered_slot_ids_cpu = live_slot_ids_cpu[live_reentered_mask_cpu]
                live_reentered_union_row_ids_cpu = live_union_row_ids_cpu[live_reentered_mask_cpu]
                live_reentered_slot_ids_device = live_reentered_slot_ids_cpu.to(self.device)
                live_reentered_union_row_ids_device = live_reentered_union_row_ids_cpu.to(self.device)
                for name in self.table_specs:
                    grad = self.fixed_params[name].grad
                    if grad is None:
                        grad = torch.zeros_like(self.fixed_params[name])
                        self.fixed_params[name].grad = grad
                    buffered_grad_rows = self._grad_accum_buffers[name].index_select(0, live_reentered_union_row_ids_device)
                    target_slot_ids_device = live_reentered_slot_ids_device
                    if name == "lm_head":
                        target_slot_ids_device = live_reentered_union_row_ids_device
                    grad.index_add_(0, target_slot_ids_device, buffered_grad_rows.to(dtype=grad.dtype))

        t_apply_start = time.perf_counter()
        for name in self.table_specs:
            spec = self.table_specs[name]
            state = self.state[spec["param"]]
            live_slot_ids_for_name_cpu = live_slot_ids_cpu
            if name == "lm_head":
                live_slot_ids_for_name_cpu = live_lm_head_slot_ids_cpu
            has_live_grad = live_slot_ids_for_name_cpu.numel() > 0 and self.fixed_params[name].grad is not None
            has_cached_non_live_grad = any(name in chunk["tables"] for chunk in self._grad_accum_non_live_chunks)
            has_staged_non_live_grad = name in non_live_params and staged_non_live_union_row_ids_device is not None
            if not has_live_grad and not has_cached_non_live_grad and not has_staged_non_live_grad:
                continue
            state["step"] += 1
            if has_live_grad:
                live_step_values_cpu = self._get_token_event_step_values_cpu(
                    live_lm_head_global_ids_cpu if name == "lm_head" else live_global_ids_cpu
                )
                live_hot_counts_cpu = None
                if name == "lm_head" and grad_accum_hot_activation_counts_cpu is not None and live_lm_head_union_row_ids_cpu.numel() > 0:
                    live_hot_counts_cpu = grad_accum_hot_activation_counts_cpu.index_select(0, live_lm_head_union_row_ids_cpu)
                self._adamw_update_with_step_(
                    name,
                    self.fixed_params[name],
                    self.fixed_optimizer_state[name],
                    live_step_values_cpu,
                    slot_ids_cpu=live_slot_ids_for_name_cpu,
                    hot_activation_counts_cpu=live_hot_counts_cpu,
                )
            if has_cached_non_live_grad:
                for chunk in self._grad_accum_non_live_chunks:
                    if name not in chunk["tables"]:
                        continue
                    chunk_param = chunk["tables"][name]["param"]
                    chunk_param.grad = self._grad_accum_buffers[name].index_select(0, chunk["union_row_ids_device"])
                    chunk_step_values_cpu = self._get_token_event_step_values_cpu(chunk["global_ids_cpu"])
                    chunk_hot_counts_cpu = None
                    if name == "lm_head" and grad_accum_hot_activation_counts_cpu is not None:
                        chunk_hot_counts_cpu = grad_accum_hot_activation_counts_cpu.index_select(0, chunk["union_row_ids_cpu"])
                    self._adamw_update_with_step_(
                        name,
                        chunk_param,
                        {
                            "exp_avg": chunk["tables"][name]["exp_avg"],
                            "exp_avg_sq": chunk["tables"][name]["exp_avg_sq"],
                        },
                        chunk_step_values_cpu,
                        hot_activation_counts_cpu=chunk_hot_counts_cpu,
                    )
            if has_staged_non_live_grad:
                non_live_params[name].grad = self._grad_accum_buffers[name].index_select(0, staged_non_live_union_row_ids_device)
                staged_step_values_cpu = self._get_token_event_step_values_cpu(staged_non_live_grad_accum_ids_cpu)
                staged_hot_counts_cpu = None
                if name == "lm_head" and grad_accum_hot_activation_counts_cpu is not None:
                    staged_hot_counts_cpu = grad_accum_hot_activation_counts_cpu.index_select(0, staged_non_live_union_row_ids_cpu)
                self._adamw_update_with_step_(
                    name,
                    non_live_params[name],
                    non_live_optimizer_state[name],
                    staged_step_values_cpu,
                    hot_activation_counts_cpu=staged_hot_counts_cpu,
                )
        apply_ms = (time.perf_counter() - t_apply_start) * 1000.0
        self._increment_token_event_step_counts_(grad_accum_ids_cpu)

        restore_ms = 0.0

        t_writeback_start = time.perf_counter()
        d2h_launch_ms = 0.0
        d2h_sync_ms = 0.0
        cpu_writeback_ms = 0.0
        d2h_segment_count = 0
        d2h_row_count = 0
        d2h_bytes = 0
        live_writeback_ids_cpu = torch.empty(0, dtype=torch.long)
        if not self.use_cuda and (live_global_ids_cpu.numel() > 0 or live_lm_head_global_ids_cpu.numel() > 0):
            live_writeback_ids_cpu = torch.cat(
                tuple(
                    ids for ids in (live_global_ids_cpu, live_lm_head_global_ids_cpu)
                    if ids.numel() > 0
                )
            )
        writeback_id_chunks = [chunk["global_ids_cpu"] for chunk in self._grad_accum_non_live_chunks]
        if live_writeback_ids_cpu.numel() > 0:
            writeback_id_chunks.insert(0, live_writeback_ids_cpu)
        if staged_non_live_grad_accum_ids_cpu.numel() > 0:
            writeback_id_chunks.append(staged_non_live_grad_accum_ids_cpu)
        if writeback_id_chunks:
            writeback_ids_cpu = torch.unique(torch.cat(writeback_id_chunks), sorted=False)
        else:
            writeback_ids_cpu = torch.empty(0, dtype=torch.long)
        if writeback_ids_cpu.numel() > 0:
            self._flush_pending_cpu_writeback(writeback_ids_cpu)
            t_d2h_launch_start = time.perf_counter()
            writeback_sources = {}

            def queue_writeback_source(
                name: str,
                param: torch.Tensor,
                state: dict,
                segment_ids_cpu: torch.Tensor,
                row_source: torch.Tensor,
                exp_avg_source: torch.Tensor,
                exp_avg_sq_source: torch.Tensor,
            ) -> None:
                batch = writeback_sources.get(name)
                if batch is None:
                    batch = {
                        "param": param,
                        "state": state,
                        "ids": [],
                        "rows": [],
                        "exp_avg": [],
                        "exp_avg_sq": [],
                    }
                    writeback_sources[name] = batch
                batch["ids"].append(segment_ids_cpu)
                batch["rows"].append(row_source)
                batch["exp_avg"].append(exp_avg_source)
                batch["exp_avg_sq"].append(exp_avg_sq_source)

            writeback_segments = []
            if not self.use_cuda and live_slot_ids_cpu.numel() > 0:
                live_slot_ids_device = live_slot_ids_cpu.to(self.device)
                for name in self.table_specs:
                    if name == "lm_head":
                        continue
                    param = self.table_specs[name]["param"]
                    state = self.state[param]
                    row_source = self.fixed_params[name].detach().index_select(0, live_slot_ids_device)
                    exp_avg_source = self.fixed_optimizer_state[name]["exp_avg"].detach().index_select(0, live_slot_ids_device)
                    exp_avg_sq_source = self.fixed_optimizer_state[name]["exp_avg_sq"].detach().index_select(0, live_slot_ids_device)
                    queue_writeback_source(name, param, state, live_global_ids_cpu, row_source, exp_avg_source, exp_avg_sq_source)
            if not self.use_cuda and live_lm_head_slot_ids_cpu.numel() > 0:
                live_lm_head_slot_ids_device = live_lm_head_slot_ids_cpu.to(self.device)
                param = self.table_specs["lm_head"]["param"]
                state = self.state[param]
                row_source = self.fixed_params["lm_head"].detach().index_select(0, live_lm_head_slot_ids_device)
                exp_avg_source = self.fixed_optimizer_state["lm_head"]["exp_avg"].detach().index_select(0, live_lm_head_slot_ids_device)
                exp_avg_sq_source = self.fixed_optimizer_state["lm_head"]["exp_avg_sq"].detach().index_select(0, live_lm_head_slot_ids_device)
                queue_writeback_source("lm_head", param, state, live_lm_head_global_ids_cpu, row_source, exp_avg_source, exp_avg_sq_source)
            for name, spec in self.table_specs.items():
                param = spec["param"]
                state = self.state[param]
                for chunk in self._grad_accum_non_live_chunks:
                    if name not in chunk["tables"]:
                        continue
                    row_source = chunk["tables"][name]["param"].detach()
                    exp_avg_source = chunk["tables"][name]["exp_avg"].detach()
                    exp_avg_sq_source = chunk["tables"][name]["exp_avg_sq"].detach()
                    queue_writeback_source(name, param, state, chunk["global_ids_cpu"], row_source, exp_avg_source, exp_avg_sq_source)
                if name in non_live_params:
                    row_source = non_live_params[name].detach()
                    exp_avg_source = non_live_optimizer_state[name]["exp_avg"].detach()
                    exp_avg_sq_source = non_live_optimizer_state[name]["exp_avg_sq"].detach()
                    queue_writeback_source(name, param, state, staged_non_live_grad_accum_ids_cpu, row_source, exp_avg_source, exp_avg_sq_source)
            for name, batch in writeback_sources.items():
                segment_ids_cpu = batch["ids"][0] if len(batch["ids"]) == 1 else torch.cat(batch["ids"])
                row_source = batch["rows"][0] if len(batch["rows"]) == 1 else torch.cat(batch["rows"], dim=0)
                exp_avg_source = batch["exp_avg"][0] if len(batch["exp_avg"]) == 1 else torch.cat(batch["exp_avg"], dim=0)
                exp_avg_sq_source = batch["exp_avg_sq"][0] if len(batch["exp_avg_sq"]) == 1 else torch.cat(batch["exp_avg_sq"], dim=0)
                row_buffer = self._get_cpu_receive_buffer(
                    f"rows:accum:{name}:merged",
                    tuple(row_source.shape),
                    row_source.dtype,
                    block_reuse_while_pending=True,
                )
                exp_avg_buffer = self._get_cpu_receive_buffer(
                    f"exp_avg:accum:{name}:merged",
                    tuple(exp_avg_source.shape),
                    exp_avg_source.dtype,
                    block_reuse_while_pending=True,
                )
                exp_avg_sq_buffer = self._get_cpu_receive_buffer(
                    f"exp_avg_sq:accum:{name}:merged",
                    tuple(exp_avg_sq_source.shape),
                    exp_avg_sq_source.dtype,
                    block_reuse_while_pending=True,
                )
                row_buffer.copy_(row_source, non_blocking=self.use_cuda)
                exp_avg_buffer.copy_(exp_avg_source, non_blocking=self.use_cuda)
                exp_avg_sq_buffer.copy_(exp_avg_sq_source, non_blocking=self.use_cuda)
                d2h_segment_count += 1
                d2h_row_count += int(segment_ids_cpu.numel())
                d2h_bytes += (
                    row_buffer.numel() * row_buffer.element_size() +
                    exp_avg_buffer.numel() * exp_avg_buffer.element_size() +
                    exp_avg_sq_buffer.numel() * exp_avg_sq_buffer.element_size()
                )
                writeback_segments.append((name, batch["param"], batch["state"], segment_ids_cpu, row_buffer, exp_avg_buffer, exp_avg_sq_buffer))
            d2h_launch_ms = (time.perf_counter() - t_d2h_launch_start) * 1000.0
            ready_event = None
            if self.use_cuda:
                ready_event = torch.cuda.Event()
                ready_event.record(torch.cuda.current_stream(self.device))

            def _write_segment_when_ready(event, segment) -> None:
                if event is not None:
                    event.synchronize()
                _, param, state, segment_ids_cpu, row_buffer, exp_avg_buffer, exp_avg_sq_buffer = segment
                with torch.no_grad():
                    param.index_copy_(0, segment_ids_cpu, row_buffer)
                    state["exp_avg"].index_copy_(0, segment_ids_cpu, exp_avg_buffer)
                    state["exp_avg_sq"].index_copy_(0, segment_ids_cpu, exp_avg_sq_buffer)

            futures = [
                self._cpu_writeback_executor.submit(_write_segment_when_ready, ready_event, segment)
                for segment in writeback_segments
            ]
            self._append_pending_cpu_writeback_futures(futures, writeback_ids_cpu)
        writeback_ms = (time.perf_counter() - t_writeback_start) * 1000.0
        total_writeback_count = int(writeback_ids_cpu.numel())

        metrics = DynamicVocabStep(
            active_ids_cpu=grad_accum_ids_cpu,
            active_vocab=None,
            optimizer_state=None,
            unique_count=int(grad_accum_ids_cpu.numel()),
            live_count=int(live_lm_head_slot_ids_cpu.numel()) if live_lm_head_slot_ids_cpu.numel() > 0 else live_count,
            u_capacity=self.grad_accum_u_max if self.grad_accum_u_max > 0 else int(grad_accum_ids_cpu.numel()),
            stage_count=int(self._grad_accum_stage_count),
            writeback_count=total_writeback_count,
            grad_accum_flush_ms=flush_ms,
            grad_accum_queue_count=max(int(grad_accum_ids_cpu.numel()) - live_count, 0),
            grad_accum_stage_ms=stage_ms,
            grad_accum_apply_ms=apply_ms,
            grad_accum_restore_ms=restore_ms,
            grad_accum_writeback_ms=writeback_ms,
            grad_accum_resident_count=live_count,
            d2h_launch_ms=d2h_launch_ms,
            d2h_sync_ms=d2h_sync_ms,
            cpu_writeback_ms=cpu_writeback_ms,
            d2h_segment_count=d2h_segment_count,
            d2h_row_count=d2h_row_count,
            d2h_bytes=d2h_bytes,
            optimizer_ms=apply_ms,
            fixed_u_mode=self.fixed_u_mode,
            cold_bias_clamped_count=self._grad_accum_window_cold_bias_clamped_count,
            cold_bias_abs_max=self._grad_accum_window_cold_bias_abs_max,
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
            updated_tables: set[str] = set()
            active_step_values_cpu = self._get_token_event_step_values_cpu(step_ctx.active_ids_cpu)
            if self._adamw_update_(
                "wte",
                self.fixed_params["wte"],
                self.fixed_optimizer_state["wte"],
                active_step_values_cpu,
                slot_ids_cpu=step_ctx.active_slot_ids_cpu,
            ):
                self._mark_table_updated_("wte", updated_tables)
            if self._adamw_update_(
                "lm_head",
                self.fixed_params["lm_head"],
                self.fixed_optimizer_state["lm_head"],
                active_step_values_cpu,
                slot_ids_cpu=step_ctx.active_slot_ids_cpu,
                hot_activation_counts_cpu=step_ctx.hot_activation_counts_cpu,
            ):
                self._mark_table_updated_("lm_head", updated_tables)
            if step_ctx.warm_slot_ids_cpu is not None and step_ctx.warm_slot_ids_cpu.numel() > 0:
                assert step_ctx.warm_ids_cpu is not None
                warm_step_values_cpu = self._get_token_event_step_values_cpu(step_ctx.warm_ids_cpu)
                if self._adamw_update_(
                    "lm_head",
                    self.fixed_params["lm_head"],
                    self.fixed_optimizer_state["lm_head"],
                    warm_step_values_cpu,
                    slot_ids_cpu=step_ctx.warm_slot_ids_cpu,
                    lr_override=float(self.table_specs["lm_head"]["warm_lr"]),
                ):
                    self._mark_table_updated_("lm_head", updated_tables)
            if step_ctx.cold_slot_ids_cpu is not None and step_ctx.cold_slot_ids_cpu.numel() > 0:
                assert step_ctx.cold_ids_cpu is not None
                cold_step_values_cpu = self._get_token_event_step_values_cpu(step_ctx.cold_ids_cpu)
                if self._adamw_update_(
                    "lm_head",
                    self.fixed_params["lm_head"],
                    self.fixed_optimizer_state["lm_head"],
                    cold_step_values_cpu,
                    slot_ids_cpu=step_ctx.cold_slot_ids_cpu,
                    lr_override=float(self.table_specs["lm_head"]["cold_lr"]),
                ):
                    self._mark_table_updated_("lm_head", updated_tables)
            for layer_name in step_ctx.active_vocab["value_embeds"]:
                param_name = f"value_embeds.{layer_name}"
                if self._adamw_update_(
                    param_name,
                    self.fixed_params[param_name],
                    self.fixed_optimizer_state[param_name],
                    active_step_values_cpu,
                    slot_ids_cpu=step_ctx.active_slot_ids_cpu,
                ):
                    self._mark_table_updated_(param_name, updated_tables)

            updated_token_ids = [step_ctx.active_ids_cpu]
            if step_ctx.warm_ids_cpu is not None and step_ctx.warm_ids_cpu.numel() > 0:
                updated_token_ids.append(step_ctx.warm_ids_cpu)
            if step_ctx.cold_ids_cpu is not None and step_ctx.cold_ids_cpu.numel() > 0:
                updated_token_ids.append(step_ctx.cold_ids_cpu)
            self._increment_token_event_step_counts_(torch.cat(updated_token_ids))

            writeback_ids_cpu = step_ctx.active_ids_cpu if step_ctx.is_last_step else step_ctx.writeback_ids_cpu
            writeback_slot_ids_cpu = step_ctx.active_slot_ids_cpu if step_ctx.is_last_step else step_ctx.writeback_slot_ids_cpu
            if writeback_ids_cpu is None:
                writeback_ids_cpu = torch.empty(0, dtype=torch.long)
            if writeback_slot_ids_cpu is None:
                writeback_slot_ids_cpu = torch.empty(0, dtype=torch.long)
            writeback_ids_cpu = writeback_ids_cpu.detach().to(device="cpu", dtype=torch.long)
            writeback_slot_ids_cpu = writeback_slot_ids_cpu.detach().to(device="cpu", dtype=torch.long)
            step_ctx.writeback_count = writeback_ids_cpu.numel()

            if step_ctx.is_last_step:
                self._writeback_fixed_rows_(writeback_ids_cpu, writeback_slot_ids_cpu)
            else:
                self._queue_fixed_rows_writeback_(writeback_ids_cpu, writeback_slot_ids_cpu)
            if (
                step_ctx.is_last_step and
                step_ctx.warm_slot_ids_cpu is not None and
                step_ctx.cold_slot_ids_cpu is not None and
                step_ctx.warm_ids_cpu is not None and
                step_ctx.cold_ids_cpu is not None
            ):
                cloud_slot_ids_cpu = torch.cat((step_ctx.warm_slot_ids_cpu, step_ctx.cold_slot_ids_cpu))
                cloud_ids_cpu = torch.cat((step_ctx.warm_ids_cpu, step_ctx.cold_ids_cpu))
            else:
                cloud_slot_ids_cpu = torch.empty(0, dtype=torch.long)
                cloud_ids_cpu = torch.empty(0, dtype=torch.long)
            if cloud_slot_ids_cpu.numel() > 0:
                self._writeback_fixed_lm_head_rows_(cloud_ids_cpu, cloud_slot_ids_cpu)
                step_ctx.writeback_count += int(cloud_slot_ids_cpu.numel())
            self._clear_fixed_grads()
            step_ctx.active_vocab = None
            step_ctx.optimizer_state = None
            self.runtime_step += 1
            return step_ctx

        updated_tables: set[str] = set()
        active_step_values_cpu = self._get_token_event_step_values_cpu(step_ctx.active_ids_cpu)
        if self._adamw_update_("wte", step_ctx.active_vocab["wte"], step_ctx.optimizer_state["wte"], active_step_values_cpu):
            self._mark_table_updated_("wte", updated_tables)
        if self._adamw_update_(
            "lm_head",
            step_ctx.active_vocab["lm_head"],
            step_ctx.optimizer_state["lm_head"],
            active_step_values_cpu,
            hot_activation_counts_cpu=step_ctx.hot_activation_counts_cpu,
        ):
            self._mark_table_updated_("lm_head", updated_tables)
        for layer_name, active_param in step_ctx.active_vocab["value_embeds"].items():
            param_name = f"value_embeds.{layer_name}"
            if self._adamw_update_(param_name, active_param, step_ctx.optimizer_state[param_name], active_step_values_cpu):
                self._mark_table_updated_(param_name, updated_tables)
        self._increment_token_event_step_counts_(step_ctx.active_ids_cpu)
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
