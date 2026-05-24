import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

def _sparse_adamw_row_chunk_size() -> int:
    """Chunk sparse table AdamW row updates to shorten individual GPU kernels (helps display-GPU TDR)."""
    try:
        v = int(os.environ.get("NANOCHAT_SPARSE_ADAMW_CHUNK", "2048"))
    except ValueError:
        v = 2048
    return max(256, v)


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
    prep_cpu_reuse_map_ms: float = 0.0
    prep_gpu_reuse_input_ms: float = 0.0
    prep_gpu_reuse_lm_head_ms: float = 0.0
    prep_clear_grads_ms: float = 0.0
    prep_logit_mask_ms: float = 0.0
    prep_union_io_h2d_ms: float = 0.0
    d2h_segment_count: int = 0
    d2h_row_count: int = 0
    d2h_bytes: int = 0
    active_param_bytes: int = 0
    active_grad_bytes: int = 0
    active_optimizer_bytes: int = 0
    u_capacity: int = 0
    stage_count: int = 0
    writeback_count: int = 0
    # Note: cloud/warm/cold/hard_negative/random_fill + associated timing fields removed
    # (dead experimental code from warm/cold cloud and cold logit bias experiments).
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
        self._fixed_input_global_to_slot_cpu = None
        self._fixed_input_slot_survived_buf_cpu = None
        self._grad_accum_wte_local_to_slot_cpu = None
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
        self._grad_accum_warm_ids_cpu = None
        self._grad_accum_cold_ids_cpu = None
        self._grad_accum_random_fill_ids_cpu = None
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
        self._apply_stage_executor = ThreadPoolExecutor(max_workers=1)
        self._pending_apply_stage_future = None
        self._pending_apply_stage_ids_cpu = None
        self._gpu_stage_buffers = {}
        self._last_hidden_query_ms = 0.0
        self.global_token_count_cpu = torch.zeros((model.config.vocab_size,), dtype=torch.long)
        if self.fixed_u_mode:
            self.fixed_slot_to_global_cpu = torch.full((self.fixed_u_max,), -1, dtype=torch.long)
            self.fixed_input_slot_to_global_cpu = torch.full((self.fixed_input_u_max,), -1, dtype=torch.long)
            self.fixed_lm_head_slot_to_global_cpu = torch.full((self.lm_head_u_max,), -1, dtype=torch.long)
            self.fixed_active_mask_cpu = torch.zeros(self.fixed_u_max, dtype=torch.bool)
            # Persistent inverse map: global_id → slot_id, maintained incrementally across windows
            # to avoid per-step torch.full((vocab_size,), -1) scatter-map rebuilds in the hot path.
            self._fixed_input_global_to_slot_cpu = torch.full((self.model.config.vocab_size,), -1, dtype=torch.long)
            # Working buffer for set-difference in slot space (fixed_input_u_max, not vocab_size)
            self._fixed_input_slot_survived_buf_cpu = torch.zeros(self.fixed_input_u_max, dtype=torch.bool)
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

    def _sample_random_token_ids(self, excluded_mask_cpu: torch.Tensor, limit: int, *, seed: int) -> torch.Tensor:
        if limit <= 0:
            return self._empty_long_cpu()
        candidate_ids_cpu = torch.nonzero(~excluded_mask_cpu, as_tuple=False).flatten()
        if candidate_ids_cpu.numel() == 0:
            return self._empty_long_cpu()
        if limit >= candidate_ids_cpu.numel():
            return candidate_ids_cpu
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) & ((1 << 63) - 1))
        sampled_indices_cpu = torch.randperm(candidate_ids_cpu.numel(), generator=generator)[:limit]
        return candidate_ids_cpu.index_select(0, sampled_indices_cpu)

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
        if self._fixed_input_global_to_slot_cpu is not None:
            self._fixed_input_global_to_slot_cpu.fill_(-1)
        self._grad_accum_wte_local_to_slot_cpu = None
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
        self._grad_accum_warm_ids_cpu = None
        self._grad_accum_cold_ids_cpu = None
        self._grad_accum_random_fill_ids_cpu = None
        self._grad_accum_window_cold_steps_cpu = None
        self._grad_accum_window_cold_logit_bias_cpu = None
        self._grad_accum_window_cold_bias_clamped_count = 0
        self._grad_accum_window_cold_bias_abs_max = 0.0
        self._pending_apply_stage_future = None
        self._pending_apply_stage_ids_cpu = None

    def _resolve_grad_accum_staged_non_live_ids(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._grad_accum_live or self._grad_accum_ids_cpu is None:
            return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

        grad_accum_ids_cpu = self._grad_accum_ids_cpu
        grad_accum_count = self._grad_accum_count
        live_union_mask_cpu = torch.zeros(grad_accum_count, dtype=torch.bool)
        if self.fixed_u_mode and self._fixed_live_state:
            assert self._grad_accum_global_to_local_cpu is not None
            assert self.fixed_input_slot_to_global_cpu is not None
            assert self.fixed_lm_head_slot_to_global_cpu is not None
            live_slot_ids_cpu = torch.nonzero(self.fixed_input_slot_to_global_cpu >= 0, as_tuple=False).flatten()
            if live_slot_ids_cpu.numel() > 0:
                live_global_ids_cpu = self.fixed_input_slot_to_global_cpu.index_select(0, live_slot_ids_cpu)
                live_union_row_ids_cpu = self._grad_accum_global_to_local_cpu.index_select(0, live_global_ids_cpu)
                valid_live_mask_cpu = live_union_row_ids_cpu >= 0
                if valid_live_mask_cpu.any():
                    live_union_mask_cpu[live_union_row_ids_cpu[valid_live_mask_cpu]] = True
            live_lm_head_slot_ids_cpu = torch.nonzero(self.fixed_lm_head_slot_to_global_cpu >= 0, as_tuple=False).flatten()
            if live_lm_head_slot_ids_cpu.numel() > 0:
                live_lm_head_global_ids_cpu = self.fixed_lm_head_slot_to_global_cpu.index_select(0, live_lm_head_slot_ids_cpu)
                live_lm_head_union_row_ids_cpu = self._grad_accum_global_to_local_cpu.index_select(0, live_lm_head_global_ids_cpu)
                valid_lm_head_mask_cpu = live_lm_head_union_row_ids_cpu >= 0
                if valid_lm_head_mask_cpu.any():
                    live_union_mask_cpu[live_lm_head_union_row_ids_cpu[valid_lm_head_mask_cpu]] = True

        non_live_union_row_ids_cpu = torch.nonzero(~live_union_mask_cpu, as_tuple=False).flatten()
        if non_live_union_row_ids_cpu.numel() == 0:
            return non_live_union_row_ids_cpu, torch.empty(0, dtype=torch.long)
        cached_non_live_mask_cpu = torch.zeros(non_live_union_row_ids_cpu.numel(), dtype=torch.bool)
        if self._grad_accum_cached_union_mask_cpu is not None:
            cached_non_live_mask_cpu = self._grad_accum_cached_union_mask_cpu[non_live_union_row_ids_cpu]
        staged_non_live_union_row_ids_cpu = non_live_union_row_ids_cpu[~cached_non_live_mask_cpu]
        if staged_non_live_union_row_ids_cpu.numel() == 0:
            return staged_non_live_union_row_ids_cpu, torch.empty(0, dtype=torch.long)
        staged_non_live_grad_accum_ids_cpu = grad_accum_ids_cpu.index_select(0, staged_non_live_union_row_ids_cpu)
        return staged_non_live_union_row_ids_cpu, staged_non_live_grad_accum_ids_cpu

    def _prefetch_apply_stage_cpu_rows(self, ids_cpu: torch.Tensor) -> dict[str, dict]:
        ids_cpu = ids_cpu.detach().to(device="cpu", dtype=torch.long)
        payload = {
            "ids_cpu": ids_cpu,
        }
        if ids_cpu.numel() == 0:
            return payload
        for name, spec in self.table_specs.items():
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
            payload[name] = {
                "param": rows,
                "exp_avg": exp_avg,
                "exp_avg_sq": exp_avg_sq,
            }
        return payload

    def prefetch_apply_accumulated_gradients(self) -> None:
        if not self.use_cuda or not self._grad_accum_live or self._grad_accum_ids_cpu is None:
            return
        _, staged_non_live_grad_accum_ids_cpu = self._resolve_grad_accum_staged_non_live_ids()
        if staged_non_live_grad_accum_ids_cpu.numel() == 0:
            self._pending_apply_stage_future = None
            self._pending_apply_stage_ids_cpu = None
            return
        if (
            self._pending_apply_stage_future is not None and
            self._pending_apply_stage_ids_cpu is not None and
            torch.equal(self._pending_apply_stage_ids_cpu, staged_non_live_grad_accum_ids_cpu)
        ):
            return
        self._pending_apply_stage_ids_cpu = staged_non_live_grad_accum_ids_cpu.clone()
        self._pending_apply_stage_future = self._apply_stage_executor.submit(
            self._prefetch_apply_stage_cpu_rows,
            staged_non_live_grad_accum_ids_cpu,
        )

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
        self._grad_accum_warm_ids_cpu = torch.empty(0, dtype=torch.long)
        self._grad_accum_cold_ids_cpu = torch.empty(0, dtype=torch.long)
        self._grad_accum_random_fill_ids_cpu = torch.empty(0, dtype=torch.long)
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

    def _queue_fixed_rows_writeback_(
        self,
        global_ids_cpu: torch.Tensor,
        slot_ids_cpu: torch.Tensor,
        table_names: Optional[tuple[str, ...]] = None,
    ) -> None:
        global_ids_cpu = global_ids_cpu.detach().to(device="cpu", dtype=torch.long)
        slot_ids_cpu = slot_ids_cpu.detach().to(device="cpu", dtype=torch.long)
        if global_ids_cpu.numel() == 0:
            return
        if not self.use_cuda:
            self._writeback_fixed_rows_(global_ids_cpu, slot_ids_cpu, table_names=table_names)
            return
        self._flush_pending_cpu_writeback(global_ids_cpu)
        slot_ids_device = slot_ids_cpu.to(self.device)
        writeback_segments = []
        if table_names is None:
            table_names = tuple(self.table_specs.keys())
        for name in table_names:
            spec = self.table_specs[name]
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
        buffer_key = name
        if name.startswith("stage:fixed:"):
            parts = name.split(":", 3)
            if len(parts) == 4:
                buffer_key = f"{parts[0]}:{parts[1]}:{parts[3]}"
        buffer = self._gpu_stage_buffers.get(buffer_key)
        needs_new = (
            buffer is None or
            buffer.dtype != dtype or
            buffer.dim() != len(shape) or
            any(buffer.size(dim) < shape_dim for dim, shape_dim in enumerate(shape))
        )
        if needs_new:
            buffer = torch.empty(shape, dtype=dtype, device=self.device)
            self._gpu_stage_buffers[buffer_key] = buffer
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
                if self.disable_fixed_overlap_reuse:
                    non_lm_head_ids_cpu = grad_accum_ids_cpu
                else:
                    # Use persistent inverse map: O(n_new) lookup, no torch.full scatter-map alloc.
                    _prefetch_reused = self._fixed_input_global_to_slot_cpu.index_select(0, grad_accum_ids_cpu)
                    non_lm_head_ids_cpu = grad_accum_ids_cpu[_prefetch_reused < 0]
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
    ) -> DynamicVocabStep:
        self._flush_pending_cpu_writeback()
        active_ids_cpu = active_ids_cpu.detach().to(device="cpu", dtype=torch.long)
        cold_steps_cpu, hot_activation_counts_cpu = self._capture_cold_steps_cpu(active_ids_cpu)
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
            hot_activation_counts_cpu=hot_activation_counts_cpu,
        )

    def _prepare_fixed_step(
        self,
        step_meta: dict,
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
        # Cloud/warm/cold ids from step_meta are no longer supported (dead experimental code).
        # The dataloader and manifest path never produce them.
        warm_ids_cpu = self._empty_long_cpu()
        cold_ids_cpu = self._empty_long_cpu()
        random_fill_ids_cpu = self._empty_long_cpu()
        cloud_ids_cpu = self._empty_long_cpu()

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
        if grad_accum_steps > 1:
            if preserve_resident_grads:
                if self._grad_accum_warm_ids_cpu is None or self._grad_accum_cold_ids_cpu is None or self._grad_accum_random_fill_ids_cpu is None:
                    raise RuntimeError("Sparse grad accumulation cloud state is missing for resident window")
                warm_ids_cpu = self._grad_accum_warm_ids_cpu
                cold_ids_cpu = self._grad_accum_cold_ids_cpu
                random_fill_ids_cpu = self._grad_accum_random_fill_ids_cpu
            else:
                assert self._grad_accum_warm_ids_cpu is not None
                assert self._grad_accum_cold_ids_cpu is not None
                assert self._grad_accum_random_fill_ids_cpu is not None
                self._grad_accum_warm_ids_cpu = warm_ids_cpu.clone()
                self._grad_accum_cold_ids_cpu = cold_ids_cpu.clone()
                self._grad_accum_random_fill_ids_cpu = random_fill_ids_cpu.clone()
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
        # Cold logit bias computation removed (dead experimental feature).
        current_cold_logit_bias_cpu = self._empty_long_cpu().to(dtype=torch.float32)
        cold_bias_clamped_count = 0
        cold_bias_abs_max = 0.0
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
        if grad_accum_steps > 1:
            available_cloud_slot_ids_cpu = torch.arange(grad_accum_ids_cpu.numel(), self.lm_head_u_max, dtype=torch.long)
        else:
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
            if grad_accum_steps > 1:
                cloud_slot_ids_cpu = available_cloud_slot_ids_cpu[:cloud_ids_cpu.numel()]
                if not preserve_resident_grads:
                    cloud_stage_ids_cpu = cloud_ids_cpu
                    cloud_stage_slot_ids_cpu = cloud_slot_ids_cpu
            if self._fixed_live_state:
                assert self.fixed_lm_head_slot_to_global_cpu is not None
                assert self.fixed_active_mask_cpu is not None
                if grad_accum_steps == 1:
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
            elif grad_accum_steps == 1:
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
        prep_cpu_reuse_map_ms = 0.0
        prep_gpu_reuse_input_ms = 0.0
        prep_gpu_reuse_lm_head_ms = 0.0
        prep_clear_grads_ms = 0.0
        prep_logit_mask_ms = 0.0
        prep_union_io_h2d_ms = 0.0
        _used_persistent_input_slots = False
        deferred_writeback_ids_cpu = self._empty_long_cpu()
        deferred_writeback_slot_ids_cpu = self._empty_long_cpu()
        deferred_lm_head_writeback_ids_cpu = self._empty_long_cpu()
        deferred_lm_head_writeback_slot_ids_cpu = self._empty_long_cpu()
        input_stage_ids_cpu = stage_ids_cpu
        input_stage_slot_ids_cpu = stage_slot_ids_cpu
        if union_input_tables:
            input_stage_ids_cpu = self._empty_long_cpu() if preserve_resident_grads else grad_accum_ids_cpu
            input_stage_slot_ids_cpu = self._empty_long_cpu() if preserve_resident_grads else torch.arange(grad_accum_ids_cpu.numel(), dtype=torch.long)
        _t0_cpu_map = time.perf_counter()
        if grad_accum_steps > 1 and not preserve_resident_grads and grad_accum_ids_cpu.numel() > 0 and self._fixed_live_state and not self.disable_fixed_overlap_reuse:
            # New-window boundary only (preserve_resident_grads microsteps 1-31 are skipped:
            # the window is unchanged so no tokens leave, deferred_writeback stays empty,
            # and no GPU-GPU copies are needed — a pure win on all 31 microsteps).
            assert self.fixed_input_slot_to_global_cpu is not None
            # --- Persistent stable input-table slot assignment (replaces Block B entirely) ---
            # Step 1: which new tokens already have a slot? O(n_new) lookup, no vocab_size alloc.
            _existing_input_slots = self._fixed_input_global_to_slot_cpu.index_select(0, grad_accum_ids_cpu)
            _arriving_input_mask = _existing_input_slots < 0
            _surviving_input_slots = _existing_input_slots[~_arriving_input_mask]
            # Step 2: find leaving slots via set-difference in slot space (O(fixed_input_u_max)).
            _all_input_occupied = self.fixed_input_slot_to_global_cpu >= 0
            if _surviving_input_slots.numel() > 0:
                self._fixed_input_slot_survived_buf_cpu.zero_()
                self._fixed_input_slot_survived_buf_cpu.scatter_(0, _surviving_input_slots, True)
                _leaving_input_mask = _all_input_occupied & ~self._fixed_input_slot_survived_buf_cpu
            else:
                _leaving_input_mask = _all_input_occupied
            _leaving_input_slot_ids = torch.nonzero(_leaving_input_mask, as_tuple=False).flatten()
            _leaving_input_ids = (
                self.fixed_input_slot_to_global_cpu[_leaving_input_slot_ids]
                if _leaving_input_slot_ids.numel() > 0 else self._empty_long_cpu()
            )
            # Step 3: assign slots to arriving tokens (recycle from leaving pool first).
            _arriving_input_ids = grad_accum_ids_cpu[_arriving_input_mask]
            _n_arriving = _arriving_input_ids.numel()
            _n_leaving = _leaving_input_slot_ids.numel()
            if _n_arriving > 0:
                if _n_arriving <= _n_leaving:
                    _arriving_input_slot_ids = _leaving_input_slot_ids[:_n_arriving]
                    _freed_input_slot_ids = _leaving_input_slot_ids[_n_arriving:]
                else:
                    # Window grew past old occupancy: pull extras from unoccupied range.
                    _extra_slots = torch.nonzero(~_all_input_occupied, as_tuple=False).flatten()[:_n_arriving - _n_leaving]
                    _arriving_input_slot_ids = (
                        torch.cat((_leaving_input_slot_ids, _extra_slots)) if _n_leaving > 0 else _extra_slots
                    )
                    _freed_input_slot_ids = self._empty_long_cpu()
            else:
                _arriving_input_slot_ids = self._empty_long_cpu()
                _freed_input_slot_ids = _leaving_input_slot_ids
            # Step 4: update persistent forward and inverse maps.
            if _leaving_input_ids.numel() > 0:
                self._fixed_input_global_to_slot_cpu[_leaving_input_ids] = -1
            if _freed_input_slot_ids.numel() > 0:
                self.fixed_input_slot_to_global_cpu[_freed_input_slot_ids] = -1
            if _arriving_input_ids.numel() > 0:
                self._fixed_input_global_to_slot_cpu[_arriving_input_ids] = _arriving_input_slot_ids
                self.fixed_input_slot_to_global_cpu[_arriving_input_slot_ids] = _arriving_input_ids
            # Deferred writeback: leaving tokens may have uncommitted GPU→CPU write jobs.
            deferred_writeback_ids_cpu = _leaving_input_ids
            deferred_writeback_slot_ids_cpu = _leaving_input_slot_ids[:_leaving_input_ids.numel()]
            # Only arriving tokens need CPU→GPU staging; surviving tokens stay in their correct slot.
            input_stage_ids_cpu = _arriving_input_ids
            input_stage_slot_ids_cpu = _arriving_input_slot_ids
            _used_persistent_input_slots = True
            # Cache local-position → wte-slot mapping so union_inputs can be remapped on every
            # microstep in this window (including preserve_resident_grads ones).  Index i in this
            # tensor gives the GPU slot that holds grad_accum_ids_cpu[i]'s embedding row.
            self._grad_accum_wte_local_to_slot_cpu = self._fixed_input_global_to_slot_cpu.index_select(
                0, grad_accum_ids_cpu
            )
            # --- lm_head deferred writeback detection (one torch.full; lm_head fix is a follow-up) ---
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
        prep_cpu_reuse_map_ms = (time.perf_counter() - _t0_cpu_map) * 1000.0
        prep_gpu_reuse_input_ms = 0.0
        _t0_gpu_lm_head = time.perf_counter()
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

        prep_gpu_reuse_lm_head_ms = (time.perf_counter() - _t0_gpu_lm_head) * 1000.0
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
                table_names=self._union_input_table_names(),
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
        _t0_clear_grads = time.perf_counter()
        if preserve_resident_grads:
            if not union_input_tables:
                self._zero_fixed_grad_slots_(
                    stage_slot_ids_cpu,
                    table_names=self._union_input_table_names(),
                )
        else:
            self._clear_fixed_grads()
        prep_clear_grads_ms = (time.perf_counter() - _t0_clear_grads) * 1000.0
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
                        if cloud_stage_ids_cpu.numel() > 0:
                            stage_ids_for_name = (
                                torch.cat((grad_accum_stage_ids_cpu, cloud_stage_ids_cpu))
                                if grad_accum_stage_ids_cpu.numel() > 0 else cloud_stage_ids_cpu
                            )
                            stage_slots_for_name = (
                                torch.cat((grad_accum_stage_slot_ids_cpu, cloud_stage_slot_ids_cpu))
                                if grad_accum_stage_slot_ids_cpu.numel() > 0 else cloud_stage_slot_ids_cpu
                            )
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

        self.fixed_slot_to_global_cpu.copy_(slot_to_global_cpu)
        assert self.fixed_input_slot_to_global_cpu is not None
        if not _used_persistent_input_slots and not (grad_accum_steps > 1 and not self.disable_fixed_overlap_reuse and preserve_resident_grads):
            # Non-persistent path (first step, disabled, or grad_accum_steps==1):
            # reset forward map to contiguous slot assignment. Slots are sequential (slot i = position i),
            # so no union_inputs remapping is needed — clear the cache.
            self._grad_accum_wte_local_to_slot_cpu = None
            self.fixed_input_slot_to_global_cpu.fill_(-1)
            if grad_accum_steps > 1:
                if grad_accum_ids_cpu.numel() > 0:
                    self.fixed_input_slot_to_global_cpu[:grad_accum_ids_cpu.numel()].copy_(grad_accum_ids_cpu)
                    if not self.disable_fixed_overlap_reuse:
                        # Initialize persistent inverse map so the next step can use stable slots.
                        self._fixed_input_global_to_slot_cpu.fill_(-1)
                        self._fixed_input_global_to_slot_cpu[grad_accum_ids_cpu] = torch.arange(
                            grad_accum_ids_cpu.numel(), dtype=torch.long
                        )
            else:
                self.fixed_input_slot_to_global_cpu[:self.fixed_u_max].copy_(slot_to_global_cpu)
        # else: persistent slots (forward map already updated incrementally in the block above),
        #       or preserve_resident_grads (forward map is correct from microstep 0, no reset needed).
        if grad_accum_steps > 1:
            if not preserve_resident_grads:
                self.fixed_lm_head_slot_to_global_cpu.fill_(-1)
                self.fixed_lm_head_slot_to_global_cpu[:grad_accum_ids_cpu.numel()].copy_(grad_accum_ids_cpu)
        else:
            self.fixed_lm_head_slot_to_global_cpu.fill_(-1)
            self.fixed_lm_head_slot_to_global_cpu[:self.fixed_u_max].copy_(slot_to_global_cpu)
        if cloud_ids_cpu.numel() > 0 and (grad_accum_steps == 1 or not preserve_resident_grads):
            self.fixed_lm_head_slot_to_global_cpu.index_copy_(0, cloud_slot_ids_cpu, cloud_ids_cpu)
        self.fixed_active_mask_cpu.copy_(active_mask_cpu)
        _t0_logit_mask = time.perf_counter()
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
            if cloud_slot_ids_cpu.numel() > 0 and (grad_accum_steps == 1 or not preserve_resident_grads):
                cloud_slot_ids_device = cloud_slot_ids_cpu.to(self.device)
                self.fixed_logit_mask.index_fill_(0, cloud_slot_ids_device, True)
        # Cold logit bias support removed (dead experimental feature).
        self._fixed_live_state = True
        prep_logit_mask_ms = (time.perf_counter() - _t0_logit_mask) * 1000.0

        _t0_union_io = time.perf_counter()
        union_inputs = None
        if union_inputs_cpu_local is not None and grad_accum_steps > 1:
            _inputs_local = union_inputs_cpu_local
            if self._grad_accum_wte_local_to_slot_cpu is not None:
                # Persistent stable slots: remap local-position indices to actual GPU slot indices.
                # union_inputs_cpu_local contains positions (0..N-1) in grad_accum_ids_cpu;
                # with stable slots, slot(token) != position(token), so we must translate.
                _inputs_local = self._grad_accum_wte_local_to_slot_cpu[_inputs_local]
            union_inputs = _inputs_local.detach().to(self.device, non_blocking=self.use_cuda)
        union_targets = None
        if union_targets_cpu_local is not None and grad_accum_steps > 1:
            union_targets = union_targets_cpu_local.detach().to(self.device, non_blocking=self.use_cuda)
        prep_union_io_h2d_ms = (time.perf_counter() - _t0_union_io) * 1000.0

        step_active_vocab = {
            **self.fixed_active_vocab,
            "value_embeds": self.fixed_active_vocab["value_embeds"],
            "lm_head": self.fixed_active_vocab["lm_head"],
        }
        if grad_accum_steps > 1:
            if use_logit_mask:
                step_active_vocab["logit_mask"] = self.fixed_logit_mask
        else:
            if use_logit_mask:
                step_active_vocab["logit_mask"] = self.fixed_logit_mask
        step_optimizer_state = {
            **self.fixed_optimizer_state,
        }

        # Note: cloud/warm/cold/hard_negative fields removed (dead experimental code).
        # lm_head_active now equals the regular active/grad_accum set (lm_head_u_max == fixed_u_max, no clouds).
        return DynamicVocabStep(
            active_ids_cpu=active_ids_cpu,
            active_vocab=step_active_vocab,
            optimizer_state=step_optimizer_state,
            union_inputs=union_inputs,
            union_targets=union_targets,
            unique_count=int(grad_accum_ids_cpu.numel()) if grad_accum_steps > 1 else lm_head_active_ids_cpu.numel(),
            live_count=lm_head_active_ids_cpu.numel(),
            step_u_count=active_ids_cpu.numel(),
            u_capacity=self.fixed_u_max,
            stage_count=stage_ids_cpu.numel() + (grad_accum_stage_ids_cpu.numel() if grad_accum_steps > 1 and not preserve_resident_grads else 0),
            active_slot_ids_cpu=active_slot_ids_cpu,
            active_mask_cpu=active_mask_cpu,
            slot_to_global_cpu=slot_to_global_cpu,
            stage_ids_cpu=stage_ids_cpu,
            stage_slot_ids_cpu=stage_slot_ids_cpu,
            writeback_ids_cpu=writeback_ids_cpu,
            writeback_slot_ids_cpu=writeback_slot_ids_cpu,
            grad_accum_ids_cpu=grad_accum_ids_cpu,
            lm_head_active_ids_cpu=grad_accum_ids_cpu if grad_accum_steps > 1 else lm_head_active_ids_cpu,
            lm_head_active_slot_ids_cpu=grad_accum_stage_slot_ids_cpu if grad_accum_steps > 1 else lm_head_active_slot_ids_cpu,
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
            prep_cpu_reuse_map_ms=prep_cpu_reuse_map_ms,
            prep_gpu_reuse_input_ms=prep_gpu_reuse_input_ms,
            prep_gpu_reuse_lm_head_ms=prep_gpu_reuse_lm_head_ms,
            prep_clear_grads_ms=prep_clear_grads_ms,
            prep_logit_mask_ms=prep_logit_mask_ms,
            prep_union_io_h2d_ms=prep_union_io_h2d_ms,
        )

    def prepare_step(
        self,
        active_ids_cpu,
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
                prefetched_stage=prefetched_stage,
                prefetched_wait_ms=prefetch_wait_ms,
                prefetched_hit=prefetch_hit,
            )
        return self._prepare_dynamic_step(
            active_ids_cpu,
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

        def _apply_adamw_rows(
            param_rows: torch.Tensor,
            grad_rows: torch.Tensor,
            exp_avg_rows: torch.Tensor,
            exp_avg_sq_rows: torch.Tensor,
            row_lr_arg: Optional[torch.Tensor],
            step_val: int | torch.Tensor,
        ) -> None:
            if self.weight_decay != 0.0:
                param_rows.mul_(1 - base_lr * self.weight_decay)
            exp_avg_rows.lerp_(grad_rows, 1 - self.beta1)
            exp_avg_sq_rows.lerp_(grad_rows.square(), 1 - self.beta2)

            if isinstance(step_val, torch.Tensor):
                step_values = step_val.detach().to(device=param_rows.device, dtype=torch.float32)
                if step_values.dim() != 1 or step_values.numel() != param_rows.size(0):
                    raise ValueError(
                        f"Sparse AdamW step values must be a 1D tensor with one entry per row, got shape {tuple(step_values.shape)} for {param_rows.size(0)} rows"
                    )
                if step_values.numel() > 0 and torch.equal(step_values, step_values[:1].expand_as(step_values)):
                    scalar_step_value = int(step_values[0].item())
                    bias1 = 1 - self.beta1 ** scalar_step_value
                    bias2 = 1 - self.beta2 ** scalar_step_value
                    denom = (exp_avg_sq_rows / bias2).sqrt().add_(self.eps)
                    if row_lr_arg is not None:
                        row_lr_view = row_lr_arg.view((-1,) + (1,) * (param_rows.dim() - 1))
                        param_rows.add_((exp_avg_rows / denom) * row_lr_view, alpha=-1.0 / bias1)
                    else:
                        step_size = base_lr / bias1
                        param_rows.addcdiv_(exp_avg_rows, denom, value=-step_size)
                else:
                    bias1 = 1 - torch.pow(torch.full_like(step_values, self.beta1), step_values)
                    bias2 = 1 - torch.pow(torch.full_like(step_values, self.beta2), step_values)
                    view_shape = (-1,) + (1,) * (param_rows.dim() - 1)
                    denom = (exp_avg_sq_rows / bias2.view(view_shape)).sqrt().add_(self.eps)
                    if row_lr_arg is not None:
                        scale = row_lr_arg / bias1.to(dtype=row_lr_arg.dtype)
                    else:
                        scale = torch.full_like(bias1, base_lr, dtype=torch.float32) / bias1
                    param_rows.add_((exp_avg_rows / denom) * scale.to(dtype=param_rows.dtype).view(view_shape), alpha=-1.0)
            else:
                bias1 = 1 - self.beta1 ** step_val
                bias2 = 1 - self.beta2 ** step_val
                denom = (exp_avg_sq_rows / bias2).sqrt().add_(self.eps)
                if row_lr_arg is not None:
                    row_lr_view = row_lr_arg.view((-1,) + (1,) * (param_rows.dim() - 1))
                    param_rows.add_((exp_avg_rows / denom) * row_lr_view, alpha=-1.0 / bias1)
                else:
                    step_size = base_lr / bias1
                    param_rows.addcdiv_(exp_avg_rows, denom, value=-step_size)

        if slot_ids_cpu is None:
            _apply_adamw_rows(active_param, grad, exp_avg, exp_avg_sq, row_lr, step_value)
            return

        slot_ids_cpu = slot_ids_cpu.detach().to(device="cpu", dtype=torch.long)
        n = int(slot_ids_cpu.numel())
        if n == 0:
            return

        trace = os.environ.get("NANOCHAT_SPARSE_ADAMW_TRACE", "").strip() not in ("", "0", "false", "False")
        if trace:
            print(
                f"[sparse_adamw] runtime_step={int(self.runtime_step)} param={param_name} "
                f"rows={n} device={active_param.device} chunk={_sparse_adamw_row_chunk_size()}",
                file=sys.stderr,
                flush=True,
            )

        chunk = _sparse_adamw_row_chunk_size()
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            ids_cpu = slot_ids_cpu[start:end]
            if trace and start == 0:
                print(
                    f"[sparse_adamw]   -> chunk [{start}:{end}) ids_cpu_numel={ids_cpu.numel()}",
                    file=sys.stderr,
                    flush=True,
                )
            slot_ids = ids_cpu.to(active_param.device, non_blocking=True)
            param_rows = active_param.index_select(0, slot_ids)
            grad_rows = grad.index_select(0, slot_ids)
            exp_avg_rows = exp_avg.index_select(0, slot_ids)
            exp_avg_sq_rows = exp_avg_sq.index_select(0, slot_ids)
            row_chunk = row_lr[start:end] if row_lr is not None else None
            if isinstance(step_value, torch.Tensor):
                step_chunk = step_value[start:end]
            else:
                step_chunk = step_value
            _apply_adamw_rows(param_rows, grad_rows, exp_avg_rows, exp_avg_sq_rows, row_chunk, step_chunk)
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
            exposed_count = int(grad.size(0))
            target_grad[:exposed_count].copy_(grad.to(dtype=target_grad.dtype))
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
        grad_accum_warm_ids_cpu = self._grad_accum_warm_ids_cpu if self._grad_accum_warm_ids_cpu is not None else torch.empty(0, dtype=torch.long)
        grad_accum_cold_ids_cpu = self._grad_accum_cold_ids_cpu if self._grad_accum_cold_ids_cpu is not None else torch.empty(0, dtype=torch.long)
        live_count = int(grad_accum_ids_cpu.numel())
        live_slot_ids_cpu = torch.empty(0, dtype=torch.long)
        live_global_ids_cpu = torch.empty(0, dtype=torch.long)
        live_union_row_ids_cpu = torch.empty(0, dtype=torch.long)
        live_union_mask_cpu = torch.zeros(grad_accum_count, dtype=torch.bool)
        live_lm_head_slot_ids_cpu = torch.empty(0, dtype=torch.long)
        live_lm_head_global_ids_cpu = torch.empty(0, dtype=torch.long)
        live_lm_head_union_row_ids_cpu = torch.empty(0, dtype=torch.long)
        live_lm_head_union_slot_ids_cpu = torch.empty(0, dtype=torch.long)
        live_lm_head_union_global_ids_cpu = torch.empty(0, dtype=torch.long)
        live_lm_head_warm_slot_ids_cpu = torch.empty(0, dtype=torch.long)
        live_lm_head_warm_global_ids_cpu = torch.empty(0, dtype=torch.long)
        live_lm_head_cold_slot_ids_cpu = torch.empty(0, dtype=torch.long)
        live_lm_head_cold_global_ids_cpu = torch.empty(0, dtype=torch.long)
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
                live_lm_head_union_mask_cpu = live_lm_head_union_row_ids_cpu >= 0
                if live_lm_head_union_mask_cpu.any():
                    live_lm_head_union_slot_ids_cpu = live_lm_head_slot_ids_cpu[live_lm_head_union_mask_cpu]
                    live_lm_head_union_global_ids_cpu = live_lm_head_global_ids_cpu[live_lm_head_union_mask_cpu]
                    live_lm_head_union_row_ids_cpu = live_lm_head_union_row_ids_cpu[live_lm_head_union_mask_cpu]
                    live_union_mask_cpu[live_lm_head_union_row_ids_cpu] = True
                else:
                    live_lm_head_union_row_ids_cpu = torch.empty(0, dtype=torch.long)
                live_lm_head_cloud_mask_cpu = ~live_lm_head_union_mask_cpu
                if live_lm_head_cloud_mask_cpu.any():
                    live_lm_head_cloud_slot_ids_cpu = live_lm_head_slot_ids_cpu[live_lm_head_cloud_mask_cpu]
                    live_lm_head_cloud_global_ids_cpu = live_lm_head_global_ids_cpu[live_lm_head_cloud_mask_cpu]
                    warm_cloud_mask_cpu = torch.isin(live_lm_head_cloud_global_ids_cpu, grad_accum_warm_ids_cpu)
                    cold_cloud_mask_cpu = torch.isin(live_lm_head_cloud_global_ids_cpu, grad_accum_cold_ids_cpu)
                    if warm_cloud_mask_cpu.any():
                        live_lm_head_warm_slot_ids_cpu = live_lm_head_cloud_slot_ids_cpu[warm_cloud_mask_cpu]
                        live_lm_head_warm_global_ids_cpu = live_lm_head_cloud_global_ids_cpu[warm_cloud_mask_cpu]
                    if cold_cloud_mask_cpu.any():
                        live_lm_head_cold_slot_ids_cpu = live_lm_head_cloud_slot_ids_cpu[cold_cloud_mask_cpu]
                        live_lm_head_cold_global_ids_cpu = live_lm_head_cloud_global_ids_cpu[cold_cloud_mask_cpu]

        non_live_union_row_ids_cpu = torch.nonzero(~live_union_mask_cpu, as_tuple=False).flatten()
        non_live_grad_accum_ids_cpu = grad_accum_ids_cpu.index_select(0, non_live_union_row_ids_cpu) if non_live_union_row_ids_cpu.numel() > 0 else torch.empty(0, dtype=torch.long)
        cached_non_live_mask_cpu = torch.zeros(non_live_union_row_ids_cpu.numel(), dtype=torch.bool)
        if self._grad_accum_cached_union_mask_cpu is not None and non_live_union_row_ids_cpu.numel() > 0:
            cached_non_live_mask_cpu = self._grad_accum_cached_union_mask_cpu[non_live_union_row_ids_cpu]
        staged_non_live_union_row_ids_cpu = non_live_union_row_ids_cpu[~cached_non_live_mask_cpu]
        staged_non_live_grad_accum_ids_cpu = grad_accum_ids_cpu.index_select(0, staged_non_live_union_row_ids_cpu) if staged_non_live_union_row_ids_cpu.numel() > 0 else torch.empty(0, dtype=torch.long)

        prefetched_apply_stage = None
        if (
            self._pending_apply_stage_future is not None and
            self._pending_apply_stage_ids_cpu is not None and
            torch.equal(self._pending_apply_stage_ids_cpu, staged_non_live_grad_accum_ids_cpu)
        ):
            prefetched_apply_stage = self._pending_apply_stage_future.result()
        self._pending_apply_stage_future = None
        self._pending_apply_stage_ids_cpu = None

        t_stage_start = time.perf_counter()
        non_live_params = {}
        non_live_optimizer_state = {}
        live_slot_ids_device = live_slot_ids_cpu.to(self.device) if live_slot_ids_cpu.numel() > 0 else None
        staged_non_live_union_row_ids_device = staged_non_live_union_row_ids_cpu.to(self.device) if staged_non_live_union_row_ids_cpu.numel() > 0 else None
        for name, spec in self.table_specs.items():
            param = spec["param"]
            state = self.state[param]
            if staged_non_live_grad_accum_ids_cpu.numel() > 0 and staged_non_live_union_row_ids_device is not None:
                if prefetched_apply_stage is not None:
                    prefetched_entry = prefetched_apply_stage[name]
                    rows_gpu, _, _ = self._stage_prefetched_cpu_tensor_to_device(
                        f"apply:{name}:param",
                        prefetched_entry["param"],
                    )
                    exp_avg_gpu, _, _ = self._stage_prefetched_cpu_tensor_to_device(
                        f"apply:{name}:exp_avg",
                        prefetched_entry["exp_avg"],
                    )
                    exp_avg_sq_gpu, _, _ = self._stage_prefetched_cpu_tensor_to_device(
                        f"apply:{name}:exp_avg_sq",
                        prefetched_entry["exp_avg_sq"],
                    )
                else:
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
                live_slot_ids_for_name_cpu = live_lm_head_union_slot_ids_cpu
            has_live_grad = live_slot_ids_for_name_cpu.numel() > 0 and self.fixed_params[name].grad is not None
            has_cached_non_live_grad = any(name in chunk["tables"] for chunk in self._grad_accum_non_live_chunks)
            has_staged_non_live_grad = name in non_live_params and staged_non_live_union_row_ids_device is not None
            has_live_warm_grad = name == "lm_head" and live_lm_head_warm_slot_ids_cpu.numel() > 0 and self.fixed_params[name].grad is not None
            has_live_cold_grad = name == "lm_head" and live_lm_head_cold_slot_ids_cpu.numel() > 0 and self.fixed_params[name].grad is not None
            if not has_live_grad and not has_cached_non_live_grad and not has_staged_non_live_grad and not has_live_warm_grad and not has_live_cold_grad:
                continue
            state["step"] += 1
            if has_live_grad:
                live_step_values_cpu = self._get_token_event_step_values_cpu(
                    live_lm_head_union_global_ids_cpu if name == "lm_head" else live_global_ids_cpu
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
            if has_live_warm_grad:
                warm_step_values_cpu = self._get_token_event_step_values_cpu(live_lm_head_warm_global_ids_cpu)
                self._adamw_update_with_step_(
                    name,
                    self.fixed_params[name],
                    self.fixed_optimizer_state[name],
                    warm_step_values_cpu,
                    slot_ids_cpu=live_lm_head_warm_slot_ids_cpu,
                    lr_override=float(self.table_specs["lm_head"]["lr"]),
                )
            if has_live_cold_grad:
                cold_step_values_cpu = self._get_token_event_step_values_cpu(live_lm_head_cold_global_ids_cpu)
                self._adamw_update_with_step_(
                    name,
                    self.fixed_params[name],
                    self.fixed_optimizer_state[name],
                    cold_step_values_cpu,
                    slot_ids_cpu=live_lm_head_cold_slot_ids_cpu,
                    lr_override=float(self.table_specs["lm_head"]["lr"]),
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
        updated_token_ids = [grad_accum_ids_cpu]
        if grad_accum_warm_ids_cpu.numel() > 0:
            updated_token_ids.append(grad_accum_warm_ids_cpu)
        if grad_accum_cold_ids_cpu.numel() > 0:
            updated_token_ids.append(grad_accum_cold_ids_cpu)
        self._increment_token_event_step_counts_(torch.cat(updated_token_ids))

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
            # cloud/warm/cold/random_fill fields removed (dead code)
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
                    lr_override=float(self.table_specs["lm_head"]["lr"]),
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
                    lr_override=float(self.table_specs["lm_head"]["lr"]),
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
