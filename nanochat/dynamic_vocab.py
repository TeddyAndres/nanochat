import math
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn


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
    d2h_launch_ms: float = 0.0
    d2h_sync_ms: float = 0.0
    cpu_writeback_ms: float = 0.0
    active_param_bytes: int = 0
    active_grad_bytes: int = 0
    active_optimizer_bytes: int = 0
    u_capacity: int = 0
    stage_count: int = 0
    writeback_count: int = 0
    active_slot_ids_cpu: Optional[torch.Tensor] = None
    active_mask_cpu: Optional[torch.Tensor] = None
    slot_to_global_cpu: Optional[torch.Tensor] = None
    stage_ids_cpu: Optional[torch.Tensor] = None
    stage_slot_ids_cpu: Optional[torch.Tensor] = None
    writeback_ids_cpu: Optional[torch.Tensor] = None
    writeback_slot_ids_cpu: Optional[torch.Tensor] = None
    sampled_negative_ids_cpu: Optional[torch.Tensor] = None
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
        fixed_u_max=None,
        grad_accum_u_max=None,
        sampled_negative_count=0,
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
        self.sampled_negative_count = int(sampled_negative_count)
        if self.sampled_negative_count < 0:
            raise ValueError(f"sampled_negative_count must be non-negative, got {self.sampled_negative_count}")
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
        self.fixed_slot_to_global_cpu = None
        self.fixed_active_mask_cpu = None
        self._fixed_live_state = False
        self._grad_accum_ids_cpu = None
        self._grad_accum_global_to_local_cpu = None
        self._grad_accum_buffers = None
        self._grad_accum_count = 0
        self._grad_accum_live = False
        self._grad_accum_stage_count = 0
        self._grad_accum_pending_transfers = []
        self._grad_accum_transfer_stream = torch.cuda.Stream(device=self.device) if self.use_cuda else None
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
            self.fixed_active_vocab = {
                "wte": self.fixed_params["wte"],
                "lm_head": self.fixed_params["lm_head"],
                "value_embeds": {
                    name.split(".", 1)[1]: self.fixed_params[name]
                    for name in self.fixed_params
                    if name.startswith("value_embeds.")
                },
                "logit_mask": self.fixed_logit_mask,
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
            "version": 1,
            "tables": serialized,
        }

    def load_state_dict(self, state_dict):
        tables = state_dict.get("tables", {})
        for name, table_state in tables.items():
            spec = self.table_specs[name]
            param = spec["param"]
            state = self.state[param]
            state["step"] = int(table_state["step"])
            state["exp_avg"].copy_(table_state["exp_avg"].to("cpu"))
            state["exp_avg_sq"].copy_(table_state["exp_avg_sq"].to("cpu"))

    @contextmanager
    def materialize_dense_params(self):
        """Temporarily move full vocab tables to the model device for dense eval/inference."""
        original_data = {}
        try:
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
                self._grad_accum_buffers[name] = torch.zeros(shape, dtype=param.dtype, device="cpu")

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

    def _writeback_fixed_rows_(self, global_ids_cpu: torch.Tensor, slot_ids_cpu: torch.Tensor):
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

    def _sample_cold_negative_ids(self, active_ids_cpu: torch.Tensor) -> torch.Tensor:
        if self.sampled_negative_count == 0:
            return torch.empty(0, dtype=torch.long)
        vocab_size = int(self.model.config.vocab_size)
        active_ids_cpu = active_ids_cpu.detach().to(device="cpu", dtype=torch.long)
        cold_mask = torch.ones(vocab_size, dtype=torch.bool)
        cold_mask[active_ids_cpu] = False
        cold_ids = torch.nonzero(cold_mask, as_tuple=False).flatten()
        if cold_ids.numel() < self.sampled_negative_count:
            raise ValueError(
                f"Sparse cold-negative sampling requires at least {self.sampled_negative_count} cold vocab rows, found {cold_ids.numel()}"
            )
        sample_order = torch.randperm(cold_ids.numel())[:self.sampled_negative_count]
        return cold_ids.index_select(0, sample_order)

    def _stage_sampled_negative_state(self, sampled_ids_cpu: torch.Tensor):
        sampled_ids_cpu = sampled_ids_cpu.detach().to(device="cpu", dtype=torch.long)
        if sampled_ids_cpu.numel() == 0:
            return None, None
        spec = self.table_specs["lm_head"]
        param = spec["param"]
        state = self.state[param]
        rows = param.index_select(0, sampled_ids_cpu)
        exp_avg = state["exp_avg"].index_select(0, sampled_ids_cpu)
        exp_avg_sq = state["exp_avg_sq"].index_select(0, sampled_ids_cpu)
        rows_gpu = rows.pin_memory().to(self.device, non_blocking=self.use_cuda) if self.use_cuda else rows.to(self.device)
        exp_avg_gpu = exp_avg.pin_memory().to(self.device, non_blocking=self.use_cuda) if self.use_cuda else exp_avg.to(self.device)
        exp_avg_sq_gpu = exp_avg_sq.pin_memory().to(self.device, non_blocking=self.use_cuda) if self.use_cuda else exp_avg_sq.to(self.device)
        return (
            nn.Parameter(rows_gpu, requires_grad=True),
            {
                "exp_avg": exp_avg_gpu,
                "exp_avg_sq": exp_avg_sq_gpu,
            },
        )

    def _compute_sampled_negative_logit_bias(self, active_ids_cpu: torch.Tensor, sampled_ids_cpu: torch.Tensor) -> float:
        sampled_count = int(sampled_ids_cpu.numel())
        if sampled_count == 0:
            return 0.0
        cold_count = int(self.model.config.vocab_size) - int(active_ids_cpu.numel())
        if cold_count <= sampled_count:
            return 0.0
        return math.log(cold_count / sampled_count)

    def _writeback_sampled_negative_rows_(self, sampled_ids_cpu: torch.Tensor, sampled_param: nn.Parameter, sampled_state: dict):
        sampled_ids_cpu = sampled_ids_cpu.detach().to(device="cpu", dtype=torch.long)
        if sampled_ids_cpu.numel() == 0:
            return
        row_buffer = self._get_cpu_receive_buffer("rows:lm_head_negatives", tuple(sampled_param.shape), sampled_param.dtype)
        exp_avg_buffer = self._get_cpu_receive_buffer("exp_avg:lm_head_negatives", tuple(sampled_state["exp_avg"].shape), sampled_state["exp_avg"].dtype)
        exp_avg_sq_buffer = self._get_cpu_receive_buffer("exp_avg_sq:lm_head_negatives", tuple(sampled_state["exp_avg_sq"].shape), sampled_state["exp_avg_sq"].dtype)
        row_buffer.copy_(sampled_param.detach(), non_blocking=self.use_cuda)
        exp_avg_buffer.copy_(sampled_state["exp_avg"].detach(), non_blocking=self.use_cuda)
        exp_avg_sq_buffer.copy_(sampled_state["exp_avg_sq"].detach(), non_blocking=self.use_cuda)
        if self.use_cuda:
            torch.cuda.synchronize(self.device)
        spec = self.table_specs["lm_head"]
        param = spec["param"]
        state = self.state[param]
        param.index_copy_(0, sampled_ids_cpu, row_buffer)
        state["exp_avg"].index_copy_(0, sampled_ids_cpu, exp_avg_buffer)
        state["exp_avg_sq"].index_copy_(0, sampled_ids_cpu, exp_avg_sq_buffer)

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

    def _prepare_dynamic_step(self, active_ids_cpu: torch.Tensor) -> DynamicVocabStep:
        active_ids_cpu = active_ids_cpu.detach().to(device="cpu", dtype=torch.long)
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
        }
        optimizer_state = {
            name: {
                "exp_avg": gpu_exp_avg[name],
                "exp_avg_sq": gpu_exp_avg_sq[name],
            }
            for name in gpu_rows
        }
        sampled_negative_ids_cpu = self._sample_cold_negative_ids(active_ids_cpu)
        sampled_negative_param, sampled_negative_state = self._stage_sampled_negative_state(sampled_negative_ids_cpu)
        if sampled_negative_param is not None and sampled_negative_state is not None:
            active_vocab["lm_head_negatives"] = sampled_negative_param
            active_vocab["lm_head_negative_logit_bias"] = torch.tensor(
                self._compute_sampled_negative_logit_bias(active_ids_cpu, sampled_negative_ids_cpu),
                device=self.device,
                dtype=gpu_rows["lm_head"].dtype,
            )
            optimizer_state["lm_head_negatives"] = sampled_negative_state
        return DynamicVocabStep(
            active_ids_cpu=active_ids_cpu,
            active_vocab=active_vocab,
            optimizer_state=optimizer_state,
            unique_count=active_ids_cpu.numel(),
            live_count=active_ids_cpu.numel(),
            u_capacity=active_ids_cpu.numel(),
            stage_count=active_ids_cpu.numel(),
            sampled_negative_ids_cpu=sampled_negative_ids_cpu,
        )

    def _prepare_fixed_step(self, step_meta: dict) -> DynamicVocabStep:
        assert self.fixed_u_mode, "Fixed-U step requested without fixed_u_max runtime configuration"
        active_ids_cpu = step_meta["active_ids_cpu"].detach().to(device="cpu", dtype=torch.long)
        grad_accum_ids_cpu = step_meta.get("grad_accum_ids_cpu", active_ids_cpu).detach().to(device="cpu", dtype=torch.long)
        grad_accum_steps = int(step_meta.get("grad_accum_steps", 1))
        grad_accum_micro_step = int(step_meta.get("grad_accum_micro_step", 0))
        is_grad_accum_boundary = bool(step_meta.get("is_grad_accum_boundary", True))
        active_slot_ids_cpu = step_meta["active_slot_ids_cpu"].detach().to(device="cpu", dtype=torch.long)
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

        if not self._fixed_live_state:
            stage_ids_cpu = active_ids_cpu
            stage_slot_ids_cpu = active_slot_ids_cpu

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
        self._fixed_live_state = True

        if grad_accum_steps > 1 and self.sampled_negative_count > 0:
            raise ValueError("Sparse grad accumulation does not yet support sampled cold negatives; set --sparse-cold-negative-count 0")
        sampled_negative_ids_cpu = self._sample_cold_negative_ids(active_ids_cpu)
        sampled_negative_param, sampled_negative_state = self._stage_sampled_negative_state(sampled_negative_ids_cpu)
        step_active_vocab = {
            **self.fixed_active_vocab,
            "value_embeds": self.fixed_active_vocab["value_embeds"],
        }
        step_optimizer_state = {
            **self.fixed_optimizer_state,
        }
        if sampled_negative_param is not None and sampled_negative_state is not None:
            step_active_vocab["lm_head_negatives"] = sampled_negative_param
            step_active_vocab["lm_head_negative_logit_bias"] = torch.tensor(
                self._compute_sampled_negative_logit_bias(active_ids_cpu, sampled_negative_ids_cpu),
                device=self.device,
                dtype=self.fixed_params["lm_head"].dtype,
            )
            step_optimizer_state["lm_head_negatives"] = sampled_negative_state

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
            sampled_negative_ids_cpu=sampled_negative_ids_cpu,
            grad_accum_ids_cpu=grad_accum_ids_cpu,
            grad_accum_steps=grad_accum_steps,
            grad_accum_micro_step=grad_accum_micro_step,
            is_grad_accum_boundary=is_grad_accum_boundary,
            is_last_step=is_last_step,
            fixed_u_mode=True,
        )

    def prepare_step(self, active_ids_cpu) -> DynamicVocabStep:
        if isinstance(active_ids_cpu, dict):
            return self._prepare_fixed_step(active_ids_cpu)
        return self._prepare_dynamic_step(active_ids_cpu)

    def _adamw_update_(self, param_name: str, active_param: nn.Parameter, active_state: dict, slot_ids_cpu: Optional[torch.Tensor] = None) -> None:
        grad = active_param.grad
        if grad is None:
            return
        spec = self.table_specs[param_name]
        state = self.state[spec["param"]]
        state["step"] += 1
        self._adamw_update_with_step_(param_name, active_param, active_state, state["step"], slot_ids_cpu=slot_ids_cpu)

    def _adamw_update_with_step_(self, param_name: str, active_param: nn.Parameter, active_state: dict, step_value: int, slot_ids_cpu: Optional[torch.Tensor] = None) -> None:
        grad = active_param.grad
        if grad is None:
            return
        spec = self.table_specs[param_name]
        exp_avg = active_state["exp_avg"]
        exp_avg_sq = active_state["exp_avg_sq"]
        if slot_ids_cpu is None:
            if self.weight_decay != 0.0:
                active_param.mul_(1 - spec["lr"] * self.weight_decay)
            exp_avg.lerp_(grad, 1 - self.beta1)
            exp_avg_sq.lerp_(grad.square(), 1 - self.beta2)
            bias1 = 1 - self.beta1 ** step_value
            bias2 = 1 - self.beta2 ** step_value
            denom = (exp_avg_sq / bias2).sqrt().add_(self.eps)
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
        step_size = spec["lr"] / bias1
        param_rows.addcdiv_(exp_avg_rows, denom, value=-step_size)
        active_param.index_copy_(0, slot_ids, param_rows)
        exp_avg.index_copy_(0, slot_ids, exp_avg_rows)
        exp_avg_sq.index_copy_(0, slot_ids, exp_avg_sq_rows)

    @torch.no_grad()
    def accumulate_gradients(self, step_ctx: DynamicVocabStep) -> DynamicVocabStep:
        assert step_ctx.fixed_u_mode, "Sparse grad accumulation currently supports fixed-U mode only"
        assert step_ctx.active_slot_ids_cpu is not None
        assert step_ctx.active_ids_cpu is not None
        if step_ctx.grad_accum_ids_cpu is None:
            raise ValueError("Sparse grad accumulation requires grad_accum_ids_cpu metadata")
        if step_ctx.sampled_negative_ids_cpu is not None and step_ctx.sampled_negative_ids_cpu.numel() > 0:
            raise ValueError("Sparse grad accumulation does not yet support sampled cold negatives")

        if (not self._grad_accum_live) or self._grad_accum_ids_cpu is None or not torch.equal(self._grad_accum_ids_cpu, step_ctx.grad_accum_ids_cpu):
            self._start_grad_accum_window(step_ctx.grad_accum_ids_cpu)

        assert self._grad_accum_global_to_local_cpu is not None
        assert self._grad_accum_buffers is not None
        self._flush_pending_grad_accum_transfers(wait=False)
        accum_row_ids_cpu = self._grad_accum_global_to_local_cpu[step_ctx.active_ids_cpu]
        if (accum_row_ids_cpu < 0).any():
            raise ValueError("Sparse grad accumulation map is missing active vocab rows")
        active_slot_ids_device = step_ctx.active_slot_ids_cpu.to(self.device)

        grad_map = {}
        for name in self.table_specs:
            if name == "wte":
                grad = self.fixed_params["wte"].grad
            elif name == "lm_head":
                grad = self.fixed_params["lm_head"].grad
            else:
                grad = self.fixed_params[name].grad
            if grad is None:
                continue
            grad_map[name] = grad

        self._queue_grad_accum_transfer_(accum_row_ids_cpu, active_slot_ids_device, grad_map)

        self._clear_fixed_grads()
        step_ctx.active_vocab = None
        step_ctx.optimizer_state = None
        self._grad_accum_stage_count += int(step_ctx.stage_count)
        step_ctx.unique_count = int(step_ctx.grad_accum_ids_cpu.numel())
        step_ctx.live_count = int(step_ctx.active_ids_cpu.numel())
        return step_ctx

    @torch.no_grad()
    def apply_accumulated_gradients(self) -> DynamicVocabStep:
        if not self._grad_accum_live or self._grad_accum_ids_cpu is None or self._grad_accum_buffers is None:
            raise ValueError("No sparse accumulated gradients are pending")

        self._flush_pending_grad_accum_transfers(wait=True)

        grad_accum_ids_cpu = self._grad_accum_ids_cpu
        grad_accum_count = self._grad_accum_count
        cpu_rows = {}
        cpu_exp_avg = {}
        cpu_exp_avg_sq = {}
        for name, spec in self.table_specs.items():
            param = spec["param"]
            state = self.state[param]
            rows = param.index_select(0, grad_accum_ids_cpu)
            exp_avg = state["exp_avg"].index_select(0, grad_accum_ids_cpu)
            exp_avg_sq = state["exp_avg_sq"].index_select(0, grad_accum_ids_cpu)
            cpu_rows[name] = rows
            cpu_exp_avg[name] = exp_avg
            cpu_exp_avg_sq[name] = exp_avg_sq

        gpu_rows = self._stage_rows_to_gpu(cpu_rows)
        gpu_exp_avg = self._stage_rows_to_gpu(cpu_exp_avg)
        gpu_exp_avg_sq = self._stage_rows_to_gpu(cpu_exp_avg_sq)

        active_params = {
            name: nn.Parameter(gpu_rows[name], requires_grad=True)
            for name in gpu_rows
        }
        optimizer_state = {
            name: {
                "exp_avg": gpu_exp_avg[name],
                "exp_avg_sq": gpu_exp_avg_sq[name],
            }
            for name in gpu_rows
        }
        for name in self.table_specs:
            active_params[name].grad = self._grad_accum_buffers[name][:grad_accum_count].to(self.device, non_blocking=self.use_cuda)
            self._adamw_update_(name, active_params[name], optimizer_state[name])

        live_count = int(grad_accum_ids_cpu.numel())
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
                live_slot_ids = live_slot_ids_cpu.to(self.device)
                live_union_row_ids = live_union_row_ids_cpu.to(self.device)
                for name in self.table_specs:
                    self.fixed_params[name].data.index_copy_(
                        0,
                        live_slot_ids,
                        active_params[name].detach().index_select(0, live_union_row_ids),
                    )
                    self.fixed_optimizer_state[name]["exp_avg"].index_copy_(
                        0,
                        live_slot_ids,
                        optimizer_state[name]["exp_avg"].detach().index_select(0, live_union_row_ids),
                    )
                    self.fixed_optimizer_state[name]["exp_avg_sq"].index_copy_(
                        0,
                        live_slot_ids,
                        optimizer_state[name]["exp_avg_sq"].detach().index_select(0, live_union_row_ids),
                    )

        for name, spec in self.table_specs.items():
            param = spec["param"]
            state = self.state[param]
            row_buffer = self._get_cpu_receive_buffer(f"rows:accum:{name}", tuple(active_params[name].shape), active_params[name].dtype)
            exp_avg_buffer = self._get_cpu_receive_buffer(f"exp_avg:accum:{name}", tuple(optimizer_state[name]["exp_avg"].shape), optimizer_state[name]["exp_avg"].dtype)
            exp_avg_sq_buffer = self._get_cpu_receive_buffer(f"exp_avg_sq:accum:{name}", tuple(optimizer_state[name]["exp_avg_sq"].shape), optimizer_state[name]["exp_avg_sq"].dtype)
            row_buffer.copy_(active_params[name].detach(), non_blocking=self.use_cuda)
            exp_avg_buffer.copy_(optimizer_state[name]["exp_avg"].detach(), non_blocking=self.use_cuda)
            exp_avg_sq_buffer.copy_(optimizer_state[name]["exp_avg_sq"].detach(), non_blocking=self.use_cuda)
            if self.use_cuda:
                torch.cuda.synchronize(self.device)
            param.index_copy_(0, grad_accum_ids_cpu, row_buffer)
            state["exp_avg"].index_copy_(0, grad_accum_ids_cpu, exp_avg_buffer)
            state["exp_avg_sq"].index_copy_(0, grad_accum_ids_cpu, exp_avg_sq_buffer)

        metrics = DynamicVocabStep(
            active_ids_cpu=grad_accum_ids_cpu,
            active_vocab=None,
            optimizer_state=None,
            unique_count=int(grad_accum_ids_cpu.numel()),
            live_count=live_count,
            u_capacity=self.grad_accum_u_max if self.grad_accum_u_max > 0 else int(grad_accum_ids_cpu.numel()),
            stage_count=int(self._grad_accum_stage_count),
            writeback_count=int(grad_accum_ids_cpu.numel()),
            fixed_u_mode=self.fixed_u_mode,
        )
        self._clear_fixed_grads()
        self._clear_grad_accum_window()
        return metrics

    @torch.no_grad()
    def step(self, step_ctx: DynamicVocabStep) -> DynamicVocabStep:
        assert step_ctx.active_vocab is not None
        assert step_ctx.optimizer_state is not None
        if step_ctx.fixed_u_mode:
            assert step_ctx.active_slot_ids_cpu is not None
            self._adamw_update_("wte", self.fixed_params["wte"], self.fixed_optimizer_state["wte"], slot_ids_cpu=step_ctx.active_slot_ids_cpu)
            self._adamw_update_("lm_head", self.fixed_params["lm_head"], self.fixed_optimizer_state["lm_head"], slot_ids_cpu=step_ctx.active_slot_ids_cpu)
            if step_ctx.sampled_negative_ids_cpu is not None and step_ctx.sampled_negative_ids_cpu.numel() > 0:
                self._adamw_update_with_step_(
                    "lm_head",
                    step_ctx.active_vocab["lm_head_negatives"],
                    step_ctx.optimizer_state["lm_head_negatives"],
                    self.state[self.table_specs["lm_head"]["param"]]["step"],
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
            if step_ctx.sampled_negative_ids_cpu is not None and step_ctx.sampled_negative_ids_cpu.numel() > 0:
                self._writeback_sampled_negative_rows_(
                    step_ctx.sampled_negative_ids_cpu,
                    step_ctx.active_vocab["lm_head_negatives"],
                    step_ctx.optimizer_state["lm_head_negatives"],
                )
            self._clear_fixed_grads()
            step_ctx.active_vocab = None
            step_ctx.optimizer_state = None
            return step_ctx

        self._adamw_update_("wte", step_ctx.active_vocab["wte"], step_ctx.optimizer_state["wte"])
        self._adamw_update_("lm_head", step_ctx.active_vocab["lm_head"], step_ctx.optimizer_state["lm_head"])
        if step_ctx.sampled_negative_ids_cpu is not None and step_ctx.sampled_negative_ids_cpu.numel() > 0:
            self._adamw_update_with_step_(
                "lm_head",
                step_ctx.active_vocab["lm_head_negatives"],
                step_ctx.optimizer_state["lm_head_negatives"],
                self.state[self.table_specs["lm_head"]["param"]]["step"],
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
        if step_ctx.sampled_negative_ids_cpu is not None and step_ctx.sampled_negative_ids_cpu.numel() > 0:
            self._writeback_sampled_negative_rows_(
                step_ctx.sampled_negative_ids_cpu,
                step_ctx.active_vocab["lm_head_negatives"],
                step_ctx.optimizer_state["lm_head_negatives"],
            )
        step_ctx.active_vocab = None
        step_ctx.optimizer_state = None
        return step_ctx
