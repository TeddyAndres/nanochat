from dataclasses import dataclass
import time
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
    bytes_h2d: int
    h2d_ms: float
    bytes_d2h: int = 0
    d2h_ms: float = 0.0
    optimizer_ms: float = 0.0
    d2h_launch_ms: float = 0.0
    d2h_sync_ms: float = 0.0
    cpu_writeback_ms: float = 0.0
    active_param_bytes: int = 0
    active_grad_bytes: int = 0
    active_optimizer_bytes: int = 0


class DynamicVocabRuntime:
    """CPU-master / GPU-active runtime for vocab-dimension tables.

    First-pass constraints:
    - single GPU only
    - grad_accum_steps == 1
    - vocab tables are offloaded to CPU masters between optimizer steps
    - optimizer moments for vocab tables live on CPU and are staged to GPU only for active rows
    """

    def __init__(
        self,
        model,
        device,
        embedding_lr,
        unembedding_lr,
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

    def _stage_rows_to_gpu(self, cpu_tensor_map):
        gpu_tensor_map = {}
        for name, tensor in cpu_tensor_map.items():
            staged = tensor.pin_memory() if self.use_cuda else tensor
            gpu_tensor_map[name] = staged.to(self.device, non_blocking=self.use_cuda)
        return gpu_tensor_map

    def _get_cpu_receive_buffer(self, name: str, shape: tuple[int, ...], dtype: torch.dtype):
        buffer = self._cpu_receive_buffers.get(name)
        needs_new = (
            buffer is None or
            buffer.dtype != dtype or
            buffer.dim() != len(shape) or
            any(buffer.size(dim) < shape_dim for dim, shape_dim in enumerate(shape))
        )
        if needs_new:
            buffer = torch.empty(shape, dtype=dtype, pin_memory=self.use_cuda)
            self._cpu_receive_buffers[name] = buffer
        assert buffer is not None
        slices = tuple(slice(0, dim) for dim in shape)
        return buffer[slices]

    def prepare_step(self, active_ids_cpu: torch.Tensor) -> DynamicVocabStep:
        active_ids_cpu = active_ids_cpu.detach().to(device="cpu", dtype=torch.long)
        bytes_h2d = 0
        active_param_bytes = 0
        active_optimizer_bytes = 0
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
            bytes_h2d += rows.numel() * rows.element_size()
            bytes_h2d += exp_avg.numel() * exp_avg.element_size()
            bytes_h2d += exp_avg_sq.numel() * exp_avg_sq.element_size()
            active_param_bytes += rows.numel() * rows.element_size()
            active_optimizer_bytes += exp_avg.numel() * exp_avg.element_size()
            active_optimizer_bytes += exp_avg_sq.numel() * exp_avg_sq.element_size()

        t0 = time.perf_counter()
        gpu_rows = self._stage_rows_to_gpu(cpu_rows)
        gpu_exp_avg = self._stage_rows_to_gpu(cpu_exp_avg)
        gpu_exp_avg_sq = self._stage_rows_to_gpu(cpu_exp_avg_sq)
        if self.use_cuda:
            torch.cuda.synchronize(self.device)
        h2d_ms = (time.perf_counter() - t0) * 1000.0

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
            bytes_h2d=bytes_h2d,
            h2d_ms=h2d_ms,
            active_param_bytes=active_param_bytes,
            active_optimizer_bytes=active_optimizer_bytes,
        )

    def _adamw_update_(self, param_name: str, active_param: nn.Parameter, active_state: dict) -> None:
        grad = active_param.grad
        if grad is None:
            return
        spec = self.table_specs[param_name]
        state = self.state[spec["param"]]
        state["step"] += 1
        exp_avg = active_state["exp_avg"]
        exp_avg_sq = active_state["exp_avg_sq"]
        if self.weight_decay != 0.0:
            active_param.mul_(1 - spec["lr"] * self.weight_decay)
        exp_avg.lerp_(grad, 1 - self.beta1)
        exp_avg_sq.lerp_(grad.square(), 1 - self.beta2)
        bias1 = 1 - self.beta1 ** state["step"]
        bias2 = 1 - self.beta2 ** state["step"]
        denom = (exp_avg_sq / bias2).sqrt().add_(self.eps)
        step_size = spec["lr"] / bias1
        active_param.addcdiv_(exp_avg, denom, value=-step_size)

    @torch.no_grad()
    def step(self, step_ctx: DynamicVocabStep) -> DynamicVocabStep:
        assert step_ctx.active_vocab is not None
        assert step_ctx.optimizer_state is not None
        if self.use_cuda:
            torch.cuda.synchronize(self.device)
        t_opt = time.perf_counter()
        self._adamw_update_("wte", step_ctx.active_vocab["wte"], step_ctx.optimizer_state["wte"])
        self._adamw_update_("lm_head", step_ctx.active_vocab["lm_head"], step_ctx.optimizer_state["lm_head"])
        for layer_name, active_param in step_ctx.active_vocab["value_embeds"].items():
            param_name = f"value_embeds.{layer_name}"
            self._adamw_update_(param_name, active_param, step_ctx.optimizer_state[param_name])
        if self.use_cuda:
            torch.cuda.synchronize(self.device)
        step_ctx.optimizer_ms = (time.perf_counter() - t_opt) * 1000.0

        cpu_rows = {}
        cpu_exp_avg = {}
        cpu_exp_avg_sq = {}
        bytes_d2h = 0
        active_grad_bytes = 0
        t_launch = time.perf_counter()
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
            bytes_d2h += rows.numel() * rows.element_size()
            bytes_d2h += exp_avg.numel() * exp_avg.element_size()
            bytes_d2h += exp_avg_sq.numel() * exp_avg_sq.element_size()
            if active_param.grad is not None:
                active_grad_bytes += active_param.grad.numel() * active_param.grad.element_size()
        step_ctx.d2h_launch_ms = (time.perf_counter() - t_launch) * 1000.0
        t_sync = time.perf_counter()
        if self.use_cuda:
            torch.cuda.synchronize(self.device)
        step_ctx.d2h_sync_ms = (time.perf_counter() - t_sync) * 1000.0

        t_writeback = time.perf_counter()
        for name, spec in self.table_specs.items():
            param = spec["param"]
            state = self.state[param]
            param.index_copy_(0, step_ctx.active_ids_cpu, cpu_rows[name])
            state["exp_avg"].index_copy_(0, step_ctx.active_ids_cpu, cpu_exp_avg[name])
            state["exp_avg_sq"].index_copy_(0, step_ctx.active_ids_cpu, cpu_exp_avg_sq[name])
        step_ctx.cpu_writeback_ms = (time.perf_counter() - t_writeback) * 1000.0

        step_ctx.bytes_d2h = bytes_d2h
        step_ctx.active_grad_bytes = active_grad_bytes
        step_ctx.d2h_ms = step_ctx.d2h_launch_ms + step_ctx.d2h_sync_ms
        step_ctx.active_vocab = None
        step_ctx.optimizer_state = None
        return step_ctx
