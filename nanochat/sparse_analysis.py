from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Optional

import torch
import torch.nn.functional as F

from nanochat.token_cache import resolve_token_cache_dir


CORRECT_RECORD_COLS = 7
INCORRECT_RECORD_COLS = 9


def _record_key_columns(records: torch.Tensor) -> tuple[int, ...]:
    if records.size(-1) == CORRECT_RECORD_COLS:
        return (6,)
    if records.size(-1) == INCORRECT_RECORD_COLS:
        return (6, 8)
    raise ValueError(f"Unsupported sparse record width {records.size(-1)}")


def resolve_sparse_analysis_dir(output_dir: str | Path | None, token_cache_dir: str | Path | None) -> Path:
    if output_dir is None or str(output_dir).strip() == "":
        cache_dir = resolve_token_cache_dir(token_cache_dir)
        return cache_dir.parent / f"{cache_dir.name}_sparse_analysis"
    return resolve_token_cache_dir(output_dir)


def _empty_records(topk: int, cols: int, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.full((topk,), -float("inf"), dtype=torch.float32, device=device)
    records = torch.full((topk, cols), -1, dtype=torch.long, device=device)
    return scores, records


def _topk_from_candidates(
    candidate_scores: torch.Tensor,
    candidate_records: torch.Tensor,
    *,
    topk: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if candidate_scores.numel() == 0:
        if topk is None:
            return candidate_scores, candidate_records
        return _empty_records(topk, candidate_records.size(-1), device=candidate_records.device)
    if topk is None:
        top_scores, top_idx = torch.sort(candidate_scores, descending=True)
        return top_scores, candidate_records.index_select(0, top_idx)
    keep = min(int(topk), int(candidate_scores.numel()))
    top_scores, top_idx = torch.topk(candidate_scores, k=keep, largest=True, sorted=True)
    top_records = candidate_records.index_select(0, top_idx)
    if keep == topk:
        return top_scores, top_records
    pad_scores, pad_records = _empty_records(topk - keep, candidate_records.size(-1), device=candidate_records.device)
    return torch.cat((top_scores, pad_scores), dim=0), torch.cat((top_records, pad_records), dim=0)


def aggregate_sparse_loss_candidates(
    candidate_scores: torch.Tensor,
    candidate_records: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if candidate_scores.numel() == 0:
        return candidate_scores, candidate_records
    key_columns = _record_key_columns(candidate_records)
    key_tensor = candidate_records[:, key_columns]
    if key_tensor.ndim == 1:
        key_tensor = key_tensor.unsqueeze(1)
    unique_keys, inverse = torch.unique(key_tensor, dim=0, sorted=False, return_inverse=True)
    aggregated_scores = torch.zeros(unique_keys.size(0), dtype=torch.float32, device=candidate_scores.device)
    aggregated_scores.index_add_(0, inverse, candidate_scores.to(dtype=torch.float32))
    positions = torch.arange(candidate_scores.numel(), device=candidate_scores.device, dtype=torch.long)
    representative_positions = torch.full((unique_keys.size(0),), candidate_scores.numel(), dtype=torch.long, device=candidate_scores.device)
    representative_positions.scatter_reduce_(0, inverse, positions, reduce="amin", include_self=True)
    aggregated_records = candidate_records.index_select(0, representative_positions)
    return aggregated_scores, aggregated_records


def select_topk_records(
    candidate_scores: torch.Tensor,
    candidate_records: torch.Tensor,
    *,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid_mask = torch.isfinite(candidate_scores)
    if valid_mask.any():
        candidate_scores = candidate_scores[valid_mask]
        candidate_records = candidate_records[valid_mask]
    else:
        return _empty_records(topk, candidate_records.size(-1), device=candidate_records.device)
    aggregated_scores, aggregated_records = aggregate_sparse_loss_candidates(candidate_scores, candidate_records)
    return _topk_from_candidates(aggregated_scores, aggregated_records, topk=topk)


def merge_topk_records(
    current_scores: Optional[torch.Tensor],
    current_records: Optional[torch.Tensor],
    new_scores: torch.Tensor,
    new_records: torch.Tensor,
    *,
    topk: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if current_scores is None or current_records is None:
        aggregated_scores, aggregated_records = aggregate_sparse_loss_candidates(new_scores, new_records)
        return _topk_from_candidates(aggregated_scores, aggregated_records, topk=topk)
    candidate_scores = torch.cat((current_scores, new_scores), dim=0)
    candidate_records = torch.cat((current_records, new_records), dim=0)
    valid_mask = torch.isfinite(candidate_scores)
    candidate_scores = candidate_scores[valid_mask]
    candidate_records = candidate_records[valid_mask]
    aggregated_scores, aggregated_records = aggregate_sparse_loss_candidates(candidate_scores, candidate_records)
    return _topk_from_candidates(aggregated_scores, aggregated_records, topk=topk)


def collect_sparse_loss_topk(
    logits: torch.Tensor,
    targets: torch.Tensor,
    active_global_ids_cpu: torch.Tensor,
    *,
    topk_correct: int | None,
    topk_incorrect: int | None,
    step: int,
    micro_step: int,
    sequence_id: int = -1,
    losses: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if logits.ndim != 3 or targets.ndim != 2:
        raise ValueError("Expected logits shape (B, T, U) and targets shape (B, T)")
    if logits.shape[:2] != targets.shape:
        raise ValueError("Logits and targets batch dimensions must match")
    device = logits.device
    active_global_ids = active_global_ids_cpu.to(device=device, dtype=torch.long)
    if losses is None:
        losses = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
            reduction="none",
        ).view_as(targets)
    else:
        if tuple(losses.shape) != tuple(targets.shape):
            raise ValueError("Precomputed sparse token losses must match target shape")
        losses = losses.to(device=device, dtype=torch.float32)
    safe_targets = targets.clamp_min(0)
    target_logits = logits.gather(2, safe_targets.unsqueeze(-1)).squeeze(-1)
    top2_logits, top2_local = torch.topk(logits, k=min(2, logits.size(-1)), dim=-1)
    if top2_logits.size(-1) < 2:
        pad_shape = (*top2_logits.shape[:2], 2 - top2_logits.size(-1))
        top2_logits = torch.cat(
            (
                top2_logits,
                torch.full(pad_shape, -float("inf"), dtype=top2_logits.dtype, device=device),
            ),
            dim=-1,
        )
        top2_local = torch.cat(
            (
                top2_local,
                torch.full(pad_shape, -1, dtype=top2_local.dtype, device=device),
            ),
            dim=-1,
        )
    return collect_sparse_loss_topk_from_stats(
        targets,
        active_global_ids_cpu,
        topk_correct=topk_correct,
        topk_incorrect=topk_incorrect,
        step=step,
        micro_step=micro_step,
        sequence_id=sequence_id,
        losses=losses,
        top2_logits=top2_logits,
        top2_local=top2_local,
        target_logits=target_logits,
    )


def collect_sparse_loss_topk_from_stats(
    targets: torch.Tensor,
    active_global_ids_cpu: torch.Tensor,
    *,
    topk_correct: int | None,
    topk_incorrect: int | None,
    step: int,
    micro_step: int,
    sequence_id: int = -1,
    losses: torch.Tensor,
    top2_logits: torch.Tensor,
    top2_local: torch.Tensor,
    target_logits: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if targets.ndim != 2:
        raise ValueError("Expected targets shape (B, T)")
    if tuple(losses.shape) != tuple(targets.shape):
        raise ValueError("Precomputed sparse token losses must match target shape")
    if tuple(target_logits.shape) != tuple(targets.shape):
        raise ValueError("Target logits must match target shape")
    if top2_logits.shape[:2] != targets.shape or top2_local.shape[:2] != targets.shape:
        raise ValueError("Top-k sparse analysis tensors must match target batch dimensions")
    if top2_logits.size(-1) != top2_local.size(-1):
        raise ValueError("Top-k sparse analysis logits and ids must have matching widths")

    device = losses.device
    active_global_ids = active_global_ids_cpu.to(device=device, dtype=torch.long)
    valid_mask = targets != -1
    if not valid_mask.any():
        correct_scores, correct_records = _empty_records(topk_correct, CORRECT_RECORD_COLS, device=device)
        incorrect_scores, incorrect_records = _empty_records(topk_incorrect, INCORRECT_RECORD_COLS, device=device)
        return {
            "correct_scores": correct_scores,
            "correct_records": correct_records,
            "incorrect_scores": incorrect_scores,
            "incorrect_records": incorrect_records,
        }

    flat_targets = targets.view(-1)
    flat_losses = losses.to(device=device, dtype=torch.float32).view(-1)
    flat_target_logits = target_logits.to(device=device, dtype=torch.float32).view(-1)
    flat_top2_logits = top2_logits.to(device=device, dtype=torch.float32).view(-1, top2_logits.size(-1))
    flat_top2_local = top2_local.to(device=device, dtype=torch.long).view(-1, top2_local.size(-1))
    flat_valid = valid_mask.view(-1)
    flat_indices = torch.nonzero(flat_valid, as_tuple=False).flatten()
    valid_targets = flat_targets[flat_valid]
    valid_losses = flat_losses[flat_valid]
    valid_target_logits = flat_target_logits[flat_valid]
    valid_top2_logits = flat_top2_logits[flat_valid]
    valid_top2_local = flat_top2_local[flat_valid]

    pred_local = valid_top2_local[:, 0]
    underpred_mask = pred_local != valid_targets
    row_idx = flat_indices // targets.size(1)
    pos_idx = flat_indices % targets.size(1)
    target_global = active_global_ids.index_select(0, valid_targets.to(dtype=torch.long))

    correct_candidate_scores = valid_losses[underpred_mask]
    correct_candidate_records = torch.stack(
        (
            torch.full_like(row_idx[underpred_mask], int(step)),
            torch.full_like(row_idx[underpred_mask], int(micro_step)),
            torch.full_like(row_idx[underpred_mask], int(sequence_id)),
            row_idx[underpred_mask].to(dtype=torch.long),
            pos_idx[underpred_mask].to(dtype=torch.long),
            valid_targets[underpred_mask].to(dtype=torch.long),
            target_global[underpred_mask].to(dtype=torch.long),
        ),
        dim=1,
    ) if underpred_mask.any() else torch.empty((0, CORRECT_RECORD_COLS), dtype=torch.long, device=device)
    correct_candidate_scores, correct_candidate_records = aggregate_sparse_loss_candidates(correct_candidate_scores, correct_candidate_records)
    correct_scores, correct_records = _topk_from_candidates(correct_candidate_scores, correct_candidate_records, topk=topk_correct)

    wrong_local = valid_top2_local[:, 0]
    wrong_logit = valid_top2_logits[:, 0]
    if valid_top2_local.size(1) > 1:
        choose_second = wrong_local == valid_targets
        wrong_local = torch.where(choose_second, valid_top2_local[:, 1], wrong_local)
        wrong_logit = torch.where(choose_second, valid_top2_logits[:, 1], wrong_logit)
    valid_wrong_mask = (wrong_local != valid_targets) & (wrong_local >= 0)
    wrong_global = active_global_ids.index_select(0, wrong_local.clamp_min(0).to(dtype=torch.long))
    incorrect_candidate_scores = (wrong_logit - valid_target_logits)[valid_wrong_mask]
    incorrect_candidate_records = torch.stack(
        (
            torch.full_like(row_idx[valid_wrong_mask], int(step)),
            torch.full_like(row_idx[valid_wrong_mask], int(micro_step)),
            torch.full_like(row_idx[valid_wrong_mask], int(sequence_id)),
            row_idx[valid_wrong_mask].to(dtype=torch.long),
            pos_idx[valid_wrong_mask].to(dtype=torch.long),
            valid_targets[valid_wrong_mask].to(dtype=torch.long),
            target_global[valid_wrong_mask].to(dtype=torch.long),
            wrong_local[valid_wrong_mask].to(dtype=torch.long),
            wrong_global[valid_wrong_mask].to(dtype=torch.long),
        ),
        dim=1,
    ) if valid_wrong_mask.any() else torch.empty((0, INCORRECT_RECORD_COLS), dtype=torch.long, device=device)
    incorrect_candidate_scores, incorrect_candidate_records = aggregate_sparse_loss_candidates(incorrect_candidate_scores, incorrect_candidate_records)
    incorrect_scores, incorrect_records = _topk_from_candidates(incorrect_candidate_scores, incorrect_candidate_records, topk=topk_incorrect)

    return {
        "correct_scores": correct_scores,
        "correct_records": correct_records,
        "incorrect_scores": incorrect_scores,
        "incorrect_records": incorrect_records,
    }


class SparseLossAnalysisWriter:
    def __init__(
        self,
        output_dir: str | Path | None,
        token_cache_dir: str | Path | None,
        accumulate_steps: int = 10,
    ):
        self.output_dir = resolve_sparse_analysis_dir(output_dir, token_cache_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pending = []
        self._accumulate_steps = max(1, int(accumulate_steps))
        self._step_buffer: list[tuple[int, dict]] = []
        self._lock = Lock()

    def submit(self, step: int, payload: dict[str, torch.Tensor | int]) -> None:
        cpu_payload = {}
        for key, value in payload.items():
            if isinstance(value, torch.Tensor):
                cpu_payload[key] = value.detach().to(device="cpu")
            else:
                cpu_payload[key] = value
        with self._lock:
            self._step_buffer.append((int(step), cpu_payload))
            if len(self._step_buffer) >= self._accumulate_steps:
                self._commit_buffer_locked()

    def _commit_buffer_locked(self) -> None:
        """Must be called with self._lock held."""
        if not self._step_buffer:
            return
        batch = self._step_buffer
        self._step_buffer = []
        first_step = batch[0][0]
        last_step = batch[-1][0]
        output_path = self.output_dir / f"steps_{first_step:06d}_{last_step:06d}.pt"
        save_data = {step: p for step, p in batch}
        self._pending.append(self._executor.submit(torch.save, save_data, output_path))
        if len(self._pending) > 8:
            future = self._pending.pop(0)
            future.result()

    def flush(self) -> None:
        with self._lock:
            self._commit_buffer_locked()
        while self._pending:
            self._pending.pop(0).result()

    def close(self) -> None:
        self.flush()
        self._executor.shutdown(wait=True)
