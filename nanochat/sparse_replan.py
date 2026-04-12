from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from nanochat.sparse_analysis import CORRECT_RECORD_COLS, INCORRECT_RECORD_COLS
from nanochat.sparse_window_accum import SparseRollingLossAccumulator
from nanochat.sparse_manifest import (
    load_sequence_manifest_shard,
    load_sparse_manifest_header,
    resolve_grouping_base_manifest_path,
    stream_sparse_manifest_steps,
)


def _dedupe_preserve_order(values: list[int]) -> list[int]:
    seen: set[int] = set()
    result: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(int(value))
    return result


@dataclass(frozen=True)
class SequenceUnitView:
    sequence_id: int
    unique_token_ids: tuple[int, ...]


def load_sequence_unit_views(base_manifest_path: str | Path) -> dict[int, SequenceUnitView]:
    base_manifest_path = Path(base_manifest_path)
    header = load_sparse_manifest_header(base_manifest_path)
    shards = header.get("shards")
    if not isinstance(shards, list) or len(shards) == 0:
        raise ValueError("Sequence base manifest must define shards")
    sequence_units: dict[int, SequenceUnitView] = {}
    for shard_entry in shards:
        shard_path = base_manifest_path.parent / str(shard_entry["path"])
        shard_payload = load_sequence_manifest_shard(shard_path)
        for unit in shard_payload.get("sequence_units", []):
            sequence_id = int(unit["sequence_id"])
            unique_token_ids = tuple(int(token_id) for token_id in unit.get("unique_token_ids", []))
            if len(unique_token_ids) == 0:
                raise ValueError(f"Sequence unit {sequence_id} must define non-empty unique_token_ids")
            sequence_units[sequence_id] = SequenceUnitView(
                sequence_id=sequence_id,
                unique_token_ids=unique_token_ids,
            )
    return sequence_units


class SparseFutureWindowPlanner:
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        lookahead_steps: int = 2,
        window_steps: int = 8,
        rolling_window_steps: int = 20,
        positive_fraction: float = 0.25,
        corrective_fraction: float = 0.25,
        max_auto_negatives_per_microstep: int = 4,
        sampling_seed: int = 0,
    ):
        self.manifest_path = Path(manifest_path)
        self.grouping_header = load_sparse_manifest_header(self.manifest_path)
        self.base_manifest_path = resolve_grouping_base_manifest_path(self.manifest_path, header=self.grouping_header)
        self.sequence_units = load_sequence_unit_views(self.base_manifest_path)
        self.num_steps = int(self.grouping_header["num_steps"])
        self.grad_accum_steps = int(self.grouping_header["grad_accum_steps"])
        self.u_max = int(self.grouping_header["u_max"])
        self.grad_accum_u_max = int(self.grouping_header.get("grad_accum_u_max", self.u_max))
        self.lookahead_steps = max(1, int(lookahead_steps))
        self.window_steps = max(1, int(window_steps))
        self.positive_fraction = float(positive_fraction)
        self.corrective_fraction = float(corrective_fraction)
        self.max_auto_negatives_per_microstep = max(0, int(max_auto_negatives_per_microstep))
        self.sampling_seed = int(sampling_seed)
        self.accumulator = SparseRollingLossAccumulator(window_steps=rolling_window_steps)
        self._step_overrides: dict[int, dict[str, Any]] = {}

    def get_step_override(self, step: int) -> dict[str, Any] | None:
        return self._step_overrides.get(int(step))

    def prune_consumed(self, current_step: int) -> None:
        stale_steps = [step for step in self._step_overrides if step <= int(current_step)]
        for step in stale_steps:
            del self._step_overrides[step]

    def state_dict(self) -> dict[str, Any]:
        return {
            "step_overrides": self._step_overrides,
            "accumulator": self.accumulator.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._step_overrides = {int(step): value for step, value in state_dict.get("step_overrides", {}).items()}
        accumulator_state = state_dict.get("accumulator")
        if isinstance(accumulator_state, dict):
            self.accumulator.load_state_dict(accumulator_state)

    def update_from_step_payload(
        self,
        step: int,
        *,
        correct_scores: torch.Tensor,
        correct_records: torch.Tensor,
        incorrect_scores: torch.Tensor,
        incorrect_records: torch.Tensor,
    ) -> dict[int, dict[str, Any]]:
        self.accumulator.update_step(
            step,
            correct_scores=correct_scores,
            correct_records=correct_records,
            incorrect_scores=incorrect_scores,
            incorrect_records=incorrect_records,
        )
        return self._plan_from_accumulator(step)

    def update_from_topk(
        self,
        step: int,
        *,
        correct_records: torch.Tensor | None,
        incorrect_records: torch.Tensor | None,
    ) -> dict[int, dict[str, Any]]:
        correct_scores = torch.ones(0 if correct_records is None else correct_records.size(0), dtype=torch.float32)
        incorrect_scores = torch.ones(0 if incorrect_records is None else incorrect_records.size(0), dtype=torch.float32)
        if correct_records is None:
            correct_records = torch.empty((0, CORRECT_RECORD_COLS), dtype=torch.long)
        if incorrect_records is None:
            incorrect_records = torch.empty((0, INCORRECT_RECORD_COLS), dtype=torch.long)
        return self.update_from_step_payload(
            step,
            correct_scores=correct_scores,
            correct_records=correct_records,
            incorrect_scores=incorrect_scores,
            incorrect_records=incorrect_records,
        )

    def _plan_from_accumulator(self, step: int) -> dict[int, dict[str, Any]]:
        start_step = int(step) + self.lookahead_steps
        if start_step >= self.num_steps:
            return {}
        baseline_steps = self._load_steps_window(start_step, self.window_steps)
        if not baseline_steps:
            return {}
        future_sequence_slots = [
            int(microstep["sequence_id"])
            for step_entry in baseline_steps
            for microstep in step_entry.get("microsteps", [])
        ]
        if len(future_sequence_slots) == 0:
            return {}

        positive_records = self.accumulator.sample_correct_tokens(len(future_sequence_slots), seed=self.sampling_seed + int(step) * 17 + 1)
        corrective_records = self.accumulator.sample_incorrect_pairs(len(future_sequence_slots), seed=self.sampling_seed + int(step) * 17 + 2)
        planned_overrides: dict[int, dict[str, Any]] = {}
        remaining_slots = list(future_sequence_slots)

        for window_idx, baseline_step_entry in enumerate(baseline_steps):
            override_step = start_step + window_idx
            slot_count = len(baseline_step_entry.get("microsteps", []))
            if slot_count <= 0:
                continue
            positive_quota = min(int(math.floor(slot_count * self.positive_fraction)), slot_count)
            corrective_quota = min(int(math.floor(slot_count * self.corrective_fraction)), slot_count - positive_quota)
            baseline_quota = max(0, slot_count - positive_quota - corrective_quota)

            selected_microsteps: list[dict[str, Any]] = []
            for token_id in positive_records:
                if positive_quota <= 0:
                    break
                sequence_id = self._pop_sequence_for_token(remaining_slots, token_id)
                if sequence_id is None:
                    continue
                selected_microsteps.append(self._build_microstep(sequence_id, bucket="positive", focus_token_id=token_id))
                positive_quota -= 1

            for wrong_token_id, correct_token_id in corrective_records:
                if corrective_quota <= 0:
                    break
                sequence_id = self._pop_sequence_for_token(remaining_slots, correct_token_id)
                if sequence_id is None:
                    continue
                selected_microsteps.append(
                    self._build_microstep(
                        sequence_id,
                        bucket="corrective",
                        focus_token_id=correct_token_id,
                        explicit_negative_entries=[(wrong_token_id, self.accumulator.incorrect_pair_totals.get((wrong_token_id, correct_token_id), 0.0))],
                    )
                )
                corrective_quota -= 1

            while baseline_quota > 0 and remaining_slots:
                sequence_id = remaining_slots.pop(0)
                selected_microsteps.append(self._build_microstep(sequence_id, bucket="baseline"))
                baseline_quota -= 1

            while len(selected_microsteps) < slot_count and remaining_slots:
                sequence_id = remaining_slots.pop(0)
                selected_microsteps.append(self._build_microstep(sequence_id, bucket="baseline"))

            if len(selected_microsteps) != slot_count:
                selected_microsteps = [
                    self._build_microstep(int(microstep["sequence_id"]), bucket="baseline")
                    for microstep in baseline_step_entry.get("microsteps", [])
                ]

            grad_accum_active_ids = _dedupe_preserve_order(
                [token_id for microstep in selected_microsteps for token_id in microstep["active_ids"]]
            )
            if len(grad_accum_active_ids) > self.grad_accum_u_max:
                selected_microsteps = [
                    self._build_baseline_microstep(microstep)
                    for microstep in baseline_step_entry.get("microsteps", [])
                ]
                grad_accum_active_ids = _dedupe_preserve_order(
                    [token_id for microstep in selected_microsteps for token_id in microstep["active_ids"]]
                )
            planned_overrides[override_step] = {
                "grad_accum_active_ids": grad_accum_active_ids,
                "grad_accum_u_size": len(grad_accum_active_ids),
                "microsteps": selected_microsteps,
                "replanned_from_step": int(step),
            }

        override_end = start_step + len(planned_overrides)
        for override_step in list(self._step_overrides.keys()):
            if start_step <= override_step < override_end:
                del self._step_overrides[override_step]
        self._step_overrides.update(planned_overrides)
        return planned_overrides

    def _load_steps_window(self, start_step: int, count: int) -> list[dict[str, Any]]:
        steps: list[dict[str, Any]] = []
        for step_entry in stream_sparse_manifest_steps(self.manifest_path, start_step=start_step):
            steps.append(step_entry)
            if len(steps) >= count:
                break
        return steps

    def _pop_sequence_for_token(self, remaining_slots: list[int], token_id: int) -> int | None:
        for index, sequence_id in enumerate(remaining_slots):
            sequence_unit = self.sequence_units.get(int(sequence_id))
            if sequence_unit is None:
                continue
            if int(token_id) in sequence_unit.unique_token_ids:
                return remaining_slots.pop(index)
        return None

    def _build_microstep(
        self,
        sequence_id: int,
        *,
        bucket: str,
        focus_token_id: int | None = None,
        explicit_negative_entries: list[tuple[int, float]] | None = None,
    ) -> dict[str, Any]:
        sequence_unit = self.sequence_units[int(sequence_id)]
        base_active_ids = list(sequence_unit.unique_token_ids)
        if len(base_active_ids) > self.u_max:
            raise ValueError(
                f"Sequence unit {sequence_id} exceeds manifest u_max: {len(base_active_ids)} > {self.u_max}"
            )
        explicit_negative_entries = explicit_negative_entries or []
        explicit_negative_entries = sorted(explicit_negative_entries, key=lambda item: (-float(item[1]), int(item[0])))
        auto_negative_entries = self.accumulator.lookup_negatives_for_tokens(
            sequence_unit.unique_token_ids,
            limit=self.max_auto_negatives_per_microstep,
            exclude={int(token_id) for token_id, _ in explicit_negative_entries},
        )
        remaining_slots = max(0, self.u_max - len(base_active_ids))
        kept_explicit_entries = explicit_negative_entries[:remaining_slots]
        remaining_slots -= len(kept_explicit_entries)
        kept_auto_entries = auto_negative_entries[:remaining_slots]
        explicit_negative_ids = [int(token_id) for token_id, _ in kept_explicit_entries]
        auto_negative_ids = [int(token_id) for token_id, _ in kept_auto_entries]
        active_ids = list(base_active_ids)
        active_ids.extend(explicit_negative_ids)
        active_ids.extend(auto_negative_ids)
        active_ids = _dedupe_preserve_order(active_ids)
        payload: dict[str, Any] = {
            "sequence_id": int(sequence_id),
            "active_ids": active_ids,
            "u_size": len(active_ids),
            "next_active_ids": active_ids,
            "next_u_size": len(active_ids),
            "next_leaving_ids": [],
            "next_new_ids": [],
            "planner_bucket": bucket,
        }
        if focus_token_id is not None:
            payload["focus_token_id"] = int(focus_token_id)
        injected_negative_ids = explicit_negative_ids + [token_id for token_id in auto_negative_ids if token_id not in explicit_negative_ids]
        if injected_negative_ids:
            payload["injected_negative_ids"] = injected_negative_ids
        if explicit_negative_ids:
            payload["explicit_injected_negative_ids"] = explicit_negative_ids
        if auto_negative_ids:
            payload["auto_injected_negative_ids"] = auto_negative_ids
        return payload

    def _build_baseline_microstep(self, microstep: dict[str, Any]) -> dict[str, Any]:
        active_ids = [int(token_id) for token_id in microstep.get("active_ids", [])]
        return {
            "sequence_id": int(microstep.get("sequence_id", -1)),
            "active_ids": active_ids,
            "u_size": len(active_ids),
            "next_active_ids": active_ids,
            "next_u_size": len(active_ids),
            "next_leaving_ids": [],
            "next_new_ids": [],
            "planner_bucket": "baseline",
        }

    def _valid_correct_records(self, records: torch.Tensor | None) -> list[int]:
        if records is None:
            return []
        records = records.detach().to(device="cpu", dtype=torch.long)
        if records.ndim != 2 or records.size(1) != CORRECT_RECORD_COLS:
            raise ValueError("Correct sparse top-k records must have shape (N, 7)")
        valid_mask = records[:, 6] >= 0
        return [int(token_id) for token_id in records[valid_mask, 6].tolist()]

    def _valid_incorrect_records(self, records: torch.Tensor | None) -> list[tuple[int, int]]:
        if records is None:
            return []
        records = records.detach().to(device="cpu", dtype=torch.long)
        if records.ndim != 2 or records.size(1) != INCORRECT_RECORD_COLS:
            raise ValueError("Incorrect sparse top-k records must have shape (N, 9)")
        valid_mask = torch.logical_and(records[:, 6] >= 0, records[:, 8] >= 0)
        valid_records = records[valid_mask]
        return [(int(row[6].item()), int(row[8].item())) for row in valid_records]