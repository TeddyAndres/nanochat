from __future__ import annotations

from collections import deque
from typing import Deque

import torch

from nanochat.sparse_analysis import (
    CORRECT_RECORD_COLS,
    INCORRECT_RECORD_COLS,
    SPARSE_LOSS_TOPK_RANKING_ACCUMULATED,
    SPARSE_LOSS_TOPK_RANKING_SINGLE,
    SparseLossTopkRankingMode,
    normalize_sparse_topk_ranking_mode,
)


class SparseRollingLossAccumulator:
    def __init__(self, window_steps: int = 20, *, ranking_mode: SparseLossTopkRankingMode = SPARSE_LOSS_TOPK_RANKING_SINGLE):
        self.window_steps = max(1, int(window_steps))
        self.ranking_mode = normalize_sparse_topk_ranking_mode(ranking_mode)
        self._step_queue: Deque[dict] = deque()
        self.correct_totals: dict[int, float] = {}
        self.incorrect_pair_totals: dict[tuple[int, int], float] = {}
        self.correct_pair_lookup: dict[int, list[tuple[int, float]]] = {}

    def update_step(
        self,
        step: int,
        *,
        correct_scores: torch.Tensor,
        correct_records: torch.Tensor,
        incorrect_scores: torch.Tensor,
        incorrect_records: torch.Tensor,
    ) -> dict:
        step_payload = {
            "step": int(step),
            "correct": self._extract_correct_contributions(correct_scores, correct_records),
            "incorrect": self._extract_incorrect_contributions(incorrect_scores, incorrect_records),
        }
        self._step_queue.append(step_payload)
        while len(self._step_queue) > self.window_steps:
            self._step_queue.popleft()
        self._rebuild_window_state()
        return step_payload

    def ranked_correct_tokens(self, limit: int | None = None) -> list[tuple[int, float]]:
        items = sorted(self.correct_totals.items(), key=lambda item: (-item[1], item[0]))
        return items if limit is None else items[:limit]

    @property
    def current_window_size(self) -> int:
        return len(self._step_queue)

    @property
    def is_warmed_up(self) -> bool:
        return len(self._step_queue) >= self.window_steps

    def ranked_incorrect_pairs(self, limit: int | None = None) -> list[tuple[tuple[int, int], float]]:
        items = sorted(self.incorrect_pair_totals.items(), key=lambda item: (-item[1], item[0][1], item[0][0]))
        return items if limit is None else items[:limit]

    def sample_correct_tokens(self, count: int, *, seed: int) -> list[int]:
        return [token_id for token_id, _ in self._weighted_sample(self.ranked_correct_tokens(), count, seed=seed)]

    def sample_incorrect_pairs(self, count: int, *, seed: int) -> list[tuple[int, int]]:
        return [pair for pair, _ in self._weighted_sample(self.ranked_incorrect_pairs(), count, seed=seed)]

    def lookup_negatives_for_tokens(
        self,
        token_ids: list[int] | tuple[int, ...],
        *,
        limit: int | None = None,
        exclude: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        exclude = exclude or set()
        aggregated: dict[int, float] = {}
        for token_id in token_ids:
            for wrong_id, score in self.correct_pair_lookup.get(int(token_id), []):
                if wrong_id in exclude:
                    continue
                aggregated[wrong_id] = aggregated.get(wrong_id, 0.0) + float(score)
        ranked = sorted(aggregated.items(), key=lambda item: (-item[1], item[0]))
        return ranked if limit is None else ranked[:limit]

    def state_dict(self) -> dict:
        return {
            "window_steps": self.window_steps,
            "ranking_mode": self.ranking_mode,
            "step_queue": [
                {
                    "step": int(payload["step"]),
                    "correct": [[int(token_id), float(score)] for token_id, score in payload["correct"].items()],
                    "incorrect": [[int(wrong_id), int(correct_id), float(score)] for (wrong_id, correct_id), score in payload["incorrect"].items()],
                }
                for payload in self._step_queue
            ],
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.window_steps = max(1, int(state_dict.get("window_steps", self.window_steps)))
        self.ranking_mode = normalize_sparse_topk_ranking_mode(state_dict.get("ranking_mode", self.ranking_mode))
        self._step_queue.clear()
        self.correct_totals.clear()
        self.incorrect_pair_totals.clear()
        for payload in state_dict.get("step_queue", []):
            step_payload = {
                "step": int(payload.get("step", -1)),
                "correct": {int(token_id): float(score) for token_id, score in payload.get("correct", [])},
                "incorrect": {
                    (int(wrong_id), int(correct_id)): float(score)
                    for wrong_id, correct_id, score in payload.get("incorrect", [])
                },
            }
            self._step_queue.append(step_payload)
        self._rebuild_window_state()

    def _extract_correct_contributions(self, scores: torch.Tensor, records: torch.Tensor) -> dict[int, float]:
        scores = scores.detach().to(device="cpu", dtype=torch.float32)
        records = records.detach().to(device="cpu", dtype=torch.long)
        if records.ndim != 2 or records.size(1) != CORRECT_RECORD_COLS:
            raise ValueError("Correct sparse loss records must have shape (N, 7)")
        contributions: dict[int, float] = {}
        for score, record in zip(scores.tolist(), records.tolist()):
            if not torch.isfinite(torch.tensor(score)):
                continue
            token_id = int(record[6])
            if token_id < 0:
                continue
            if self.ranking_mode == SPARSE_LOSS_TOPK_RANKING_ACCUMULATED:
                contributions[token_id] = contributions.get(token_id, 0.0) + float(score)
            else:
                contributions[token_id] = max(contributions.get(token_id, float("-inf")), float(score))
        return contributions

    def _extract_incorrect_contributions(self, scores: torch.Tensor, records: torch.Tensor) -> dict[tuple[int, int], float]:
        scores = scores.detach().to(device="cpu", dtype=torch.float32)
        records = records.detach().to(device="cpu", dtype=torch.long)
        if records.ndim != 2 or records.size(1) != INCORRECT_RECORD_COLS:
            raise ValueError("Incorrect sparse loss records must have shape (N, 9)")
        contributions: dict[tuple[int, int], float] = {}
        for score, record in zip(scores.tolist(), records.tolist()):
            if not torch.isfinite(torch.tensor(score)):
                continue
            correct_id = int(record[6])
            wrong_id = int(record[8])
            if correct_id < 0 or wrong_id < 0:
                continue
            key = (wrong_id, correct_id)
            if self.ranking_mode == SPARSE_LOSS_TOPK_RANKING_ACCUMULATED:
                contributions[key] = contributions.get(key, 0.0) + float(score)
            else:
                contributions[key] = max(contributions.get(key, float("-inf")), float(score))
        return contributions

    def _rebuild_window_state(self) -> None:
        self.correct_totals.clear()
        self.incorrect_pair_totals.clear()
        for step_payload in self._step_queue:
            for token_id, score in step_payload["correct"].items():
                score = float(score)
                if self.ranking_mode == SPARSE_LOSS_TOPK_RANKING_ACCUMULATED:
                    self.correct_totals[token_id] = self.correct_totals.get(token_id, 0.0) + score
                else:
                    self.correct_totals[token_id] = max(self.correct_totals.get(token_id, float("-inf")), score)
            for pair_key, score in step_payload["incorrect"].items():
                score = float(score)
                if self.ranking_mode == SPARSE_LOSS_TOPK_RANKING_ACCUMULATED:
                    self.incorrect_pair_totals[pair_key] = self.incorrect_pair_totals.get(pair_key, 0.0) + score
                else:
                    self.incorrect_pair_totals[pair_key] = max(self.incorrect_pair_totals.get(pair_key, float("-inf")), score)
        self.correct_totals = {token_id: score for token_id, score in self.correct_totals.items() if score > 0.0}
        self.incorrect_pair_totals = {pair_key: score for pair_key, score in self.incorrect_pair_totals.items() if score > 0.0}
        self._rebuild_correct_pair_lookup()

    def _rebuild_correct_pair_lookup(self) -> None:
        lookup: dict[int, list[tuple[int, float]]] = {}
        for (wrong_id, correct_id), score in self.incorrect_pair_totals.items():
            lookup.setdefault(int(correct_id), []).append((int(wrong_id), float(score)))
        for correct_id in lookup:
            lookup[correct_id].sort(key=lambda item: (-item[1], item[0]))
        self.correct_pair_lookup = lookup

    def _weighted_sample(self, ranked_items: list[tuple], count: int, *, seed: int) -> list[tuple]:
        if count <= 0 or len(ranked_items) == 0:
            return []
        if count >= len(ranked_items):
            return ranked_items
        weights = torch.tensor([max(float(score), 0.0) for _, score in ranked_items], dtype=torch.float32)
        if float(weights.sum().item()) <= 0.0:
            return ranked_items[:count]
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        indices = torch.multinomial(weights, num_samples=count, replacement=False, generator=generator)
        sampled = [ranked_items[int(index)] for index in indices.tolist()]
        sampled.sort(key=lambda item: (-item[1], item[0]))
        return sampled