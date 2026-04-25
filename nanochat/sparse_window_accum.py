from __future__ import annotations

from collections import deque
import math
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


def _extract_correct_contributions(
    scores: torch.Tensor,
    records: torch.Tensor,
    *,
    ranking_mode: SparseLossTopkRankingMode,
) -> dict[int, float]:
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
        if ranking_mode == SPARSE_LOSS_TOPK_RANKING_ACCUMULATED:
            contributions[token_id] = contributions.get(token_id, 0.0) + float(score)
        else:
            contributions[token_id] = max(contributions.get(token_id, float("-inf")), float(score))
    return contributions


def _extract_incorrect_contributions(
    scores: torch.Tensor,
    records: torch.Tensor,
    *,
    ranking_mode: SparseLossTopkRankingMode,
) -> dict[tuple[int, int], float]:
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
        if ranking_mode == SPARSE_LOSS_TOPK_RANKING_ACCUMULATED:
            contributions[key] = contributions.get(key, 0.0) + float(score)
        else:
            contributions[key] = max(contributions.get(key, float("-inf")), float(score))
    return contributions


class SparseDecayedHardNegativePool:
    def __init__(
        self,
        *,
        pool_size: int = 1000,
        decay: float = 1.0,
        ranking_mode: SparseLossTopkRankingMode = SPARSE_LOSS_TOPK_RANKING_SINGLE,
    ):
        self.pool_size = max(1, int(pool_size))
        self.decay = float(decay)
        if not 0.0 < self.decay <= 1.0:
            raise ValueError(f"decay must be in (0, 1], got {decay}")
        self.ranking_mode = normalize_sparse_topk_ranking_mode(ranking_mode)
        self._pair_state: dict[tuple[int, int], tuple[float, int]] = {}
        self.last_update_step: int = -1

    def state_dict(self) -> dict:
        return {
            "pool_size": self.pool_size,
            "decay": self.decay,
            "ranking_mode": self.ranking_mode,
            "last_update_step": self.last_update_step,
            "pairs": [
                [int(wrong_id), int(correct_id), float(score), int(last_step)]
                for (wrong_id, correct_id), (score, last_step) in self._pair_state.items()
            ],
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.pool_size = max(1, int(state_dict.get("pool_size", self.pool_size)))
        self.decay = float(state_dict.get("decay", self.decay))
        if not 0.0 < self.decay <= 1.0:
            raise ValueError(f"decay must be in (0, 1], got {self.decay}")
        self.ranking_mode = normalize_sparse_topk_ranking_mode(state_dict.get("ranking_mode", self.ranking_mode))
        self.last_update_step = int(state_dict.get("last_update_step", -1))
        self._pair_state.clear()
        for wrong_id, correct_id, score, last_step in state_dict.get("pairs", []):
            score = float(score)
            if score <= 0.0:
                continue
            self._pair_state[(int(wrong_id), int(correct_id))] = (score, int(last_step))

    @property
    def pair_count(self) -> int:
        return len(self._pair_state)

    def update_step(
        self,
        step: int,
        *,
        incorrect_scores: torch.Tensor,
        incorrect_records: torch.Tensor,
    ) -> dict[tuple[int, int], float]:
        step = int(step)
        contributions = _extract_incorrect_contributions(
            incorrect_scores,
            incorrect_records,
            ranking_mode=self.ranking_mode,
        )
        if not contributions:
            self._prune(step)
            self.last_update_step = max(self.last_update_step, step)
            return {}
        for pair_key, contribution in contributions.items():
            if contribution <= 0.0:
                continue
            decayed_score = self._decayed_pair_score(pair_key, step) if pair_key in self._pair_state else 0.0
            self._pair_state[pair_key] = (decayed_score + float(contribution), step)
        self.last_update_step = max(self.last_update_step, step)
        self._prune(step)
        return contributions

    def ranked_pairs(self, step: int, limit: int | None = None) -> list[tuple[tuple[int, int], float]]:
        ranked = [
            (pair_key, self._decayed_pair_score(pair_key, step))
            for pair_key in self._pair_state
        ]
        ranked = [item for item in ranked if item[1] > 0.0]
        ranked.sort(key=lambda item: (-item[1], item[0][1], item[0][0]))
        return ranked if limit is None else ranked[:limit]

    def select_for_targets(
        self,
        target_token_ids: list[int] | tuple[int, ...] | set[int],
        *,
        step: int,
        limit: int,
        exclude: set[int] | None = None,
    ) -> dict[str, int | list[int]]:
        exclude = exclude or set()
        target_set = {int(token_id) for token_id in target_token_ids}
        if limit <= 0 or len(target_set) == 0 or len(self._pair_state) == 0:
            return {
                "negative_ids": [],
                "candidate_count": 0,
                "matched_target_count": 0,
                "matched_pair_count": 0,
            }
        aggregated: dict[int, float] = {}
        matched_targets: set[int] = set()
        matched_pair_count = 0
        for (wrong_id, correct_id), score in self.ranked_pairs(step):
            if correct_id not in target_set:
                continue
            matched_targets.add(int(correct_id))
            matched_pair_count += 1
            if wrong_id in exclude:
                continue
            aggregated[int(wrong_id)] = aggregated.get(int(wrong_id), 0.0) + float(score)
        ranked = sorted(aggregated.items(), key=lambda item: (-item[1], item[0]))
        negative_ids = [token_id for token_id, _ in ranked[:limit]]
        return {
            "negative_ids": negative_ids,
            "candidate_count": len(ranked),
            "matched_target_count": len(matched_targets),
            "matched_pair_count": matched_pair_count,
        }

    def _decayed_pair_score(self, pair_key: tuple[int, int], step: int) -> float:
        score, last_step = self._pair_state[pair_key]
        delta = max(int(step) - int(last_step), 0)
        if delta <= 0 or self.decay == 1.0:
            return float(score)
        return float(score) * math.pow(self.decay, delta)

    def _prune(self, step: int) -> None:
        if len(self._pair_state) <= self.pool_size:
            self._drop_non_positive(step)
            return
        ranked = self.ranked_pairs(step)
        keep_pairs = {pair_key for pair_key, _ in ranked[:self.pool_size]}
        self._pair_state = {
            pair_key: (self._decayed_pair_score(pair_key, step), int(step))
            for pair_key in keep_pairs
        }

    def _drop_non_positive(self, step: int) -> None:
        if not self._pair_state:
            return
        self._pair_state = {
            pair_key: (self._decayed_pair_score(pair_key, step), int(step))
            for pair_key in self._pair_state
            if self._decayed_pair_score(pair_key, step) > 0.0
        }


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
        return _extract_correct_contributions(scores, records, ranking_mode=self.ranking_mode)

    def _extract_incorrect_contributions(self, scores: torch.Tensor, records: torch.Tensor) -> dict[tuple[int, int], float]:
        return _extract_incorrect_contributions(scores, records, ranking_mode=self.ranking_mode)

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