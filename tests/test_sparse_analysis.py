from pathlib import Path

import torch

import nanochat.sparse_analysis as sparse_analysis_module
from nanochat.sparse_analysis import (
    CORRECT_CANDIDATE_RECORD_COLS,
    CORRECT_RECORD_COLS,
    INCORRECT_CANDIDATE_RECORD_COLS,
    INCORRECT_RECORD_COLS,
    SPARSE_LOSS_TOPK_APPROX_CANDIDATE_POOL,
    SPARSE_LOSS_TOPK_RANKING_ACCUMULATED,
    SPARSE_LOSS_TOPK_RANKING_SINGLE,
    SparseLossAnalysisWriter,
    collect_sparse_loss_candidate_pool_from_stats,
    collect_sparse_loss_topk,
    collect_sparse_loss_topk_from_candidate_pool,
    collect_sparse_loss_topk_from_stats,
    resolve_sparse_analysis_dir,
    merge_topk_records,
    select_topk_records,
)
from nanochat.sparse_window_accum import SparseRollingLossAccumulator
from nanochat.sparse_window_accum import SparseDecayedHardNegativePool


def test_collect_sparse_loss_topk_shapes_and_global_ids():
    logits = torch.tensor(
        [
            [
                [0.1, 3.0, 0.2],
                [2.5, 0.1, 1.9],
            ]
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([[0, 2]], dtype=torch.long)
    active_global_ids_cpu = torch.tensor([10, 20, 30], dtype=torch.long)

    payload = collect_sparse_loss_topk(
        logits,
        targets,
        active_global_ids_cpu,
        topk_correct=2,
        topk_incorrect=2,
        step=7,
        micro_step=1,
        sequence_id=99,
    )

    assert payload["correct_scores"].shape == (2,)
    assert payload["correct_records"].shape == (2, CORRECT_RECORD_COLS)
    assert payload["incorrect_scores"].shape == (2,)
    assert payload["incorrect_records"].shape == (2, INCORRECT_RECORD_COLS)

    # The first position is under-predicted: target local id 0 -> global id 10
    assert payload["correct_records"][0, 0].item() == 7
    assert payload["correct_records"][0, 1].item() == 1
    assert payload["correct_records"][0, 2].item() == 99
    assert payload["correct_records"][0, 5].item() == 0
    assert payload["correct_records"][0, 6].item() == 10

    # The highest over-predicted incorrect token for the first position is local id 1 -> global id 20
    assert payload["incorrect_records"][0, 5].item() == 0
    assert payload["incorrect_records"][0, 6].item() == 10
    assert payload["incorrect_records"][0, 7].item() == 1
    assert payload["incorrect_records"][0, 8].item() == 20


def test_step_level_sparse_loss_aggregation_sums_duplicate_tokens_and_pairs():
    logits0 = torch.tensor(
        [[[0.2, 4.0, 0.1], [0.1, 3.0, 0.2]]],
        dtype=torch.float32,
    )
    targets0 = torch.tensor([[0, 0]], dtype=torch.long)
    logits1 = torch.tensor(
        [[[0.3, 2.5, 0.1], [0.2, 0.1, 2.0]]],
        dtype=torch.float32,
    )
    targets1 = torch.tensor([[0, 2]], dtype=torch.long)
    active_global_ids_cpu = torch.tensor([10, 20, 30], dtype=torch.long)

    payload0 = collect_sparse_loss_topk(
        logits0,
        targets0,
        active_global_ids_cpu,
        topk_correct=None,
        topk_incorrect=None,
        step=0,
        micro_step=0,
        sequence_id=1,
    )
    payload1 = collect_sparse_loss_topk(
        logits1,
        targets1,
        active_global_ids_cpu,
        topk_correct=None,
        topk_incorrect=None,
        step=0,
        micro_step=1,
        sequence_id=2,
    )

    merged_correct_scores, merged_correct_records = merge_topk_records(
        payload0["correct_scores"],
        payload0["correct_records"],
        payload1["correct_scores"],
        payload1["correct_records"],
        topk=None,
        ranking_mode=SPARSE_LOSS_TOPK_RANKING_ACCUMULATED,
    )
    merged_incorrect_scores, merged_incorrect_records = merge_topk_records(
        payload0["incorrect_scores"],
        payload0["incorrect_records"],
        payload1["incorrect_scores"],
        payload1["incorrect_records"],
        topk=None,
        ranking_mode=SPARSE_LOSS_TOPK_RANKING_ACCUMULATED,
    )

    bounded_correct_scores, bounded_correct_records = select_topk_records(
        merged_correct_scores,
        merged_correct_records,
        topk=1,
        ranking_mode=SPARSE_LOSS_TOPK_RANKING_ACCUMULATED,
    )
    bounded_incorrect_scores, bounded_incorrect_records = select_topk_records(
        merged_incorrect_scores,
        merged_incorrect_records,
        topk=1,
        ranking_mode=SPARSE_LOSS_TOPK_RANKING_ACCUMULATED,
    )

    assert bounded_correct_records[0, 6].item() == 10
    assert bounded_correct_scores[0].item() > payload0["correct_scores"][0].item()
    assert bounded_incorrect_records[0, 6].item() == 10
    assert bounded_incorrect_records[0, 8].item() == 20
    assert bounded_incorrect_scores[0].item() > payload0["incorrect_scores"][0].item()


def test_step_level_sparse_loss_aggregation_defaults_to_single_occurrence_max_score():
    logits0 = torch.tensor(
        [[[0.2, 4.0, 0.1], [0.1, 3.0, 0.2]]],
        dtype=torch.float32,
    )
    targets0 = torch.tensor([[0, 0]], dtype=torch.long)
    logits1 = torch.tensor(
        [[[0.3, 2.5, 0.1], [0.2, 0.1, 2.0]]],
        dtype=torch.float32,
    )
    targets1 = torch.tensor([[0, 2]], dtype=torch.long)
    active_global_ids_cpu = torch.tensor([10, 20, 30], dtype=torch.long)

    payload0 = collect_sparse_loss_topk(
        logits0,
        targets0,
        active_global_ids_cpu,
        topk_correct=None,
        topk_incorrect=None,
        step=0,
        micro_step=0,
        sequence_id=1,
    )
    payload1 = collect_sparse_loss_topk(
        logits1,
        targets1,
        active_global_ids_cpu,
        topk_correct=None,
        topk_incorrect=None,
        step=0,
        micro_step=1,
        sequence_id=2,
    )

    merged_correct_scores, merged_correct_records = merge_topk_records(
        payload0["correct_scores"],
        payload0["correct_records"],
        payload1["correct_scores"],
        payload1["correct_records"],
        topk=None,
    )
    merged_incorrect_scores, merged_incorrect_records = merge_topk_records(
        payload0["incorrect_scores"],
        payload0["incorrect_records"],
        payload1["incorrect_scores"],
        payload1["incorrect_records"],
        topk=None,
    )

    bounded_correct_scores, bounded_correct_records = select_topk_records(
        merged_correct_scores,
        merged_correct_records,
        topk=1,
    )
    bounded_incorrect_scores, bounded_incorrect_records = select_topk_records(
        merged_incorrect_scores,
        merged_incorrect_records,
        topk=1,
    )

    assert bounded_correct_records[0, 6].item() == 10
    assert bounded_correct_records[0, 1].item() == 0
    assert torch.allclose(bounded_correct_scores[:1], payload0["correct_scores"][:1])
    assert bounded_incorrect_records[0, 6].item() == 10
    assert bounded_incorrect_records[0, 8].item() == 20
    assert bounded_incorrect_records[0, 1].item() == 0
    assert torch.allclose(bounded_incorrect_scores[:1], payload0["incorrect_scores"][:1])


def test_collect_sparse_loss_topk_single_occurrence_keeps_highest_error_record_per_key():
    targets = torch.tensor([[0, 0, 2]], dtype=torch.long)
    active_global_ids_cpu = torch.tensor([10, 20, 30], dtype=torch.long)
    losses = torch.tensor([[1.0, 4.0, 0.5]], dtype=torch.float32)
    top2_local = torch.tensor([[[1, 0], [1, 0], [0, 2]]], dtype=torch.long)
    top2_logits = torch.tensor([[[3.0, 0.5], [5.0, 1.0], [2.5, 2.0]]], dtype=torch.float32)
    target_logits = torch.tensor([[0.5, 1.0, 2.0]], dtype=torch.float32)

    payload = collect_sparse_loss_topk_from_stats(
        targets,
        active_global_ids_cpu,
        topk_correct=1,
        topk_incorrect=1,
        step=9,
        micro_step=3,
        sequence_id=17,
        losses=losses,
        top2_logits=top2_logits,
        top2_local=top2_local,
        target_logits=target_logits,
        ranking_mode=SPARSE_LOSS_TOPK_RANKING_SINGLE,
    )

    assert payload["correct_scores"].tolist() == [4.0]
    assert payload["correct_records"][0, 4].item() == 1
    assert payload["incorrect_scores"].tolist() == [4.0]
    assert payload["incorrect_records"][0, 4].item() == 1


def test_collect_sparse_loss_topk_accumulated_mode_sums_duplicate_scores():
    targets = torch.tensor([[0, 0]], dtype=torch.long)
    active_global_ids_cpu = torch.tensor([10, 20], dtype=torch.long)
    losses = torch.tensor([[1.5, 2.5]], dtype=torch.float32)
    top2_local = torch.tensor([[[1, 0], [1, 0]]], dtype=torch.long)
    top2_logits = torch.tensor([[[3.0, 0.5], [4.0, 1.5]]], dtype=torch.float32)
    target_logits = torch.tensor([[0.5, 1.5]], dtype=torch.float32)

    payload = collect_sparse_loss_topk_from_stats(
        targets,
        active_global_ids_cpu,
        topk_correct=None,
        topk_incorrect=None,
        step=4,
        micro_step=2,
        sequence_id=13,
        losses=losses,
        top2_logits=top2_logits,
        top2_local=top2_local,
        target_logits=target_logits,
        ranking_mode=SPARSE_LOSS_TOPK_RANKING_ACCUMULATED,
    )

    assert payload["correct_scores"].tolist() == [4.0]
    assert payload["incorrect_scores"].tolist() == [5.0]


def test_collect_sparse_loss_candidate_pool_from_stats_prunes_raw_occurrences():
    targets = torch.tensor([[0, 0, 0]], dtype=torch.long)
    losses = torch.tensor([[1.0, 3.0, 2.0]], dtype=torch.float32)
    top2_local = torch.tensor([[[1, 0], [1, 0], [1, 0]]], dtype=torch.long)
    top2_logits = torch.tensor([[[2.5, 0.5], [4.5, 0.5], [3.5, 0.5]]], dtype=torch.float32)
    target_logits = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32)

    payload = collect_sparse_loss_candidate_pool_from_stats(
        targets,
        candidate_pool_correct=2,
        candidate_pool_incorrect=1,
        step=8,
        micro_step=2,
        sequence_id=31,
        losses=losses,
        top2_logits=top2_logits,
        top2_local=top2_local,
        target_logits=target_logits,
    )

    assert payload["correct_candidate_scores"].shape == (2,)
    assert payload["correct_candidate_records"].shape == (2, CORRECT_CANDIDATE_RECORD_COLS)
    assert payload["correct_candidate_scores"].tolist() == [3.0, 2.0]
    assert payload["correct_candidate_records"][0, 4].item() == 1
    assert payload["correct_candidate_records"][1, 4].item() == 2
    assert payload["incorrect_candidate_scores"].shape == (1,)
    assert payload["incorrect_candidate_records"].shape == (1, INCORRECT_CANDIDATE_RECORD_COLS)
    assert payload["incorrect_candidate_scores"].tolist() == [4.0]
    assert payload["incorrect_candidate_records"][0, 4].item() == 1


def test_candidate_pool_finalization_matches_full_stats_path_when_pool_keeps_all_candidates():
    targets = torch.tensor([[0, 0, 2, 1]], dtype=torch.long)
    active_global_ids_cpu = torch.tensor([10, 20, 30], dtype=torch.long)
    losses = torch.tensor([[1.0, 4.0, 0.5, 2.0]], dtype=torch.float32)
    top2_local = torch.tensor([[[1, 0], [1, 0], [0, 2], [2, 1]]], dtype=torch.long)
    top2_logits = torch.tensor([[[3.0, 0.5], [5.0, 1.0], [2.5, 2.0], [4.0, 1.5]]], dtype=torch.float32)
    target_logits = torch.tensor([[0.5, 1.0, 2.0, 1.5]], dtype=torch.float32)

    candidate_payload = collect_sparse_loss_candidate_pool_from_stats(
        targets,
        candidate_pool_correct=SPARSE_LOSS_TOPK_APPROX_CANDIDATE_POOL,
        candidate_pool_incorrect=SPARSE_LOSS_TOPK_APPROX_CANDIDATE_POOL,
        step=9,
        micro_step=3,
        sequence_id=17,
        losses=losses,
        top2_logits=top2_logits,
        top2_local=top2_local,
        target_logits=target_logits,
    )

    pool_payload = collect_sparse_loss_topk_from_candidate_pool(
        active_global_ids_cpu,
        correct_candidate_scores=candidate_payload["correct_candidate_scores"],
        correct_candidate_records=candidate_payload["correct_candidate_records"],
        incorrect_candidate_scores=candidate_payload["incorrect_candidate_scores"],
        incorrect_candidate_records=candidate_payload["incorrect_candidate_records"],
        topk_correct=None,
        topk_incorrect=None,
        ranking_mode=SPARSE_LOSS_TOPK_RANKING_SINGLE,
    )
    full_payload = collect_sparse_loss_topk_from_stats(
        targets,
        active_global_ids_cpu,
        topk_correct=None,
        topk_incorrect=None,
        step=9,
        micro_step=3,
        sequence_id=17,
        losses=losses,
        top2_logits=top2_logits,
        top2_local=top2_local,
        target_logits=target_logits,
        ranking_mode=SPARSE_LOSS_TOPK_RANKING_SINGLE,
    )

    assert torch.equal(pool_payload["correct_records"], full_payload["correct_records"])
    assert torch.equal(pool_payload["incorrect_records"], full_payload["incorrect_records"])
    assert torch.allclose(pool_payload["correct_scores"], full_payload["correct_scores"])
    assert torch.allclose(pool_payload["incorrect_scores"], full_payload["incorrect_scores"])


def test_collect_sparse_loss_topk_from_stats_matches_full_logits_path():
    logits = torch.tensor(
        [
            [
                [0.1, 3.0, 0.2],
                [2.5, 0.1, 1.9],
                [0.4, 0.3, 2.2],
            ]
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([[0, 2, 1]], dtype=torch.long)
    active_global_ids_cpu = torch.tensor([10, 20, 30], dtype=torch.long)
    losses = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=-1,
        reduction="none",
    ).view_as(targets)
    target_logits = logits.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    top2_logits, top2_local = torch.topk(logits, k=2, dim=-1)

    full_payload = collect_sparse_loss_topk(
        logits,
        targets,
        active_global_ids_cpu,
        topk_correct=None,
        topk_incorrect=None,
        step=5,
        micro_step=0,
        sequence_id=11,
        losses=losses,
    )
    stats_payload = collect_sparse_loss_topk_from_stats(
        targets,
        active_global_ids_cpu,
        topk_correct=None,
        topk_incorrect=None,
        step=5,
        micro_step=0,
        sequence_id=11,
        losses=losses,
        top2_logits=top2_logits,
        top2_local=top2_local,
        target_logits=target_logits,
    )

    assert torch.equal(stats_payload["correct_records"], full_payload["correct_records"])
    assert torch.equal(stats_payload["incorrect_records"], full_payload["incorrect_records"])
    assert torch.allclose(stats_payload["correct_scores"], full_payload["correct_scores"])
    assert torch.allclose(stats_payload["incorrect_scores"], full_payload["incorrect_scores"])


def test_sparse_loss_analysis_writer_persists_cpu_tensors(tmp_path, monkeypatch):
    output_dir = Path(tmp_path) / "analysis"
    monkeypatch.setattr(sparse_analysis_module, "resolve_token_cache_dir", lambda path: Path(path) if path is not None else output_dir)
    writer = SparseLossAnalysisWriter(output_dir, token_cache_dir=None)
    payload = {
        "step": 3,
        "correct_scores": torch.tensor([1.5, 0.5], device="cpu"),
        "correct_records": torch.tensor([[3, 0, 1, 0, 0, 4, 44], [3, 0, -1, -1, -1, -1, -1]], dtype=torch.long),
        "incorrect_scores": torch.tensor([0.7, -float("inf")], device="cpu"),
        "incorrect_records": torch.tensor([[3, 0, 1, 0, 0, 4, 44, 7, 77], [3, 0, -1, -1, -1, -1, -1, -1, -1]], dtype=torch.long),
    }
    writer.submit(3, payload)
    writer.close()

    saved = torch.load(output_dir / "steps_000003_000003.pt", map_location="cpu")
    assert saved[3]["step"] == 3
    assert torch.equal(saved[3]["correct_records"], payload["correct_records"])
    assert torch.equal(saved[3]["incorrect_records"], payload["incorrect_records"])


def test_resolve_sparse_analysis_dir_defaults_beside_token_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(sparse_analysis_module, "resolve_token_cache_dir", lambda path: Path(tmp_path) / "token_cache")
    resolved = resolve_sparse_analysis_dir(None, tmp_path)
    assert resolved.name.endswith("_sparse_analysis")


def test_sparse_rolling_loss_accumulator_evicts_exactly_after_window():
    accumulator = SparseRollingLossAccumulator(window_steps=2)
    correct_records = torch.tensor([[0, 0, 0, 0, 0, 0, 10]], dtype=torch.long)
    incorrect_records = torch.tensor([[0, 0, 0, 0, 0, 0, 10, 0, 20]], dtype=torch.long)

    accumulator.update_step(
        0,
        correct_scores=torch.tensor([1.0], dtype=torch.float32),
        correct_records=correct_records,
        incorrect_scores=torch.tensor([0.5], dtype=torch.float32),
        incorrect_records=incorrect_records,
    )
    accumulator.update_step(
        1,
        correct_scores=torch.tensor([2.0], dtype=torch.float32),
        correct_records=correct_records,
        incorrect_scores=torch.tensor([1.5], dtype=torch.float32),
        incorrect_records=incorrect_records,
    )
    accumulator.update_step(
        2,
        correct_scores=torch.tensor([4.0], dtype=torch.float32),
        correct_records=correct_records,
        incorrect_scores=torch.tensor([3.0], dtype=torch.float32),
        incorrect_records=incorrect_records,
    )

    assert accumulator.correct_totals[10] == 4.0
    assert accumulator.incorrect_pair_totals[(20, 10)] == 3.0


def test_sparse_rolling_loss_accumulator_accumulated_mode_sums_across_window():
    accumulator = SparseRollingLossAccumulator(window_steps=2, ranking_mode=SPARSE_LOSS_TOPK_RANKING_ACCUMULATED)
    correct_records = torch.tensor([[0, 0, 0, 0, 0, 0, 10]], dtype=torch.long)
    incorrect_records = torch.tensor([[0, 0, 0, 0, 0, 0, 10, 0, 20]], dtype=torch.long)

    accumulator.update_step(
        0,
        correct_scores=torch.tensor([1.0], dtype=torch.float32),
        correct_records=correct_records,
        incorrect_scores=torch.tensor([0.5], dtype=torch.float32),
        incorrect_records=incorrect_records,
    )
    accumulator.update_step(
        1,
        correct_scores=torch.tensor([2.0], dtype=torch.float32),
        correct_records=correct_records,
        incorrect_scores=torch.tensor([1.5], dtype=torch.float32),
        incorrect_records=incorrect_records,
    )
    accumulator.update_step(
        2,
        correct_scores=torch.tensor([4.0], dtype=torch.float32),
        correct_records=correct_records,
        incorrect_scores=torch.tensor([3.0], dtype=torch.float32),
        incorrect_records=incorrect_records,
    )

    assert accumulator.correct_totals[10] == 6.0
    assert accumulator.incorrect_pair_totals[(20, 10)] == 4.5


def test_sparse_decayed_hard_negative_pool_decays_and_prunes_pairs():
    pool = SparseDecayedHardNegativePool(pool_size=2, decay=0.5)
    incorrect_records = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0, 10, 0, 20],
            [0, 0, 0, 0, 0, 0, 11, 0, 30],
        ],
        dtype=torch.long,
    )
    pool.update_step(
        0,
        incorrect_scores=torch.tensor([4.0, 3.0], dtype=torch.float32),
        incorrect_records=incorrect_records,
    )
    pool.update_step(
        2,
        incorrect_scores=torch.tensor([5.0], dtype=torch.float32),
        incorrect_records=torch.tensor([[2, 0, 0, 0, 0, 0, 12, 0, 40]], dtype=torch.long),
    )

    ranked_pairs = pool.ranked_pairs(step=2)
    assert [pair for pair, _ in ranked_pairs] == [(40, 12), (20, 10)]
    assert pool.pair_count == 2


def test_sparse_decayed_hard_negative_pool_selects_only_for_present_targets():
    pool = SparseDecayedHardNegativePool(pool_size=4, decay=1.0)
    incorrect_records = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0, 10, 0, 20],
            [0, 0, 0, 0, 0, 0, 10, 0, 21],
            [0, 0, 0, 0, 0, 0, 11, 0, 30],
        ],
        dtype=torch.long,
    )
    pool.update_step(
        0,
        incorrect_scores=torch.tensor([4.0, 1.0, 3.0], dtype=torch.float32),
        incorrect_records=incorrect_records,
    )

    selection = pool.select_for_targets({11}, step=0, limit=2, exclude={30})

    assert selection["negative_ids"] == []
    assert selection["candidate_count"] == 0
    assert selection["matched_target_count"] == 1
    assert selection["matched_pair_count"] == 1

    selection = pool.select_for_targets({10}, step=0, limit=2, exclude={21})

    assert selection["negative_ids"] == [20]
    assert selection["candidate_count"] == 1
    assert selection["matched_target_count"] == 1
    assert selection["matched_pair_count"] == 2
