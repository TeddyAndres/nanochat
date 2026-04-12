from pathlib import Path

import torch

import nanochat.sparse_analysis as sparse_analysis_module
from nanochat.sparse_analysis import (
    CORRECT_RECORD_COLS,
    INCORRECT_RECORD_COLS,
    SparseLossAnalysisWriter,
    collect_sparse_loss_topk,
    collect_sparse_loss_topk_from_stats,
    resolve_sparse_analysis_dir,
    merge_topk_records,
    select_topk_records,
)
from nanochat.sparse_window_accum import SparseRollingLossAccumulator


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
    assert bounded_correct_scores[0].item() > payload0["correct_scores"][0].item()
    assert bounded_incorrect_records[0, 6].item() == 10
    assert bounded_incorrect_records[0, 8].item() == 20
    assert bounded_incorrect_scores[0].item() > payload0["incorrect_scores"][0].item()


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

    saved = torch.load(output_dir / "step_000003.pt", map_location="cpu")
    assert saved["step"] == 3
    assert torch.equal(saved["correct_records"], payload["correct_records"])
    assert torch.equal(saved["incorrect_records"], payload["incorrect_records"])


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

    assert accumulator.correct_totals[10] == 6.0
    assert accumulator.incorrect_pair_totals[(20, 10)] == 4.5
