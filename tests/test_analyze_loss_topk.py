from pathlib import Path

import torch

from scripts import analyze_loss_topk


def test_iter_step_payloads_supports_legacy_and_batched_files(tmp_path):
    analysis_dir = Path(tmp_path)
    torch.save(
        {
            "step": 1,
            "correct_scores": torch.tensor([1.0, -float("inf")], dtype=torch.float32),
            "correct_records": torch.tensor([[1, 0, 0, 0, 0, 0, 11], [-1, -1, -1, -1, -1, -1, -1]], dtype=torch.long),
            "incorrect_scores": torch.tensor([0.5, -float("inf")], dtype=torch.float32),
            "incorrect_records": torch.tensor([[1, 0, 0, 0, 0, 0, 11, 0, 21], [-1, -1, -1, -1, -1, -1, -1, -1, -1]], dtype=torch.long),
        },
        analysis_dir / "step_000001.pt",
    )
    torch.save(
        {
            2: {
                "step": 2,
                "correct_scores": torch.tensor([2.0], dtype=torch.float32),
                "correct_records": torch.tensor([[2, 0, 0, 0, 0, 0, 12]], dtype=torch.long),
                "incorrect_scores": torch.tensor([1.5], dtype=torch.float32),
                "incorrect_records": torch.tensor([[2, 0, 0, 0, 0, 0, 12, 0, 22]], dtype=torch.long),
            },
            3: {
                "step": 3,
                "correct_scores": torch.tensor([3.0], dtype=torch.float32),
                "correct_records": torch.tensor([[3, 0, 0, 0, 0, 0, 13]], dtype=torch.long),
                "incorrect_scores": torch.tensor([2.5], dtype=torch.float32),
                "incorrect_records": torch.tensor([[3, 0, 0, 0, 0, 0, 13, 0, 23]], dtype=torch.long),
            },
        },
        analysis_dir / "steps_000002_000003.pt",
    )

    payloads = analyze_loss_topk.iter_step_payloads(analysis_dir)

    assert [step for step, _payload in payloads] == [1, 2, 3]
    assert payloads[0][1]["correct_scores"].tolist() == [1.0]
    assert payloads[1][1]["incorrect_records"][0, analyze_loss_topk._COL_WRONG_GLOBAL].item() == 22


def test_sort_ranked_items_respects_single_vs_accumulated_mode():
    items = {
        (11,): {"count": 2, "max_score": 3.0, "total_score": 5.0},
        (22,): {"count": 1, "max_score": 4.0, "total_score": 4.0},
    }

    single_sorted = analyze_loss_topk.sort_ranked_items(items, ranking_mode="single")
    accumulated_sorted = analyze_loss_topk.sort_ranked_items(items, ranking_mode="accumulated")

    assert [key for key, _stats in single_sorted] == [(22,), (11,)]
    assert [key for key, _stats in accumulated_sorted] == [(11,), (22,)]


def test_aggregate_ranked_records_tracks_count_total_and_max():
    scores = torch.tensor([1.0, 3.0, 2.5], dtype=torch.float32)
    records = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0, 11],
            [0, 0, 0, 0, 1, 0, 11],
            [0, 0, 0, 0, 2, 0, 12],
        ],
        dtype=torch.long,
    )

    aggregated = analyze_loss_topk.aggregate_ranked_records(
        scores,
        records,
        key_columns=(analyze_loss_topk._COL_TARGET_GLOBAL,),
        ranking_mode="single",
    )

    assert aggregated[(11,)]["count"] == 2
    assert aggregated[(11,)]["total_score"] == 4.0
    assert aggregated[(11,)]["max_score"] == 3.0
    assert aggregated[(12,)]["count"] == 1