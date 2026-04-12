from pathlib import Path

import torch

import nanochat.sparse_analysis as sparse_analysis_module
from nanochat.sparse_analysis import (
    CORRECT_RECORD_COLS,
    INCORRECT_RECORD_COLS,
    SparseLossAnalysisWriter,
    collect_sparse_loss_topk,
    resolve_sparse_analysis_dir,
)


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
