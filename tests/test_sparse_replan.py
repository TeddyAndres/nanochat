import json

import torch

from nanochat import dataloader as dataloader_module
import nanochat.sparse_replan as sparse_replan_module
from nanochat.sparse_replan import SparseFutureWindowPlanner
from nanochat.sparse_manifest import (
    build_grouping_manifest_payload,
    build_manifest_payload,
    build_sequence_manifest_payload,
    build_sequence_manifest_shard_payload,
    save_sequence_manifest_shard,
    save_sparse_manifest,
)


def test_sparse_future_window_planner_builds_mixed_overrides(tmp_path):
    base_shard_path = tmp_path / "sequence_base_shard_000.pt"
    base_manifest_path = tmp_path / "sequence_base.json"
    grouping_shard_path = tmp_path / "grouping_shard_000.json"
    grouping_manifest_path = tmp_path / "grouping_manifest.json"

    sequence_units = []
    for sequence_id, token_base in enumerate(range(10, 90, 10)):
        sequence_units.append(
            {
                "sequence_id": sequence_id,
                "state_dict": {"pq_idx": sequence_id, "rg_idx": 0, "epoch": 0},
                "sequence_recipe": {"row_capacity": 3, "rows": []},
                "num_unique_tokens": 2,
                "unique_token_ids": [token_base + 1, token_base + 2],
                "sequence_geometry": {"rows": 1, "tokens": 2},
            }
        )
    save_sequence_manifest_shard(
        base_shard_path,
        build_sequence_manifest_shard_payload(
            shard_index=0,
            start_sequence_id=0,
            sequence_units=sequence_units,
        ),
    )
    save_sparse_manifest(
        base_manifest_path,
        build_sequence_manifest_payload(
            split="train",
            vocab_size=2048,
            device_batch_size=1,
            max_seq_len=2,
            total_batch_size=8,
            grad_accum_steps=4,
            ddp_world_size=1,
            num_iterations=4,
            buffer_size=8,
            num_sequence_units=len(sequence_units),
            shard_sequence_count=len(sequence_units),
            shards=[
                {
                    "shard_index": 0,
                    "path": str(base_shard_path.relative_to(tmp_path)),
                    "start_sequence_id": 0,
                    "num_sequence_units": len(sequence_units),
                }
            ],
        ),
    )

    steps = []
    baseline_sequences = [
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [4, 5, 6, 7],
    ]
    for step_idx, sequence_ids in enumerate(baseline_sequences):
        microsteps = []
        grad_accum_active_ids = []
        for micro_idx, sequence_id in enumerate(sequence_ids):
            active_ids = sequence_units[sequence_id]["unique_token_ids"]
            grad_accum_active_ids.extend(active_ids)
            microsteps.append(
                {
                    "microstep": micro_idx,
                    "sequence_id": sequence_id,
                    "u_size": len(active_ids),
                    "active_ids": active_ids,
                    "next_active_ids": active_ids,
                    "next_u_size": len(active_ids),
                    "next_leaving_ids": [],
                    "next_new_ids": [],
                }
            )
        steps.append(
            {
                "grad_accum_u_size": len(set(grad_accum_active_ids)),
                "grad_accum_active_ids": list(dict.fromkeys(grad_accum_active_ids)),
                "microsteps": microsteps,
            }
        )
    with grouping_shard_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "version": 4,
                "manifest_kind": "grouping",
                "shard_index": 0,
                "start_step": 0,
                "num_steps": len(steps),
                    "u_max": 3,
                    "grad_accum_u_max": 9,
                "steps": steps,
            },
            f,
        )
    save_sparse_manifest(
        grouping_manifest_path,
        build_grouping_manifest_payload(
            split="train",
            vocab_size=2048,
            device_batch_size=1,
            max_seq_len=2,
            total_batch_size=8,
            grad_accum_steps=4,
            ddp_world_size=1,
            num_iterations=4,
            buffer_size=8,
                u_max=3,
                grad_accum_u_max=9,
            shard_step_count=len(steps),
            shards=[
                {
                    "shard_index": 0,
                    "path": str(grouping_shard_path.relative_to(tmp_path)),
                    "start_step": 0,
                    "num_steps": len(steps),
                        "u_max": 3,
                        "grad_accum_u_max": 9,
                }
            ],
            base_manifest_path=str(base_manifest_path.relative_to(tmp_path)),
        ),
    )

    planner = SparseFutureWindowPlanner(
        grouping_manifest_path,
        interval_steps=2,
        rolling_window_steps=1,
    )
    correct_records = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0, 52],
            [0, 0, 0, 0, 0, 0, -1],
        ],
        dtype=torch.long,
    )
    correct_scores = torch.tensor([5.0, -float("inf")], dtype=torch.float32)
    incorrect_records = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0, 72, 0, 999],
            [0, 0, 0, 0, 0, 0, -1, -1, -1],
        ],
        dtype=torch.long,
    )
    incorrect_scores = torch.tensor([7.0, -float("inf")], dtype=torch.float32)

    planned = planner.update_from_step_payload(
        0,
        correct_scores=correct_scores,
        correct_records=correct_records,
        incorrect_scores=incorrect_scores,
        incorrect_records=incorrect_records,
    )

    assert sorted(planned.keys()) == [2]
    step2 = planned[2]
    assert len(step2["microsteps"]) == 4
    assert step2["microsteps"][0]["planner_bucket"] == "positive"
    assert step2["microsteps"][0]["sequence_id"] == 4
    assert step2["microsteps"][0]["focus_token_id"] == 52
    assert step2["microsteps"][1]["planner_bucket"] == "corrective"
    assert step2["microsteps"][1]["sequence_id"] == 6
    assert step2["microsteps"][1]["injected_negative_ids"] == [999]
    assert 999 in step2["microsteps"][1]["active_ids"]
    assert [microstep["sequence_id"] for microstep in step2["microsteps"][2:]] == [5, 7]


def test_sparse_future_window_planner_uses_interval_as_planning_cadence(tmp_path):
    base_shard_path = tmp_path / "sequence_base_shard_000.sqlite"
    base_manifest_path = tmp_path / "sequence_base.json"
    grouping_shard_path = tmp_path / "grouping_shard_000.json"
    grouping_manifest_path = tmp_path / "grouping_manifest.json"

    sequence_units = []
    for sequence_id, token_base in enumerate(range(10, 60, 10)):
        sequence_units.append(
            {
                "sequence_id": sequence_id,
                "state_dict": {"pq_idx": sequence_id, "rg_idx": 0, "epoch": 0},
                "sequence_recipe": {"row_capacity": 3, "rows": []},
                "num_unique_tokens": 2,
                "unique_token_ids": [token_base + 1, token_base + 2],
                "sequence_geometry": {"rows": 1, "tokens": 2},
            }
        )
    save_sequence_manifest_shard(
        base_shard_path,
        build_sequence_manifest_shard_payload(
            shard_index=0,
            start_sequence_id=0,
            sequence_units=sequence_units,
        ),
    )
    save_sparse_manifest(
        base_manifest_path,
        build_sequence_manifest_payload(
            split="train",
            vocab_size=2048,
            device_batch_size=1,
            max_seq_len=2,
            total_batch_size=3,
            grad_accum_steps=3,
            ddp_world_size=1,
            num_iterations=5,
            buffer_size=3,
            num_sequence_units=len(sequence_units),
            shard_sequence_count=len(sequence_units),
            shards=[
                {
                    "shard_index": 0,
                    "path": str(base_shard_path.relative_to(tmp_path)),
                    "start_sequence_id": 0,
                    "num_sequence_units": len(sequence_units),
                }
            ],
        ),
    )

    steps = []
    baseline_sequences = [
        [0, 1, 2],
        [0, 1, 2],
        [1, 2, 3],
        [1, 2, 3],
        [2, 3, 4],
    ]
    for sequence_ids in baseline_sequences:
        microsteps = []
        grad_accum_active_ids = []
        for micro_idx, sequence_id in enumerate(sequence_ids):
            active_ids = sequence_units[sequence_id]["unique_token_ids"]
            grad_accum_active_ids.extend(active_ids)
            microsteps.append(
                {
                    "microstep": micro_idx,
                    "sequence_id": sequence_id,
                    "u_size": len(active_ids),
                    "active_ids": active_ids,
                    "next_active_ids": active_ids,
                    "next_u_size": len(active_ids),
                    "next_leaving_ids": [],
                    "next_new_ids": [],
                }
            )
        steps.append(
            {
                "grad_accum_u_size": len(set(grad_accum_active_ids)),
                "grad_accum_active_ids": list(dict.fromkeys(grad_accum_active_ids)),
                "microsteps": microsteps,
            }
        )
    with grouping_shard_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "version": 4,
                "manifest_kind": "grouping",
                "shard_index": 0,
                "start_step": 0,
                "num_steps": len(steps),
                "u_max": 3,
                "grad_accum_u_max": 9,
                "steps": steps,
            },
            f,
        )
    save_sparse_manifest(
        grouping_manifest_path,
        build_grouping_manifest_payload(
            split="train",
            vocab_size=2048,
            device_batch_size=1,
            max_seq_len=2,
            total_batch_size=3,
            grad_accum_steps=3,
            ddp_world_size=1,
            num_iterations=5,
            buffer_size=3,
            u_max=3,
            grad_accum_u_max=9,
            shard_step_count=len(steps),
            shards=[
                {
                    "shard_index": 0,
                    "path": str(grouping_shard_path.relative_to(tmp_path)),
                    "start_step": 0,
                    "num_steps": len(steps),
                    "u_max": 3,
                    "grad_accum_u_max": 9,
                }
            ],
            base_manifest_path=str(base_manifest_path.relative_to(tmp_path)),
        ),
    )

    planner = SparseFutureWindowPlanner(grouping_manifest_path, interval_steps=2, rolling_window_steps=1)
    planner.update_from_step_payload(
        0,
        correct_scores=torch.tensor([5.0], dtype=torch.float32),
        correct_records=torch.tensor([[0, 0, 0, 0, 0, 0, 32]], dtype=torch.long),
        incorrect_scores=torch.tensor([1.0], dtype=torch.float32),
        incorrect_records=torch.tensor([[0, 0, 0, 0, 0, 0, 32, 0, 999]], dtype=torch.long),
    )
    first_override = planner.get_step_override(2)
    assert first_override is not None
    assert planner.get_step_override(3) is None

    planner.update_from_step_payload(
        1,
        correct_scores=torch.tensor([4.0], dtype=torch.float32),
        correct_records=torch.tensor([[1, 0, 0, 0, 0, 0, 42]], dtype=torch.long),
        incorrect_scores=torch.tensor([2.0], dtype=torch.float32),
        incorrect_records=torch.tensor([[1, 0, 0, 0, 0, 0, 42, 0, 888]], dtype=torch.long),
    )

    assert planner.get_step_override(2) == first_override
    assert planner.get_step_override(3) is None

    planner.update_from_step_payload(
        2,
        correct_scores=torch.tensor([3.0], dtype=torch.float32),
        correct_records=torch.tensor([[2, 0, 0, 0, 0, 0, 52]], dtype=torch.long),
        incorrect_scores=torch.tensor([2.5], dtype=torch.float32),
        incorrect_records=torch.tensor([[2, 0, 0, 0, 0, 0, 52, 0, 777]], dtype=torch.long),
    )

    assert planner.get_step_override(4) is not None


def test_sparse_future_window_planner_reuses_monotonic_manifest_stream(tmp_path, monkeypatch):
    manifest_path = tmp_path / "grouping_manifest.json"
    save_sparse_manifest(
        manifest_path,
        build_manifest_payload(
            split="train",
            vocab_size=32,
            device_batch_size=1,
            max_seq_len=1,
            total_batch_size=1,
            grad_accum_steps=1,
            ddp_world_size=1,
            num_iterations=6,
            buffer_size=1,
            steps=[
                {
                    "u_size": 1,
                    "active_ids": [step_idx],
                    "next_u_size": 1,
                    "next_active_ids": [step_idx],
                    "next_leaving_ids": [],
                    "next_new_ids": [],
                }
                for step_idx in range(6)
            ],
        ),
    )

    streamed_steps = [{"step": step_idx} for step_idx in range(6)]
    stream_calls = []

    def fake_stream_sparse_manifest_steps(path, start_step=0):
        stream_calls.append(int(start_step))
        for step_entry in streamed_steps[int(start_step):]:
            yield step_entry

    monkeypatch.setattr(sparse_replan_module, "stream_sparse_manifest_steps", fake_stream_sparse_manifest_steps)

    planner = SparseFutureWindowPlanner(manifest_path, interval_steps=2, rolling_window_steps=1, negative_only=True)

    assert planner._load_steps_window(2, 1) == [streamed_steps[2]]
    assert planner._load_steps_window(4, 1) == [streamed_steps[4]]
    assert stream_calls == [0]



def test_sparse_future_window_planner_auto_injects_cold_negative_for_baseline_sequence(tmp_path):
    base_shard_path = tmp_path / "sequence_base_shard_000.sqlite"
    base_manifest_path = tmp_path / "sequence_base.json"
    grouping_shard_path = tmp_path / "grouping_shard_000.json"
    grouping_manifest_path = tmp_path / "grouping_manifest.json"

    sequence_units = [
        {
            "sequence_id": 0,
            "state_dict": {"pq_idx": 0, "rg_idx": 0, "epoch": 0},
            "sequence_recipe": {"row_capacity": 3, "rows": []},
            "num_unique_tokens": 2,
            "unique_token_ids": [11, 12],
            "sequence_geometry": {"rows": 1, "tokens": 2},
        },
        {
            "sequence_id": 1,
            "state_dict": {"pq_idx": 1, "rg_idx": 0, "epoch": 0},
            "sequence_recipe": {"row_capacity": 3, "rows": []},
            "num_unique_tokens": 2,
            "unique_token_ids": [41, 42],
            "sequence_geometry": {"rows": 1, "tokens": 2},
        },
    ]
    save_sequence_manifest_shard(
        base_shard_path,
        build_sequence_manifest_shard_payload(shard_index=0, start_sequence_id=0, sequence_units=sequence_units),
    )
    save_sparse_manifest(
        base_manifest_path,
        build_sequence_manifest_payload(
            split="train",
            vocab_size=2048,
            device_batch_size=1,
            max_seq_len=2,
            total_batch_size=2,
            grad_accum_steps=2,
            ddp_world_size=1,
            num_iterations=2,
            buffer_size=2,
            num_sequence_units=2,
            shard_sequence_count=2,
            shards=[{"shard_index": 0, "path": str(base_shard_path.relative_to(tmp_path)), "start_sequence_id": 0, "num_sequence_units": 2}],
        ),
    )
    steps = [
        {
            "grad_accum_u_size": 4,
            "grad_accum_active_ids": [11, 12, 41, 42],
            "microsteps": [
                {"microstep": 0, "sequence_id": 0, "u_size": 2, "active_ids": [11, 12], "next_active_ids": [11, 12], "next_u_size": 2, "next_leaving_ids": [], "next_new_ids": []},
                {"microstep": 1, "sequence_id": 1, "u_size": 2, "active_ids": [41, 42], "next_active_ids": [41, 42], "next_u_size": 2, "next_leaving_ids": [], "next_new_ids": []},
            ],
        },
        {
            "grad_accum_u_size": 4,
            "grad_accum_active_ids": [11, 12, 41, 42],
            "microsteps": [
                {"microstep": 0, "sequence_id": 0, "u_size": 2, "active_ids": [11, 12], "next_active_ids": [11, 12], "next_u_size": 2, "next_leaving_ids": [], "next_new_ids": []},
                {"microstep": 1, "sequence_id": 1, "u_size": 2, "active_ids": [41, 42], "next_active_ids": [41, 42], "next_u_size": 2, "next_leaving_ids": [], "next_new_ids": []},
            ],
        },
    ]
    with grouping_shard_path.open("w", encoding="utf-8") as f:
        json.dump({"version": 4, "manifest_kind": "grouping", "shard_index": 0, "start_step": 0, "num_steps": 2, "u_max": 3, "grad_accum_u_max": 6, "steps": steps}, f)
    save_sparse_manifest(
        grouping_manifest_path,
        build_grouping_manifest_payload(
            split="train",
            vocab_size=2048,
            device_batch_size=1,
            max_seq_len=2,
            total_batch_size=2,
            grad_accum_steps=2,
            ddp_world_size=1,
            num_iterations=2,
            buffer_size=2,
            u_max=3,
            grad_accum_u_max=6,
            shard_step_count=2,
            shards=[{"shard_index": 0, "path": str(grouping_shard_path.relative_to(tmp_path)), "start_step": 0, "num_steps": 2, "u_max": 3, "grad_accum_u_max": 6}],
            base_manifest_path=str(base_manifest_path.relative_to(tmp_path)),
        ),
    )

    planner = SparseFutureWindowPlanner(
        grouping_manifest_path,
        interval_steps=1,
        rolling_window_steps=1,
        positive_fraction=0.0,
        corrective_fraction=0.0,
    )
    planner.update_from_step_payload(
        0,
        correct_scores=torch.tensor([1.0], dtype=torch.float32),
        correct_records=torch.tensor([[0, 0, 0, 0, 0, 0, 41]], dtype=torch.long),
        incorrect_scores=torch.tensor([3.0], dtype=torch.float32),
        incorrect_records=torch.tensor([[0, 0, 0, 0, 0, 0, 41, 0, 999]], dtype=torch.long),
    )

    override = planner.get_step_override(1)
    assert override is not None
    baseline_microstep = override["microsteps"][1]
    assert baseline_microstep["planner_bucket"] == "baseline"
    assert baseline_microstep["sequence_id"] == 1
    assert baseline_microstep["auto_injected_negative_ids"] == [999]
    assert baseline_microstep["active_ids"] == [41, 42, 999]


def test_manifest_loader_applies_runtime_step_override(monkeypatch, tmp_path):
    steps = [
        {
            "step": 0,
            "u_size": 4,
            "active_ids": [1, 2, 3, 8],
            "next_common_ids": [2, 3],
            "next_leaving_ids": [1],
            "next_new_ids": [4],
        },
        {
            "step": 1,
            "u_size": 4,
            "active_ids": [2, 3, 4, 8],
            "next_common_ids": [],
            "next_leaving_ids": [],
            "next_new_ids": [],
        },
    ]
    manifest_path = tmp_path / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(
            build_manifest_payload(
                split="train",
                vocab_size=32,
                device_batch_size=1,
                max_seq_len=2,
                total_batch_size=4,
                grad_accum_steps=1,
                ddp_world_size=1,
                num_iterations=2,
                buffer_size=4,
                steps=steps,
            ),
            f,
        )

    def fake_base_loader(*args, **kwargs):
        yield torch.tensor([[1, 2]], dtype=torch.long), torch.tensor([[2, 3]], dtype=torch.long), {"pq_idx": 0, "rg_idx": 0, "epoch": 0}
        yield torch.tensor([[2, 3]], dtype=torch.long), torch.tensor([[3, 4]], dtype=torch.long), {"pq_idx": 1, "rg_idx": 0, "epoch": 0}

    monkeypatch.setattr(
        dataloader_module,
        "tokenizing_distributed_data_loader_with_state_bos_bestfit",
        fake_base_loader,
    )

    def override_provider(step_index):
        if step_index != 1:
            return None
        return {
            "u_size": 4,
            "active_ids": [2, 3, 4, 9],
            "next_active_ids": [2, 3, 4, 9],
            "next_u_size": 4,
            "next_leaving_ids": [],
            "next_new_ids": [],
            "injected_negative_ids": [9],
        }

    loader = dataloader_module.tokenizing_distributed_data_loader_with_state_bos_bestfit_manifest(
        tokenizer=None,
        B=1,
        T=2,
        split="train",
        manifest_path=manifest_path,
        device="cpu",
        vocab_size=32,
        include_local_batch=False,
        step_override_provider=override_provider,
    )

    next(loader)
    _, targets1, step_meta1, state1 = next(loader)

    assert torch.equal(targets1, torch.tensor([[1, 2]], dtype=torch.long))
    assert torch.equal(step_meta1["active_ids_cpu"], torch.tensor([2, 3, 4, 9], dtype=torch.long))
    assert torch.equal(step_meta1["stage_ids_cpu"], torch.tensor([2, 3, 4, 9], dtype=torch.long))
    assert state1["manifest_step"] == 1


def test_manifest_loader_uses_sequence_base_manifest_for_runtime_overrides(monkeypatch, tmp_path):
    base_shard_path = tmp_path / "sequence_base_shard_000.pt"
    base_manifest_path = tmp_path / "sequence_base.json"
    grouping_shard_path = tmp_path / "grouping_shard_000.json"
    grouping_manifest_path = tmp_path / "grouping_manifest.json"

    sequence_units = [
        {
            "sequence_id": 0,
            "state_dict": {"pq_idx": 0, "rg_idx": 0, "epoch": 0},
            "inputs": [[11, 12]],
            "targets": [[12, 11]],
            "num_unique_tokens": 2,
            "unique_token_ids": [11, 12],
            "sequence_geometry": {"rows": 1, "tokens": 2},
        },
        {
            "sequence_id": 1,
            "state_dict": {"pq_idx": 1, "rg_idx": 0, "epoch": 0},
            "inputs": [[21, 22]],
            "targets": [[22, 21]],
            "num_unique_tokens": 2,
            "unique_token_ids": [21, 22],
            "sequence_geometry": {"rows": 1, "tokens": 2},
        },
    ]
    save_sequence_manifest_shard(
        base_shard_path,
        build_sequence_manifest_shard_payload(shard_index=0, start_sequence_id=0, sequence_units=sequence_units),
    )
    save_sparse_manifest(
        base_manifest_path,
        build_sequence_manifest_payload(
            split="train",
            vocab_size=128,
            device_batch_size=1,
            max_seq_len=2,
            total_batch_size=2,
            grad_accum_steps=1,
            ddp_world_size=1,
            num_iterations=2,
            buffer_size=2,
            num_sequence_units=2,
            shard_sequence_count=2,
            shards=[
                {
                    "shard_index": 0,
                    "path": str(base_shard_path.relative_to(tmp_path)),
                    "start_sequence_id": 0,
                    "num_sequence_units": 2,
                }
            ],
        ),
    )

    steps = [
        {
            "grad_accum_u_size": 2,
            "grad_accum_active_ids": [11, 12],
            "microsteps": [
                {
                    "microstep": 0,
                    "sequence_id": 0,
                    "u_size": 2,
                    "active_ids": [11, 12],
                    "next_active_ids": [11, 12],
                    "next_u_size": 2,
                    "next_leaving_ids": [],
                    "next_new_ids": [],
                }
            ],
        },
        {
            "grad_accum_u_size": 2,
            "grad_accum_active_ids": [11, 12],
            "microsteps": [
                {
                    "microstep": 0,
                    "sequence_id": 0,
                    "u_size": 2,
                    "active_ids": [11, 12],
                    "next_active_ids": [11, 12],
                    "next_u_size": 2,
                    "next_leaving_ids": [],
                    "next_new_ids": [],
                }
            ],
        },
    ]
    with grouping_shard_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "version": 4,
                "manifest_kind": "grouping",
                "shard_index": 0,
                "start_step": 0,
                "num_steps": 2,
                "u_max": 3,
                "grad_accum_u_max": 3,
                "steps": steps,
            },
            f,
        )
    save_sparse_manifest(
        grouping_manifest_path,
        build_grouping_manifest_payload(
            split="train",
            vocab_size=128,
            device_batch_size=1,
            max_seq_len=2,
            total_batch_size=2,
            grad_accum_steps=1,
            ddp_world_size=1,
            num_iterations=2,
            buffer_size=2,
            u_max=3,
            grad_accum_u_max=3,
            shard_step_count=2,
            shards=[
                {
                    "shard_index": 0,
                    "path": str(grouping_shard_path.relative_to(tmp_path)),
                    "start_step": 0,
                    "num_steps": 2,
                    "u_max": 3,
                    "grad_accum_u_max": 3,
                }
            ],
            base_manifest_path=str(base_manifest_path.relative_to(tmp_path)),
        ),
    )

    def fail_if_live_loader_called(*args, **kwargs):
        raise AssertionError("runtime override path should load from the sequence-base manifest")

    monkeypatch.setattr(
        dataloader_module,
        "tokenizing_distributed_data_loader_with_state_bos_bestfit",
        fail_if_live_loader_called,
    )
    def override_provider(step_index):
        if step_index != 1:
            return None
        return {
            "grad_accum_u_size": 3,
            "grad_accum_active_ids": [21, 22, 99],
            "microsteps": [
                {
                    "microstep": 0,
                    "sequence_id": 1,
                    "u_size": 3,
                    "active_ids": [21, 22, 99],
                    "next_active_ids": [21, 22, 99],
                    "next_u_size": 3,
                    "next_leaving_ids": [],
                    "next_new_ids": [],
                    "injected_negative_ids": [99],
                }
            ],
        }

    loader = dataloader_module.tokenizing_distributed_data_loader_with_state_bos_bestfit_manifest(
        tokenizer=None,
        B=1,
        T=2,
        split="train",
        manifest_path=grouping_manifest_path,
        device="cpu",
        vocab_size=128,
        include_local_batch=False,
        step_override_provider=override_provider,
    )

    next(loader)
    inputs1, targets1, step_meta1, state1 = next(loader)

    assert torch.equal(inputs1, torch.tensor([[0, 1]], dtype=torch.long))
    assert torch.equal(targets1, torch.tensor([[1, 0]], dtype=torch.long))
    assert torch.equal(step_meta1["active_ids_cpu"], torch.tensor([21, 22, 99], dtype=torch.long))
    assert torch.equal(step_meta1["stage_ids_cpu"], torch.tensor([21, 22, 99], dtype=torch.long))
    assert state1["manifest_step"] == 1


def test_manifest_loader_uses_live_loader_for_sequence_stable_runtime_overrides(monkeypatch, tmp_path):
    grouping_shard_path = tmp_path / "grouping_shard_000.json"
    grouping_manifest_path = tmp_path / "grouping_manifest.json"

    steps = [
        {
            "grad_accum_u_size": 2,
            "grad_accum_active_ids": [11, 12],
            "microsteps": [
                {
                    "microstep": 0,
                    "sequence_id": 0,
                    "u_size": 2,
                    "active_ids": [11, 12],
                    "next_active_ids": [11, 12],
                    "next_u_size": 2,
                    "next_leaving_ids": [],
                    "next_new_ids": [],
                }
            ],
        },
        {
            "grad_accum_u_size": 2,
            "grad_accum_active_ids": [11, 12],
            "microsteps": [
                {
                    "microstep": 0,
                    "sequence_id": 0,
                    "u_size": 2,
                    "active_ids": [11, 12],
                    "next_active_ids": [11, 12],
                    "next_u_size": 2,
                    "next_leaving_ids": [],
                    "next_new_ids": [],
                }
            ],
        },
    ]
    with grouping_shard_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "version": 4,
                "manifest_kind": "grouping",
                "shard_index": 0,
                "start_step": 0,
                "num_steps": 2,
                "u_max": 3,
                "grad_accum_u_max": 3,
                "steps": steps,
            },
            f,
        )
    save_sparse_manifest(
        grouping_manifest_path,
        build_grouping_manifest_payload(
            split="train",
            vocab_size=128,
            device_batch_size=1,
            max_seq_len=2,
            total_batch_size=2,
            grad_accum_steps=1,
            ddp_world_size=1,
            num_iterations=2,
            buffer_size=2,
            u_max=3,
            grad_accum_u_max=3,
            shard_step_count=2,
            shards=[
                {
                    "shard_index": 0,
                    "path": str(grouping_shard_path.relative_to(tmp_path)),
                    "start_step": 0,
                    "num_steps": 2,
                    "u_max": 3,
                    "grad_accum_u_max": 3,
                }
            ],
            base_manifest_path="missing_sequence_base.json",
        ),
    )

    def fake_live_loader(*args, **kwargs):
        yield torch.tensor([[11, 12]], dtype=torch.long), torch.tensor([[12, 11]], dtype=torch.long), {"pq_idx": 0, "rg_idx": 0, "epoch": 0}
        yield torch.tensor([[11, 12]], dtype=torch.long), torch.tensor([[12, 11]], dtype=torch.long), {"pq_idx": 1, "rg_idx": 0, "epoch": 0}

    monkeypatch.setattr(
        dataloader_module,
        "tokenizing_distributed_data_loader_with_state_bos_bestfit",
        fake_live_loader,
    )

    def override_provider(step_index):
        if step_index != 1:
            return None
        return {
            "grad_accum_u_size": 3,
            "grad_accum_active_ids": [11, 12, 99],
            "microsteps": [
                {
                    "microstep": 0,
                    "sequence_id": 0,
                    "u_size": 3,
                    "active_ids": [11, 12, 99],
                    "next_active_ids": [11, 12, 99],
                    "next_u_size": 3,
                    "next_leaving_ids": [],
                    "next_new_ids": [],
                    "injected_negative_ids": [99],
                }
            ],
        }

    loader = dataloader_module.tokenizing_distributed_data_loader_with_state_bos_bestfit_manifest(
        tokenizer=None,
        B=1,
        T=2,
        split="train",
        manifest_path=grouping_manifest_path,
        device="cpu",
        vocab_size=128,
        include_local_batch=False,
        step_override_provider=override_provider,
        use_sequence_base_manifest=False,
    )

    next(loader)
    inputs1, targets1, step_meta1, state1 = next(loader)

    assert torch.equal(inputs1, torch.tensor([[0, 1]], dtype=torch.long))
    assert torch.equal(targets1, torch.tensor([[1, 0]], dtype=torch.long))
    assert torch.equal(step_meta1["active_ids_cpu"], torch.tensor([11, 12, 99], dtype=torch.long))
    assert torch.equal(step_meta1["stage_ids_cpu"], torch.tensor([11, 12, 99], dtype=torch.long))
    assert state1["manifest_step"] == 1


def test_sparse_future_window_planner_negative_only_preserves_sequences(tmp_path):
    base_shard_path = tmp_path / "sequence_base_shard_000.pt"
    base_manifest_path = tmp_path / "sequence_base.json"
    grouping_shard_path = tmp_path / "grouping_shard_000.json"
    grouping_manifest_path = tmp_path / "grouping_manifest.json"

    sequence_units = [
        {
            "sequence_id": 0,
            "state_dict": {"pq_idx": 0, "rg_idx": 0, "epoch": 0},
            "inputs": [[41, 42]],
            "targets": [[42, 41]],
            "num_unique_tokens": 2,
            "unique_token_ids": [41, 42],
            "sequence_geometry": {"rows": 1, "tokens": 2},
        },
        {
            "sequence_id": 1,
            "state_dict": {"pq_idx": 1, "rg_idx": 0, "epoch": 0},
            "inputs": [[51, 52]],
            "targets": [[52, 51]],
            "num_unique_tokens": 2,
            "unique_token_ids": [51, 52],
            "sequence_geometry": {"rows": 1, "tokens": 2},
        },
    ]
    save_sequence_manifest_shard(
        base_shard_path,
        build_sequence_manifest_shard_payload(shard_index=0, start_sequence_id=0, sequence_units=sequence_units),
    )
    save_sparse_manifest(
        base_manifest_path,
        build_sequence_manifest_payload(
            split="train",
            vocab_size=2048,
            device_batch_size=1,
            max_seq_len=2,
            total_batch_size=2,
            grad_accum_steps=2,
            ddp_world_size=1,
            num_iterations=2,
            buffer_size=2,
            num_sequence_units=2,
            shard_sequence_count=2,
            shards=[
                {
                    "shard_index": 0,
                    "path": str(base_shard_path.relative_to(tmp_path)),
                    "start_sequence_id": 0,
                    "num_sequence_units": 2,
                }
            ],
        ),
    )

    steps = [
        {
            "grad_accum_u_size": 4,
            "grad_accum_active_ids": [41, 42, 51, 52],
            "microsteps": [
                {"microstep": 0, "sequence_id": 0, "u_size": 2, "active_ids": [41, 42], "next_active_ids": [41, 42], "next_u_size": 2, "next_leaving_ids": [], "next_new_ids": []},
                {"microstep": 1, "sequence_id": 1, "u_size": 2, "active_ids": [51, 52], "next_active_ids": [51, 52], "next_u_size": 2, "next_leaving_ids": [], "next_new_ids": []},
            ],
        },
        {
            "grad_accum_u_size": 4,
            "grad_accum_active_ids": [41, 42, 51, 52],
            "microsteps": [
                {"microstep": 0, "sequence_id": 0, "u_size": 2, "active_ids": [41, 42], "next_active_ids": [41, 42], "next_u_size": 2, "next_leaving_ids": [], "next_new_ids": []},
                {"microstep": 1, "sequence_id": 1, "u_size": 2, "active_ids": [51, 52], "next_active_ids": [51, 52], "next_u_size": 2, "next_leaving_ids": [], "next_new_ids": []},
            ],
        },
    ]
    with grouping_shard_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"version": 4, "manifest_kind": "grouping", "shard_index": 0, "start_step": 0, "num_steps": 2, "u_max": 3, "grad_accum_u_max": 6, "steps": steps},
            f,
        )
    save_sparse_manifest(
        grouping_manifest_path,
        build_grouping_manifest_payload(
            split="train",
            vocab_size=2048,
            device_batch_size=1,
            max_seq_len=2,
            total_batch_size=2,
            grad_accum_steps=2,
            ddp_world_size=1,
            num_iterations=2,
            buffer_size=2,
            u_max=3,
            grad_accum_u_max=6,
            shard_step_count=2,
            shards=[{"shard_index": 0, "path": str(grouping_shard_path.relative_to(tmp_path)), "start_step": 0, "num_steps": 2, "u_max": 3, "grad_accum_u_max": 6}],
            base_manifest_path=str(base_manifest_path.relative_to(tmp_path)),
        ),
    )

    planner = SparseFutureWindowPlanner(
        grouping_manifest_path,
        interval_steps=1,
        rolling_window_steps=1,
        max_auto_negatives_per_microstep=1,
        negative_only=True,
    )

    assert planner.requires_sequence_base_manifest is False
    planner.update_from_step_payload(
        0,
        correct_scores=torch.tensor([1.0], dtype=torch.float32),
        correct_records=torch.tensor([[0, 0, 0, 0, 0, 0, 41]], dtype=torch.long),
        incorrect_scores=torch.tensor([3.0], dtype=torch.float32),
        incorrect_records=torch.tensor([[0, 0, 0, 0, 0, 0, 41, 0, 999]], dtype=torch.long),
    )

    override = planner.get_step_override(1)
    assert override is not None
    assert [microstep["sequence_id"] for microstep in override["microsteps"]] == [0, 1]
    assert override["microsteps"][0]["active_ids"] == [41, 42, 999]
    assert override["microsteps"][0]["auto_injected_negative_ids"] == [999]
    assert override["microsteps"][1]["active_ids"] == [51, 52]


def test_dual_manifest_token_batch_cache_is_bounded(monkeypatch, tmp_path):
    base_shard_path = tmp_path / "sequence_base_shard_000.pt"
    base_manifest_path = tmp_path / "sequence_base.json"
    grouping_shard_path = tmp_path / "grouping_shard_000.json"
    grouping_manifest_path = tmp_path / "grouping_manifest.json"

    sequence_units = [
        {
            "sequence_id": 0,
            "state_dict": {"pq_idx": 0, "rg_idx": 0, "epoch": 0},
            "sequence_recipe": {
                "row_capacity": 3,
                "rows": [{
                    "segments": [{
                        "source_state": {"pq_idx": 0, "rg_idx": 0, "text_batch_index": 0},
                        "doc_index_in_batch": 0,
                        "start_offset": 0,
                        "end_offset": 3,
                    }],
                }],
            },
            "num_unique_tokens": 2,
            "unique_token_ids": [11, 12],
            "sequence_geometry": {"rows": 1, "tokens": 2},
        },
        {
            "sequence_id": 1,
            "state_dict": {"pq_idx": 1, "rg_idx": 0, "epoch": 0},
            "sequence_recipe": {
                "row_capacity": 3,
                "rows": [{
                    "segments": [{
                        "source_state": {"pq_idx": 1, "rg_idx": 0, "text_batch_index": 0},
                        "doc_index_in_batch": 0,
                        "start_offset": 0,
                        "end_offset": 3,
                    }],
                }],
            },
            "num_unique_tokens": 2,
            "unique_token_ids": [21, 22],
            "sequence_geometry": {"rows": 1, "tokens": 2},
        },
    ]
    save_sequence_manifest_shard(
        base_shard_path,
        build_sequence_manifest_shard_payload(shard_index=0, start_sequence_id=0, sequence_units=sequence_units),
    )
    save_sparse_manifest(
        base_manifest_path,
        build_sequence_manifest_payload(
            split="train",
            vocab_size=128,
            device_batch_size=1,
            max_seq_len=2,
            total_batch_size=1,
            grad_accum_steps=1,
            ddp_world_size=1,
            num_iterations=3,
            buffer_size=1,
            num_sequence_units=2,
            shard_sequence_count=2,
            shards=[
                {
                    "shard_index": 0,
                    "path": str(base_shard_path.relative_to(tmp_path)),
                    "start_sequence_id": 0,
                    "num_sequence_units": 2,
                }
            ],
        ),
    )

    steps = [
        {
            "grad_accum_u_size": 2,
            "grad_accum_active_ids": [11, 12],
            "microsteps": [
                {"microstep": 0, "sequence_id": 0, "u_size": 2, "active_ids": [11, 12], "next_active_ids": [21, 22], "next_u_size": 2, "next_leaving_ids": [11, 12], "next_new_ids": [21, 22]},
            ],
        },
        {
            "grad_accum_u_size": 2,
            "grad_accum_active_ids": [21, 22],
            "microsteps": [
                {"microstep": 0, "sequence_id": 1, "u_size": 2, "active_ids": [21, 22], "next_active_ids": [11, 12], "next_u_size": 2, "next_leaving_ids": [21, 22], "next_new_ids": [11, 12]},
            ],
        },
        {
            "grad_accum_u_size": 2,
            "grad_accum_active_ids": [11, 12],
            "microsteps": [
                {"microstep": 0, "sequence_id": 0, "u_size": 2, "active_ids": [11, 12], "next_active_ids": [11, 12], "next_u_size": 2, "next_leaving_ids": [], "next_new_ids": []},
            ],
        },
    ]
    with grouping_shard_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "version": 4,
                "manifest_kind": "grouping",
                "shard_index": 0,
                "start_step": 0,
                "num_steps": 3,
                "u_max": 2,
                "grad_accum_u_max": 2,
                "steps": steps,
            },
            f,
        )
    save_sparse_manifest(
        grouping_manifest_path,
        build_grouping_manifest_payload(
            split="train",
            vocab_size=128,
            device_batch_size=1,
            max_seq_len=2,
            total_batch_size=1,
            grad_accum_steps=1,
            ddp_world_size=1,
            num_iterations=3,
            buffer_size=1,
            u_max=2,
            grad_accum_u_max=2,
            shard_step_count=3,
            shards=[
                {
                    "shard_index": 0,
                    "path": str(grouping_shard_path.relative_to(tmp_path)),
                    "start_step": 0,
                    "num_steps": 3,
                    "u_max": 2,
                    "grad_accum_u_max": 2,
                }
            ],
            base_manifest_path=str(base_manifest_path.relative_to(tmp_path)),
        ),
    )

    load_calls = []

    def fail_if_live_loader_called(*args, **kwargs):
        raise AssertionError("dual-manifest path should not call the live loader")

    def fake_load_cached_token_batch_by_state(cache_dir, split, *, pq_idx, rg_idx, text_batch_index):
        load_calls.append((pq_idx, rg_idx, text_batch_index))
        if pq_idx == 0:
            return [torch.tensor([11, 12, 11], dtype=torch.long)]
        if pq_idx == 1:
            return [torch.tensor([21, 22, 21], dtype=torch.long)]
        raise AssertionError(f"unexpected pq_idx={pq_idx}")

    monkeypatch.setattr(
        dataloader_module,
        "tokenizing_distributed_data_loader_with_state_bos_bestfit",
        fail_if_live_loader_called,
    )
    monkeypatch.setattr(
        dataloader_module,
        "load_cached_token_batch_by_state",
        fake_load_cached_token_batch_by_state,
    )
    monkeypatch.setattr(dataloader_module, "DUAL_MANIFEST_TOKEN_BATCH_CACHE_LIMIT", 1)

    loader = dataloader_module.tokenizing_distributed_data_loader_with_state_bos_bestfit_manifest(
        tokenizer=None,
        B=1,
        T=2,
        split="train",
        manifest_path=grouping_manifest_path,
        device="cpu",
        vocab_size=128,
        include_local_batch=False,
        use_sequence_base_manifest=True,
        token_cache_dir="",
    )

    next(loader)
    next(loader)
    next(loader)

    assert load_calls == [(0, 0, 0), (1, 0, 0), (0, 0, 0)]


def test_sparse_future_window_planner_tracks_single_occurrence_rolling_maxima(tmp_path):
    base_shard_path = tmp_path / "sequence_base_shard_000.sqlite"
    base_manifest_path = tmp_path / "sequence_base.json"
    grouping_shard_path = tmp_path / "grouping_shard_000.json"
    grouping_manifest_path = tmp_path / "grouping_manifest.json"

    sequence_units = [
        {
            "sequence_id": 0,
            "state_dict": {"pq_idx": 0, "rg_idx": 0, "epoch": 0},
            "sequence_recipe": {"row_capacity": 3, "rows": []},
            "num_unique_tokens": 2,
            "unique_token_ids": [41, 42],
            "sequence_geometry": {"rows": 1, "tokens": 2},
        },
        {
            "sequence_id": 1,
            "state_dict": {"pq_idx": 1, "rg_idx": 0, "epoch": 0},
            "sequence_recipe": {"row_capacity": 3, "rows": []},
            "num_unique_tokens": 2,
            "unique_token_ids": [51, 52],
            "sequence_geometry": {"rows": 1, "tokens": 2},
        },
    ]
    save_sequence_manifest_shard(
        base_shard_path,
        build_sequence_manifest_shard_payload(shard_index=0, start_sequence_id=0, sequence_units=sequence_units),
    )
    save_sparse_manifest(
        base_manifest_path,
        build_sequence_manifest_payload(
            split="train",
            vocab_size=2048,
            device_batch_size=1,
            max_seq_len=2,
            total_batch_size=2,
            grad_accum_steps=2,
            ddp_world_size=1,
            num_iterations=4,
            buffer_size=2,
            num_sequence_units=2,
            shard_sequence_count=2,
            shards=[{"shard_index": 0, "path": str(base_shard_path.relative_to(tmp_path)), "start_sequence_id": 0, "num_sequence_units": 2}],
        ),
    )
    steps = [
        {
            "grad_accum_u_size": 4,
            "grad_accum_active_ids": [41, 42, 51, 52],
            "microsteps": [
                {"microstep": 0, "sequence_id": 0, "u_size": 2, "active_ids": [41, 42], "next_active_ids": [41, 42], "next_u_size": 2, "next_leaving_ids": [], "next_new_ids": []},
                {"microstep": 1, "sequence_id": 1, "u_size": 2, "active_ids": [51, 52], "next_active_ids": [51, 52], "next_u_size": 2, "next_leaving_ids": [], "next_new_ids": []},
            ],
        },
        {
            "grad_accum_u_size": 4,
            "grad_accum_active_ids": [41, 42, 51, 52],
            "microsteps": [
                {"microstep": 0, "sequence_id": 0, "u_size": 2, "active_ids": [41, 42], "next_active_ids": [41, 42], "next_u_size": 2, "next_leaving_ids": [], "next_new_ids": []},
                {"microstep": 1, "sequence_id": 1, "u_size": 2, "active_ids": [51, 52], "next_active_ids": [51, 52], "next_u_size": 2, "next_leaving_ids": [], "next_new_ids": []},
            ],
        },
        {
            "grad_accum_u_size": 4,
            "grad_accum_active_ids": [41, 42, 51, 52],
            "microsteps": [
                {"microstep": 0, "sequence_id": 0, "u_size": 2, "active_ids": [41, 42], "next_active_ids": [41, 42], "next_u_size": 2, "next_leaving_ids": [], "next_new_ids": []},
                {"microstep": 1, "sequence_id": 1, "u_size": 2, "active_ids": [51, 52], "next_active_ids": [51, 52], "next_u_size": 2, "next_leaving_ids": [], "next_new_ids": []},
            ],
        },
        {
            "grad_accum_u_size": 4,
            "grad_accum_active_ids": [41, 42, 51, 52],
            "microsteps": [
                {"microstep": 0, "sequence_id": 0, "u_size": 2, "active_ids": [41, 42], "next_active_ids": [41, 42], "next_u_size": 2, "next_leaving_ids": [], "next_new_ids": []},
                {"microstep": 1, "sequence_id": 1, "u_size": 2, "active_ids": [51, 52], "next_active_ids": [51, 52], "next_u_size": 2, "next_leaving_ids": [], "next_new_ids": []},
            ],
        },
    ]
    with grouping_shard_path.open("w", encoding="utf-8") as f:
        json.dump({"version": 4, "manifest_kind": "grouping", "shard_index": 0, "start_step": 0, "num_steps": 4, "u_max": 3, "grad_accum_u_max": 6, "steps": steps}, f)
    save_sparse_manifest(
        grouping_manifest_path,
        build_grouping_manifest_payload(
            split="train",
            vocab_size=2048,
            device_batch_size=1,
            max_seq_len=2,
            total_batch_size=2,
            grad_accum_steps=2,
            ddp_world_size=1,
            num_iterations=4,
            buffer_size=2,
            u_max=3,
            grad_accum_u_max=6,
            shard_step_count=4,
            shards=[{"shard_index": 0, "path": str(grouping_shard_path.relative_to(tmp_path)), "start_step": 0, "num_steps": 4, "u_max": 3, "grad_accum_u_max": 6}],
            base_manifest_path=str(base_manifest_path.relative_to(tmp_path)),
        ),
    )

    planner = SparseFutureWindowPlanner(
        grouping_manifest_path,
        interval_steps=1,
        rolling_window_steps=2,
        positive_fraction=0.5,
        corrective_fraction=0.0,
    )

    planner.update_from_step_payload(
        0,
        correct_scores=torch.tensor([5.0], dtype=torch.float32),
        correct_records=torch.tensor([[0, 0, 0, 0, 0, 0, 41]], dtype=torch.long),
        incorrect_scores=torch.tensor([1.0], dtype=torch.float32),
        incorrect_records=torch.tensor([[0, 0, 0, 0, 0, 0, 41, 0, 999]], dtype=torch.long),
    )
    planner.update_from_step_payload(
        1,
        correct_scores=torch.tensor([4.0], dtype=torch.float32),
        correct_records=torch.tensor([[1, 0, 0, 0, 0, 0, 51]], dtype=torch.long),
        incorrect_scores=torch.tensor([2.0], dtype=torch.float32),
        incorrect_records=torch.tensor([[1, 0, 0, 0, 0, 0, 51, 0, 888]], dtype=torch.long),
    )

    assert planner.accumulator.correct_totals[41] == 5.0
    assert planner.accumulator.correct_totals[51] == 4.0
    override = planner.get_step_override(2)
    assert override is not None
    assert override["microsteps"][0]["focus_token_id"] == 41

    planner.update_from_step_payload(
        2,
        correct_scores=torch.tensor([4.0], dtype=torch.float32),
        correct_records=torch.tensor([[2, 0, 0, 0, 0, 0, 51]], dtype=torch.long),
        incorrect_scores=torch.tensor([3.0], dtype=torch.float32),
        incorrect_records=torch.tensor([[2, 0, 0, 0, 0, 0, 51, 0, 777]], dtype=torch.long),
    )

    assert 41 not in planner.accumulator.correct_totals
    assert planner.accumulator.correct_totals[51] == 4.0
    override = planner.get_step_override(3)
    assert override is not None
    assert override["microsteps"][0]["focus_token_id"] == 51


def test_sparse_future_window_planner_waits_for_full_rolling_window_before_emitting_overrides(tmp_path):
    base_shard_path = tmp_path / "sequence_base_shard_000.sqlite"
    base_manifest_path = tmp_path / "sequence_base.json"
    grouping_shard_path = tmp_path / "grouping_shard_000.json"
    grouping_manifest_path = tmp_path / "grouping_manifest.json"

    sequence_units = [
        {
            "sequence_id": 0,
            "state_dict": {"pq_idx": 0, "rg_idx": 0, "epoch": 0},
            "sequence_recipe": {"row_capacity": 3, "rows": []},
            "num_unique_tokens": 2,
            "unique_token_ids": [11, 12],
            "sequence_geometry": {"rows": 1, "tokens": 2},
        },
        {
            "sequence_id": 1,
            "state_dict": {"pq_idx": 1, "rg_idx": 0, "epoch": 0},
            "sequence_recipe": {"row_capacity": 3, "rows": []},
            "num_unique_tokens": 2,
            "unique_token_ids": [21, 22],
            "sequence_geometry": {"rows": 1, "tokens": 2},
        },
        {
            "sequence_id": 2,
            "state_dict": {"pq_idx": 2, "rg_idx": 0, "epoch": 0},
            "sequence_recipe": {"row_capacity": 3, "rows": []},
            "num_unique_tokens": 2,
            "unique_token_ids": [31, 32],
            "sequence_geometry": {"rows": 1, "tokens": 2},
        },
    ]
    save_sequence_manifest_shard(
        base_shard_path,
        build_sequence_manifest_shard_payload(shard_index=0, start_sequence_id=0, sequence_units=sequence_units),
    )
    save_sparse_manifest(
        base_manifest_path,
        build_sequence_manifest_payload(
            split="train",
            vocab_size=2048,
            device_batch_size=1,
            max_seq_len=2,
            total_batch_size=2,
            grad_accum_steps=2,
            ddp_world_size=1,
            num_iterations=4,
            buffer_size=2,
            num_sequence_units=3,
            shard_sequence_count=3,
            shards=[{"shard_index": 0, "path": str(base_shard_path.relative_to(tmp_path)), "start_sequence_id": 0, "num_sequence_units": 3}],
        ),
    )
    steps = [
        {
            "grad_accum_u_size": 4,
            "grad_accum_active_ids": [11, 12, 21, 22],
            "microsteps": [
                {"microstep": 0, "sequence_id": 0, "u_size": 2, "active_ids": [11, 12], "next_active_ids": [11, 12], "next_u_size": 2, "next_leaving_ids": [], "next_new_ids": []},
                {"microstep": 1, "sequence_id": 1, "u_size": 2, "active_ids": [21, 22], "next_active_ids": [21, 22], "next_u_size": 2, "next_leaving_ids": [], "next_new_ids": []},
            ],
        },
        {
            "grad_accum_u_size": 4,
            "grad_accum_active_ids": [11, 12, 21, 22],
            "microsteps": [
                {"microstep": 0, "sequence_id": 0, "u_size": 2, "active_ids": [11, 12], "next_active_ids": [11, 12], "next_u_size": 2, "next_leaving_ids": [], "next_new_ids": []},
                {"microstep": 1, "sequence_id": 1, "u_size": 2, "active_ids": [21, 22], "next_active_ids": [21, 22], "next_u_size": 2, "next_leaving_ids": [], "next_new_ids": []},
            ],
        },
        {
            "grad_accum_u_size": 4,
            "grad_accum_active_ids": [21, 22, 31, 32],
            "microsteps": [
                {"microstep": 0, "sequence_id": 1, "u_size": 2, "active_ids": [21, 22], "next_active_ids": [21, 22], "next_u_size": 2, "next_leaving_ids": [], "next_new_ids": []},
                {"microstep": 1, "sequence_id": 2, "u_size": 2, "active_ids": [31, 32], "next_active_ids": [31, 32], "next_u_size": 2, "next_leaving_ids": [], "next_new_ids": []},
            ],
        },
        {
            "grad_accum_u_size": 4,
            "grad_accum_active_ids": [21, 22, 31, 32],
            "microsteps": [
                {"microstep": 0, "sequence_id": 1, "u_size": 2, "active_ids": [21, 22], "next_active_ids": [21, 22], "next_u_size": 2, "next_leaving_ids": [], "next_new_ids": []},
                {"microstep": 1, "sequence_id": 2, "u_size": 2, "active_ids": [31, 32], "next_active_ids": [31, 32], "next_u_size": 2, "next_leaving_ids": [], "next_new_ids": []},
            ],
        },
    ]
    with grouping_shard_path.open("w", encoding="utf-8") as f:
        json.dump({"version": 4, "manifest_kind": "grouping", "shard_index": 0, "start_step": 0, "num_steps": 4, "u_max": 3, "grad_accum_u_max": 6, "steps": steps}, f)
    save_sparse_manifest(
        grouping_manifest_path,
        build_grouping_manifest_payload(
            split="train",
            vocab_size=2048,
            device_batch_size=1,
            max_seq_len=2,
            total_batch_size=2,
            grad_accum_steps=2,
            ddp_world_size=1,
            num_iterations=4,
            buffer_size=2,
            u_max=3,
            grad_accum_u_max=6,
            shard_step_count=4,
            shards=[{"shard_index": 0, "path": str(grouping_shard_path.relative_to(tmp_path)), "start_step": 0, "num_steps": 4, "u_max": 3, "grad_accum_u_max": 6}],
            base_manifest_path=str(base_manifest_path.relative_to(tmp_path)),
        ),
    )

    planner = SparseFutureWindowPlanner(grouping_manifest_path, interval_steps=1, rolling_window_steps=3)
    empty_planned = planner.update_from_step_payload(
        0,
        correct_scores=torch.tensor([5.0], dtype=torch.float32),
        correct_records=torch.tensor([[0, 0, 0, 0, 0, 0, 11]], dtype=torch.long),
        incorrect_scores=torch.tensor([1.0], dtype=torch.float32),
        incorrect_records=torch.tensor([[0, 0, 0, 0, 0, 0, 11, 0, 999]], dtype=torch.long),
    )
    assert empty_planned == {}
    assert planner.get_step_override(1) is None

    empty_planned = planner.update_from_step_payload(
        1,
        correct_scores=torch.tensor([4.0], dtype=torch.float32),
        correct_records=torch.tensor([[1, 0, 0, 0, 0, 0, 21]], dtype=torch.long),
        incorrect_scores=torch.tensor([2.0], dtype=torch.float32),
        incorrect_records=torch.tensor([[1, 0, 0, 0, 0, 0, 21, 0, 888]], dtype=torch.long),
    )
    assert empty_planned == {}
    assert planner.get_step_override(2) is None

    planned = planner.update_from_step_payload(
        2,
        correct_scores=torch.tensor([3.0], dtype=torch.float32),
        correct_records=torch.tensor([[2, 0, 0, 0, 0, 0, 31]], dtype=torch.long),
        incorrect_scores=torch.tensor([3.0], dtype=torch.float32),
        incorrect_records=torch.tensor([[2, 0, 0, 0, 0, 0, 31, 0, 777]], dtype=torch.long),
    )
    assert sorted(planned.keys()) == [3]
    assert planner.get_step_override(3) is not None