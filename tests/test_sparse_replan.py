import json

import torch

from nanochat import dataloader as dataloader_module
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
                "u_max": 2,
                "grad_accum_u_max": 8,
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
            u_max=2,
            grad_accum_u_max=8,
            shard_step_count=len(steps),
            shards=[
                {
                    "shard_index": 0,
                    "path": str(grouping_shard_path.relative_to(tmp_path)),
                    "start_step": 0,
                    "num_steps": len(steps),
                    "u_max": 2,
                    "grad_accum_u_max": 8,
                }
            ],
            base_manifest_path=str(base_manifest_path.relative_to(tmp_path)),
        ),
    )

    planner = SparseFutureWindowPlanner(
        grouping_manifest_path,
        lookahead_steps=1,
        window_steps=2,
    )
    correct_records = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0, 52],
            [0, 0, 0, 0, 0, 0, -1],
        ],
        dtype=torch.long,
    )
    incorrect_records = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0, 72, 0, 999],
            [0, 0, 0, 0, 0, 0, -1, -1, -1],
        ],
        dtype=torch.long,
    )

    planned = planner.update_from_topk(0, correct_records=correct_records, incorrect_records=incorrect_records)

    assert sorted(planned.keys()) == [1, 2]
    step1 = planned[1]
    assert len(step1["microsteps"]) == 4
    assert step1["microsteps"][0]["planner_bucket"] == "positive"
    assert step1["microsteps"][0]["sequence_id"] == 4
    assert step1["microsteps"][0]["focus_token_id"] == 52
    assert step1["microsteps"][1]["planner_bucket"] == "corrective"
    assert step1["microsteps"][1]["sequence_id"] == 6
    assert step1["microsteps"][1]["injected_negative_ids"] == [999]
    assert 999 in step1["microsteps"][1]["active_ids"]
    assert [microstep["sequence_id"] for microstep in step1["microsteps"][2:]] == [0, 1]

    step2 = planned[2]
    assert [microstep["sequence_id"] for microstep in step2["microsteps"]] == [2, 3, 5, 7]


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

    assert torch.equal(targets1, torch.tensor([[2, 0]], dtype=torch.long))
    assert torch.equal(step_meta1["active_ids_cpu"], torch.tensor([4, 2, 3, 9], dtype=torch.long))
    assert torch.equal(step_meta1["stage_ids_cpu"], torch.tensor([4, 9], dtype=torch.long))
    assert state1["manifest_step"] == 1