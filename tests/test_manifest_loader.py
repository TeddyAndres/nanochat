import json

import torch

from nanochat import dataloader


class DummyTokenizer:
    def get_bos_token_id(self):
        return 0


def test_grouping_manifest_uses_embedded_base_sequence_manifest(tmp_path, monkeypatch):
    batch_size = 1
    max_seq_len = 4
    vocab_size = 32

    def fake_live_loader(*args, **kwargs):
        while True:
            yield (
                torch.tensor([[10, 10, 10, 10]], dtype=torch.long),
                torch.tensor([[11, 11, 11, 11]], dtype=torch.long),
                {"pq_idx": 9, "rg_idx": 9, "epoch": 1},
            )

    monkeypatch.setattr(
        dataloader,
        "tokenizing_distributed_data_loader_with_state_bos_bestfit",
        fake_live_loader,
    )

    grouping_manifest_path = tmp_path / "tiny_grouping.json"
    grouping_shard_path = tmp_path / "tiny_grouping.shard00000.json"
    base_manifest_path = tmp_path / "tiny_grouping.base.json"
    base_shard_path = tmp_path / "tiny_grouping.base.shard00000.pt"

    with grouping_shard_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "version": 3,
                "shard_index": 0,
                "start_step": 0,
                "num_steps": 1,
                "u_max": 5,
                "grad_accum_u_max": 5,
                "steps": [
                    {
                        "grad_accum_u_size": 5,
                        "grad_accum_active_ids": [10, 11, 12, 13, 14],
                        "microsteps": [
                            {
                                "u_size": 5,
                                "active_ids": [10, 11, 12, 13, 14],
                                "next_leaving_ids": [],
                                "next_new_ids": [],
                                "sequence_id": 0,
                            }
                        ],
                    }
                ],
            },
            f,
        )

    torch.save(
        {
            "version": 4,
            "manifest_kind": "sequence-base",
            "shard_index": 0,
            "start_sequence_id": 0,
            "num_sequence_units": 1,
            "sequence_units": [
                {
                    "sequence_id": 0,
                    "inputs": [[10, 11, 12, 13]],
                    "targets": [[11, 12, 13, 14]],
                    "state_dict": {"pq_idx": 1, "rg_idx": 2, "epoch": 3},
                }
            ],
        },
        base_shard_path,
    )

    with base_manifest_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "version": 4,
                "manifest_kind": "sequence-base",
                "split": "train",
                "vocab_size": vocab_size,
                "device_batch_size": batch_size,
                "max_seq_len": max_seq_len,
                "total_batch_size": batch_size * max_seq_len,
                "grad_accum_steps": 1,
                "ddp_world_size": 1,
                "num_steps": 1,
                "buffer_size": 1,
                "num_sequence_units": 1,
                "shard_sequence_count": 1,
                "num_shards": 1,
                "shards": [
                    {
                        "shard_index": 0,
                        "path": base_shard_path.name,
                        "start_sequence_id": 0,
                        "num_sequence_units": 1,
                    }
                ],
            },
            f,
        )

    with grouping_manifest_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "version": 4,
                "manifest_kind": "grouping",
                "base_manifest_path": base_manifest_path.name,
                "split": "train",
                "vocab_size": vocab_size,
                "device_batch_size": batch_size,
                "max_seq_len": max_seq_len,
                "total_batch_size": batch_size * max_seq_len,
                "grad_accum_steps": 1,
                "ddp_world_size": 1,
                "num_steps": 1,
                "buffer_size": 1,
                "u_max": 5,
                "grad_accum_u_max": 5,
                "shard_step_count": 1,
                "num_shards": 1,
                "shards": [
                    {
                        "shard_index": 0,
                        "path": grouping_shard_path.name,
                        "start_step": 0,
                        "num_steps": 1,
                        "u_max": 5,
                        "grad_accum_u_max": 5,
                    }
                ],
            },
            f,
        )

    loader = dataloader.tokenizing_distributed_data_loader_with_state_bos_bestfit_manifest(
        DummyTokenizer(),
        batch_size,
        max_seq_len,
        "train",
        manifest_path=str(grouping_manifest_path),
        device="cpu",
        vocab_size=vocab_size,
    )

    inputs, targets, step_meta, state_dict = next(loader)

    assert torch.equal(inputs, torch.tensor([[0, 1, 2, 3]], dtype=torch.long))
    assert torch.equal(targets, torch.tensor([[1, 2, 3, 4]], dtype=torch.long))
    assert step_meta["sequence_id"] == 0
    assert state_dict["manifest_step"] == 0