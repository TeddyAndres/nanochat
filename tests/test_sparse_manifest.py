import json
import sys
from types import SimpleNamespace

import pytest
import torch

from scripts.build_sparse_manifest import resolve_batch_geometry

from nanochat import dataloader as dataloader_module
from nanochat.sparse_manifest import (
    build_manifest_payload,
    build_manifest_shard_payload,
    build_sharded_manifest_payload,
    load_sparse_manifest_header,
    resolve_sparse_manifest_grad_accum_u_max,
    save_sparse_manifest,
    stream_sparse_manifest_steps,
    validate_sparse_manifest,
)
import nanochat.token_cache as token_cache_module
from nanochat.token_cache import ensure_token_cache, iter_token_batches_from_cache, load_token_cache_metadata


class _FakeTokenizer:
    def get_vocab_size(self):
        return 128

    def get_bos_token_id(self):
        return 99

    def encode(self, text, prepend=None, append=None, num_threads=8):
        rows = []
        for item in text:
            ids = [ord(ch) % 31 + 1 for ch in item]
            if prepend is not None:
                ids.insert(0, int(prepend))
            if append is not None:
                ids.append(int(append))
            rows.append(ids)
        return rows


def _fake_dataset_paths(tmp_path, *names):
    dataset_dir = tmp_path / "dataset"
    return [str(dataset_dir / name) for name in names]


def test_sparse_manifest_header_and_streaming_steps(tmp_path):
    steps = [
        {
            "step": 0,
            "u_size": 3,
            "active_ids": [1, 2, 3],
            "next_common_ids": [2, 3],
            "next_leaving_ids": [1],
            "next_new_ids": [4],
        },
        {
            "step": 1,
            "u_size": 3,
            "active_ids": [2, 3, 4],
            "next_common_ids": [],
            "next_leaving_ids": [],
            "next_new_ids": [],
        },
    ]
    payload = build_manifest_payload(
        split="train",
        vocab_size=16,
        device_batch_size=2,
        max_seq_len=8,
        total_batch_size=16,
        grad_accum_steps=1,
        ddp_world_size=1,
        num_iterations=2,
        buffer_size=32,
        steps=steps,
    )
    manifest_path = tmp_path / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)

    header = load_sparse_manifest_header(manifest_path)
    validate_sparse_manifest(
        header,
        split="train",
        vocab_size=16,
        device_batch_size=2,
        max_seq_len=8,
        grad_accum_steps=1,
        ddp_world_size=1,
        num_iterations=2,
    )

    assert "steps" not in header
    assert header["u_max"] == 3
    assert header["num_steps"] == 2
    assert "tokenizer_batch_size" not in header
    assert "tokenizer_threads" not in header

    streamed_steps = list(stream_sparse_manifest_steps(manifest_path))
    assert streamed_steps == steps

    streamed_tail = list(stream_sparse_manifest_steps(manifest_path, start_step=1))
    assert streamed_tail == steps[1:]


def test_sparse_manifest_v2_accumulation_windows(tmp_path):
    steps = [
        {
            "step": 0,
            "grad_accum_u_size": 4,
            "grad_accum_active_ids": [1, 2, 3, 4],
            "microsteps": [
                {
                    "microstep": 0,
                    "u_size": 3,
                    "active_ids": [1, 2, 3],
                    "next_common_ids": [2, 3],
                    "next_leaving_ids": [1],
                    "next_new_ids": [4],
                },
                {
                    "microstep": 1,
                    "u_size": 3,
                    "active_ids": [2, 3, 4],
                    "next_common_ids": [],
                    "next_leaving_ids": [],
                    "next_new_ids": [],
                },
            ],
        },
    ]
    payload = build_manifest_payload(
        split="train",
        vocab_size=16,
        device_batch_size=2,
        max_seq_len=8,
        total_batch_size=32,
        grad_accum_steps=2,
        ddp_world_size=1,
        num_iterations=1,
        buffer_size=32,
        steps=steps,
    )
    manifest_path = tmp_path / "manifest_v2.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)

    header = load_sparse_manifest_header(manifest_path)
    validate_sparse_manifest(
        header,
        split="train",
        vocab_size=16,
        device_batch_size=2,
        max_seq_len=8,
        grad_accum_steps=2,
        ddp_world_size=1,
        num_iterations=1,
    )

    assert header["version"] == 2
    assert header["u_max"] == 3
    assert header["grad_accum_u_max"] == 4
    assert header["num_steps"] == 1
    assert list(stream_sparse_manifest_steps(manifest_path)) == steps


def test_sparse_manifest_resolves_grad_accum_u_max_from_legacy_v2_steps(tmp_path):
    payload = {
        "version": 2,
        "split": "train",
        "vocab_size": 16,
        "device_batch_size": 2,
        "max_seq_len": 8,
        "total_batch_size": 32,
        "grad_accum_steps": 2,
        "ddp_world_size": 1,
        "num_steps": 2,
        "buffer_size": 32,
        "u_max": 3,
        "steps": [
            {
                "step": 0,
                "grad_accum_u_size": 4,
                "grad_accum_active_ids": [1, 2, 3, 4],
                "microsteps": [
                    {"microstep": 0, "u_size": 3, "active_ids": [1, 2, 3], "next_common_ids": [2, 3], "next_leaving_ids": [1], "next_new_ids": [4]},
                    {"microstep": 1, "u_size": 3, "active_ids": [2, 3, 4], "next_common_ids": [], "next_leaving_ids": [], "next_new_ids": []},
                ],
            },
            {
                "step": 1,
                "grad_accum_u_size": 5,
                "grad_accum_active_ids": [2, 3, 4, 5, 6],
                "microsteps": [
                    {"microstep": 0, "u_size": 3, "active_ids": [2, 3, 4], "next_common_ids": [3, 4], "next_leaving_ids": [2], "next_new_ids": [5]},
                    {"microstep": 1, "u_size": 3, "active_ids": [3, 4, 5], "next_common_ids": [], "next_leaving_ids": [], "next_new_ids": []},
                ],
            },
        ],
    }
    manifest_path = tmp_path / "legacy_manifest_v2.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)

    header = load_sparse_manifest_header(manifest_path)
    assert "grad_accum_u_max" not in header
    assert resolve_sparse_manifest_grad_accum_u_max(manifest_path, header) == 5


def test_sparse_manifest_streams_across_shards(tmp_path):
    steps = [
        {
            "step": 0,
            "grad_accum_u_size": 4,
            "grad_accum_active_ids": [1, 2, 3, 4],
            "microsteps": [
                {
                    "microstep": 0,
                    "u_size": 3,
                    "active_ids": [1, 2, 3],
                    "next_common_ids": [2, 3],
                    "next_leaving_ids": [1],
                    "next_new_ids": [4],
                },
                {
                    "microstep": 1,
                    "u_size": 3,
                    "active_ids": [2, 3, 4],
                    "next_common_ids": [3, 4],
                    "next_leaving_ids": [2],
                    "next_new_ids": [5],
                },
            ],
        },
        {
            "step": 1,
            "grad_accum_u_size": 4,
            "grad_accum_active_ids": [3, 4, 5, 6],
            "microsteps": [
                {
                    "microstep": 0,
                    "u_size": 3,
                    "active_ids": [3, 4, 5],
                    "next_common_ids": [4, 5],
                    "next_leaving_ids": [3],
                    "next_new_ids": [6],
                },
                {
                    "microstep": 1,
                    "u_size": 3,
                    "active_ids": [4, 5, 6],
                    "next_common_ids": [],
                    "next_leaving_ids": [],
                    "next_new_ids": [],
                },
            ],
        },
    ]
    shard_dir = tmp_path / "manifest_shards"
    shard_entries = []
    for shard_index, step_entry in enumerate(steps):
        shard_payload = build_manifest_shard_payload(
            shard_index=shard_index,
            start_step=step_entry["step"],
            steps=[step_entry],
        )
        shard_path = shard_dir / f"manifest.shard{shard_index:05d}.json"
        save_sparse_manifest(shard_path, shard_payload)
        shard_entries.append(
            {
                "shard_index": shard_index,
                "path": str(shard_path.relative_to(tmp_path)),
                "start_step": step_entry["step"],
                "num_steps": 1,
                "u_max": shard_payload["u_max"],
                "grad_accum_u_max": shard_payload["grad_accum_u_max"],
            }
        )

    manifest_path = tmp_path / "manifest.json"
    save_sparse_manifest(
        manifest_path,
        build_sharded_manifest_payload(
            split="train",
            vocab_size=16,
            device_batch_size=2,
            max_seq_len=8,
            total_batch_size=32,
            grad_accum_steps=2,
            ddp_world_size=1,
            num_iterations=2,
            buffer_size=32,
            u_max=3,
            grad_accum_u_max=4,
            shard_step_count=1,
            shards=shard_entries,
        ),
    )

    header = load_sparse_manifest_header(manifest_path)
    validate_sparse_manifest(
        header,
        split="train",
        vocab_size=16,
        device_batch_size=2,
        max_seq_len=8,
        grad_accum_steps=2,
        ddp_world_size=1,
        num_iterations=2,
    )

    assert header["version"] == 3
    assert header["num_shards"] == 2
    assert header["u_max"] == 3
    assert resolve_sparse_manifest_grad_accum_u_max(manifest_path, header) == 4
    assert list(stream_sparse_manifest_steps(manifest_path)) == steps
    assert list(stream_sparse_manifest_steps(manifest_path, start_step=1)) == steps[1:]


def test_token_cache_roundtrip_matches_live_token_batches(tmp_path, monkeypatch):
    fake_batches = [
        (["aa", "bbb"], {"pq_idx": 0, "rg_idx": 0, "epoch": 1, "text_batch_index": 0}),
        (["c", "dddd"], {"pq_idx": 0, "rg_idx": 1, "epoch": 1, "text_batch_index": 0}),
    ]

    def fake_iter_document_text_batches(split, resume_state_dict, tokenizer_batch_size, *, ddp_rank, ddp_world_size):
        assert split == "train"
        assert tokenizer_batch_size == 2
        assert ddp_rank == 0
        assert ddp_world_size == 1
        yield from fake_batches

    monkeypatch.setattr("nanochat.token_cache.iter_document_text_batches", fake_iter_document_text_batches)
    monkeypatch.setattr(
        "nanochat.token_cache.list_parquet_files",
        lambda warn_on_legacy=False: _fake_dataset_paths(tmp_path, "train0.parquet", "val.parquet"),
    )
    tokenizer = _FakeTokenizer()
    ensure_token_cache(
        tmp_path,
        "train",
        tokenizer=tokenizer,
        tokenizer_threads=2,
        tokenizer_batch_size=2,
        bos_token_id=tokenizer.get_bos_token_id(),
        ddp_rank=0,
        ddp_world_size=1,
        shard_batch_count=1,
        num_workers=1,
    )

    def fail_if_materialized(cache_dir, split):
        raise AssertionError("cache replay should stream shards instead of materializing the whole cache")

    monkeypatch.setattr("nanochat.token_cache.load_cached_token_batches", fail_if_materialized)

    cached_iter = iter_token_batches_from_cache(tmp_path, "train")
    first_cached = next(cached_iter)
    second_cached = next(cached_iter)
    assert first_cached[1] == fake_batches[0][1]
    assert second_cached[1] == fake_batches[1][1]
    assert [row.tolist() for row in first_cached[0]] == tokenizer.encode(["aa", "bbb"], prepend=99)
    assert [row.tolist() for row in second_cached[0]] == tokenizer.encode(["c", "dddd"], prepend=99)


def test_loader_cache_path_matches_live_loader(tmp_path, monkeypatch):
    fake_batches = [
        (["aa", "bbb"], {"pq_idx": 0, "rg_idx": 0, "epoch": 1, "text_batch_index": 0}),
        (["cc", "d"], {"pq_idx": 0, "rg_idx": 1, "epoch": 1, "text_batch_index": 0}),
        (["eee", "f"], {"pq_idx": 0, "rg_idx": 2, "epoch": 1, "text_batch_index": 0}),
    ]

    def fake_iter_document_text_batches(split, resume_state_dict, tokenizer_batch_size, *, ddp_rank, ddp_world_size):
        for batch in fake_batches:
            yield batch
        for text_batch, state in fake_batches:
            yield text_batch, {**state, "epoch": 2}

    monkeypatch.setattr("nanochat.token_cache.iter_document_text_batches", fake_iter_document_text_batches)
    monkeypatch.setattr(
        "nanochat.token_cache.list_parquet_files",
        lambda warn_on_legacy=False: _fake_dataset_paths(tmp_path, "train0.parquet", "val.parquet"),
    )
    monkeypatch.setattr(dataloader_module, "iter_document_text_batches", fake_iter_document_text_batches)
    tokenizer = _FakeTokenizer()
    live_loader = dataloader_module.tokenizing_distributed_data_loader_with_state_bos_bestfit_dynamic(
        tokenizer,
        2,
        4,
        split="train",
        tokenizer_threads=2,
        tokenizer_batch_size=2,
        device="cpu",
        resume_state_dict=None,
        buffer_size=2,
        vocab_size=128,
    )
    live_inputs, live_targets, live_active_ids, live_state = next(live_loader)

    ensure_token_cache(
        tmp_path,
        "train",
        tokenizer=tokenizer,
        tokenizer_threads=2,
        tokenizer_batch_size=2,
        bos_token_id=tokenizer.get_bos_token_id(),
        ddp_rank=0,
        ddp_world_size=1,
        shard_batch_count=2,
        num_workers=1,
    )
    cache_loader = dataloader_module.tokenizing_distributed_data_loader_with_state_bos_bestfit_dynamic(
        tokenizer,
        2,
        4,
        split="train",
        tokenizer_threads=2,
        tokenizer_batch_size=2,
        device="cpu",
        resume_state_dict=None,
        buffer_size=2,
        vocab_size=128,
        token_cache_dir=tmp_path,
        token_cache_shard_batches=2,
        token_cache_workers=1,
    )
    cache_inputs, cache_targets, cache_active_ids, cache_state = next(cache_loader)

    assert torch.equal(live_inputs, cache_inputs)
    assert torch.equal(live_targets, cache_targets)
    assert torch.equal(live_active_ids, cache_active_ids)
    assert live_state == cache_state


def test_token_cache_replay_skips_resumed_row_group(tmp_path, monkeypatch):
    fake_batches = [
        (["aa"], {"pq_idx": 0, "rg_idx": 0, "epoch": 1, "text_batch_index": 0}),
        (["bb"], {"pq_idx": 0, "rg_idx": 2, "epoch": 1, "text_batch_index": 0}),
        (["cc"], {"pq_idx": 1, "rg_idx": 0, "epoch": 1, "text_batch_index": 0}),
    ]

    def fake_iter_document_text_batches(split, resume_state_dict, tokenizer_batch_size, *, ddp_rank, ddp_world_size):
        for batch in fake_batches:
            yield batch
        yield ["epoch2"], {"pq_idx": 0, "rg_idx": 0, "epoch": 2}

    monkeypatch.setattr("nanochat.token_cache.iter_document_text_batches", fake_iter_document_text_batches)
    monkeypatch.setattr(
        "nanochat.token_cache.list_parquet_files",
        lambda warn_on_legacy=False: _fake_dataset_paths(tmp_path, "train0.parquet", "train1.parquet", "val.parquet"),
    )
    tokenizer = _FakeTokenizer()
    ensure_token_cache(
        tmp_path,
        "train",
        tokenizer=tokenizer,
        tokenizer_threads=1,
        tokenizer_batch_size=1,
        bos_token_id=tokenizer.get_bos_token_id(),
        ddp_rank=0,
        ddp_world_size=1,
        shard_batch_count=2,
        num_workers=1,
    )

    replay = iter_token_batches_from_cache(
        tmp_path,
        "train",
        resume_state_dict={"pq_idx": 0, "rg_idx": 0, "epoch": 1},
    )
    first_tokens, first_state = next(replay)
    second_tokens, second_state = next(replay)

    assert [row.tolist() for row in first_tokens] == tokenizer.encode(["bb"], prepend=99)
    assert first_state == {"pq_idx": 0, "rg_idx": 2, "epoch": 1, "text_batch_index": 0}
    assert [row.tolist() for row in second_tokens] == tokenizer.encode(["cc"], prepend=99)
    assert second_state == {"pq_idx": 1, "rg_idx": 0, "epoch": 1, "text_batch_index": 0}
    with pytest.raises(StopIteration):
        next(replay)


def test_shared_cache_replays_ddp_row_group_stride(tmp_path, monkeypatch):
    all_batches = [
        (["aa"], {"pq_idx": 0, "rg_idx": 0, "epoch": 1, "text_batch_index": 0}),
        (["bb"], {"pq_idx": 0, "rg_idx": 1, "epoch": 1, "text_batch_index": 0}),
        (["cc"], {"pq_idx": 0, "rg_idx": 2, "epoch": 1, "text_batch_index": 0}),
        (["dd"], {"pq_idx": 0, "rg_idx": 3, "epoch": 1, "text_batch_index": 0}),
    ]

    def fake_iter_document_text_batches(split, resume_state_dict, tokenizer_batch_size, *, ddp_rank, ddp_world_size):
        del resume_state_dict, tokenizer_batch_size, ddp_rank, ddp_world_size
        yield from all_batches
        yield ["epoch2"], {"pq_idx": 0, "rg_idx": 0, "epoch": 2, "text_batch_index": 0}

    monkeypatch.setattr("nanochat.token_cache.iter_document_text_batches", fake_iter_document_text_batches)
    monkeypatch.setattr(
        "nanochat.token_cache.list_parquet_files",
        lambda warn_on_legacy=False: _fake_dataset_paths(tmp_path, "train0.parquet", "val.parquet"),
    )
    tokenizer = _FakeTokenizer()

    ensure_token_cache(
        tmp_path,
        "train",
        tokenizer=tokenizer,
        tokenizer_threads=1,
        tokenizer_batch_size=1,
        bos_token_id=tokenizer.get_bos_token_id(),
        ddp_rank=0,
        ddp_world_size=1,
        shard_batch_count=1,
        num_workers=1,
    )

    metadata = load_token_cache_metadata(tmp_path, "train")
    assert metadata is not None
    assert metadata["complete"] is True
    assert len(metadata["files"]) == 1

    rank0 = list(iter_token_batches_from_cache(tmp_path, "train", ddp_rank=0, ddp_world_size=2))
    rank1 = list(iter_token_batches_from_cache(tmp_path, "train", ddp_rank=1, ddp_world_size=2))

    assert [state["rg_idx"] for _, state in rank0] == [0, 2]
    assert [state["rg_idx"] for _, state in rank1] == [1, 3]
    assert [row.tolist() for row in rank0[0][0]] == tokenizer.encode(["aa"], prepend=99)
    assert [row.tolist() for row in rank1[1][0]] == tokenizer.encode(["dd"], prepend=99)


def test_prepare_token_cache_writer_refuses_to_overwrite_incompatible_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "nanochat.token_cache.list_parquet_files",
        lambda warn_on_legacy=False: _fake_dataset_paths(tmp_path, "train0.parquet", "val.parquet"),
    )
    split_dir = tmp_path / "train"
    split_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = split_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "version": 1,
                "split": "train",
                "tokenizer_batch_size": 128,
                "vocab_size": 128,
                "bos_token_id": 99,
                "ddp_world_size": 1,
                "shards": [{"path": "token_cache.00000.pt", "num_batches": 1}],
            }
        ),
        encoding="utf-8",
    )
    (split_dir / "token_cache.00000.pt").write_bytes(b"placeholder")

    with pytest.raises(ValueError, match="Refusing to overwrite existing token cache contents"):
        dataloader_module.prepare_token_cache_writer(
            tmp_path,
            "train",
            tokenizer=_FakeTokenizer(),
            tokenizer_batch_size=64,
            bos_token_id=99,
            ddp_world_size=1,
            shard_batch_count=1,
        )


def test_resolve_token_cache_dir_rejects_non_dataset_storage(tmp_path, monkeypatch):
    dataset_paths = _fake_dataset_paths(tmp_path, "train0.parquet", "val.parquet")
    monkeypatch.setattr("nanochat.token_cache.list_parquet_files", lambda warn_on_legacy=False: dataset_paths)

    with pytest.raises(ValueError, match="dataset-side storage"):
        dataloader_module.resolve_token_cache_dir("/tmp/not-on-dataset-storage")


def test_base_train_main_module_disables_spawn_workers(monkeypatch):
    fake_main = SimpleNamespace(__file__="/home/teddy/Desktop/dev/repo/nanochat/scripts/base_train.py", __spec__=SimpleNamespace(name="scripts.base_train"))
    monkeypatch.setitem(sys.modules, "__main__", fake_main)

    assert token_cache_module._main_module_supports_spawn_workers() is False


def test_resolve_batch_geometry_defaults_to_one_microbatch():
    total_batch_size, grad_accum_steps = resolve_batch_geometry(
        device_batch_size=2,
        max_seq_len=8,
        total_batch_size=-1,
        grad_accum_steps=-1,
    )

    assert total_batch_size == 16
    assert grad_accum_steps == 1


def test_resolve_batch_geometry_accepts_explicit_grad_accum_steps():
    total_batch_size, grad_accum_steps = resolve_batch_geometry(
        device_batch_size=2,
        max_seq_len=8,
        total_batch_size=-1,
        grad_accum_steps=4,
    )

    assert total_batch_size == 64
    assert grad_accum_steps == 4


def test_resolve_batch_geometry_rejects_inconsistent_inputs():
    try:
        resolve_batch_geometry(
            device_batch_size=2,
            max_seq_len=8,
            total_batch_size=32,
            grad_accum_steps=3,
        )
    except ValueError as exc:
        assert "grad_accum_steps mismatch" in str(exc)
    else:
        raise AssertionError("Expected resolve_batch_geometry to reject inconsistent inputs")
