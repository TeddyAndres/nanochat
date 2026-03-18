import json

from scripts.build_sparse_manifest import resolve_batch_geometry

from nanochat.sparse_manifest import (
    build_manifest_payload,
    load_sparse_manifest_header,
    resolve_sparse_manifest_grad_accum_u_max,
    stream_sparse_manifest_steps,
    validate_sparse_manifest,
)


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
