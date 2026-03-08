import json

from nanochat.sparse_manifest import (
    build_manifest_payload,
    load_sparse_manifest_header,
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
        tokenizer_batch_size=8,
        tokenizer_threads=2,
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

    streamed_steps = list(stream_sparse_manifest_steps(manifest_path))
    assert streamed_steps == steps

    streamed_tail = list(stream_sparse_manifest_steps(manifest_path, start_step=1))
    assert streamed_tail == steps[1:]
