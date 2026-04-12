import argparse
import json
from pathlib import Path

from nanochat.common import print0
from nanochat.sparse_manifest import (
    SequenceManifestShardAccessor,
    load_sequence_manifest_shard,
    load_sparse_manifest_header,
    save_sequence_manifest_shard,
    save_sparse_manifest,
    validate_sequence_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert sequence-base manifest shards from legacy .pt to SQLite-backed random-access shards")
    parser.add_argument("--base-manifest", type=str, required=True, help="path to the base sequence manifest JSON header")
    parser.add_argument("--output", type=str, default="", help="optional output path for the converted base manifest JSON (default: overwrite in place)")
    parser.add_argument("--keep-legacy-shards", action="store_true", help="do not delete old .pt shard files after conversion")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base_manifest_path = Path(args.base_manifest)
    output_manifest_path = base_manifest_path if args.output == "" else Path(args.output)

    header = load_sparse_manifest_header(base_manifest_path)
    validate_sequence_manifest(
        header,
        split=str(header["split"]),
        vocab_size=int(header["vocab_size"]),
        device_batch_size=int(header["device_batch_size"]),
        max_seq_len=int(header["max_seq_len"]),
        ddp_world_size=int(header["ddp_world_size"]),
        num_iterations=int(header["num_steps"]),
    )

    shards = header.get("shards")
    assert isinstance(shards, list) and len(shards) > 0
    converted_header = dict(header)
    converted_shards = []

    for shard_entry in shards:
        shard_entry = dict(shard_entry)
        shard_rel_path = str(shard_entry["path"])
        shard_path = base_manifest_path.parent / shard_rel_path
        if shard_path.suffix == ".sqlite":
            converted_shards.append(shard_entry)
            continue

        payload = load_sequence_manifest_shard(shard_path)
        sqlite_path = shard_path.with_suffix(".sqlite")
        save_sequence_manifest_shard(sqlite_path, payload)
        shard_entry["path"] = str(sqlite_path.relative_to(output_manifest_path.parent))
        converted_shards.append(shard_entry)
        print0(f"Converted {shard_path} -> {sqlite_path}")
        if not args.keep_legacy_shards:
            shard_path.unlink(missing_ok=True)

    converted_header["shards"] = converted_shards
    save_sparse_manifest(output_manifest_path, converted_header)
    print0(f"Wrote converted base manifest header to {output_manifest_path}")

    accessor = SequenceManifestShardAccessor(output_manifest_path.parent / converted_shards[0]["path"])
    try:
        first_sequence_id = int(converted_shards[0]["start_sequence_id"])
        first_unit = accessor.get_sequence_unit(first_sequence_id)
        if first_unit is None:
            raise ValueError("Converted SQLite sequence store is missing the first sequence unit")
        print0(
            f"Verified first converted sequence unit: sequence_id={first_sequence_id} | "
            f"unique_tokens={len(first_unit.get('unique_token_ids', []))}"
        )
    finally:
        accessor.close()


if __name__ == "__main__":
    main()