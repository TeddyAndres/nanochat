from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import torch


SPARSE_MANIFEST_VERSION = 3
DUAL_SPARSE_MANIFEST_VERSION = 4
SUPPORTED_SPARSE_MANIFEST_VERSIONS = {1, 2, 3, 4}
SPARSE_GROUPING_MANIFEST_KIND = "grouping"
SPARSE_SEQUENCE_BASE_MANIFEST_KIND = "sequence-base"


def _validate_manifest_version(payload: dict[str, Any]) -> dict[str, Any]:
    version = int(payload.get("version", -1))
    if version not in SUPPORTED_SPARSE_MANIFEST_VERSIONS:
        raise ValueError(
            f"Unsupported sparse manifest version {version}; expected one of {sorted(SUPPORTED_SPARSE_MANIFEST_VERSIONS)}"
        )
    return payload


def get_sparse_manifest_kind(payload: dict[str, Any]) -> str:
    version = int(payload.get("version", 1))
    if version < DUAL_SPARSE_MANIFEST_VERSION:
        return SPARSE_GROUPING_MANIFEST_KIND
    manifest_kind = payload.get("manifest_kind", SPARSE_GROUPING_MANIFEST_KIND)
    if not isinstance(manifest_kind, str) or manifest_kind == "":
        raise ValueError("Sparse manifest kind must be a non-empty string")
    return manifest_kind


def tensor_ids_to_list(ids: torch.Tensor) -> list[int]:
    ids = ids.detach().to(device="cpu", dtype=torch.long)
    return [int(x) for x in ids.tolist()]


def compute_next_transition(current_ids: torch.Tensor, next_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    current_ids = current_ids.detach().to(device="cpu", dtype=torch.long)
    next_ids = next_ids.detach().to(device="cpu", dtype=torch.long)
    if current_ids.numel() == 0:
        empty = current_ids
        return empty, empty, next_ids.clone()
    if next_ids.numel() == 0:
        empty = next_ids
        return empty, current_ids.clone(), empty
    current_in_next = torch.isin(current_ids, next_ids)
    next_in_current = torch.isin(next_ids, current_ids)
    common_ids = current_ids[current_in_next]
    leaving_ids = current_ids[~current_in_next]
    new_ids = next_ids[~next_in_current]
    return common_ids, leaving_ids, new_ids


def _compute_manifest_u_stats(steps: list[dict[str, Any]]) -> tuple[int, int, int]:
    first_step = steps[0] if steps else None
    version = 1
    grad_accum_u_max = 0
    if first_step is not None and "microsteps" in first_step:
        version = 2
        u_max = max(
            (
                int(microstep["u_size"])
                for step in steps
                for microstep in step.get("microsteps", [])
            ),
            default=0,
        )
        grad_accum_u_max = max(
            (int(step.get("grad_accum_u_size", 0)) for step in steps),
            default=u_max,
        )
    else:
        u_max = max((int(step["u_size"]) for step in steps), default=0)
        grad_accum_u_max = u_max
    return version, int(u_max), int(grad_accum_u_max)


def build_manifest_payload(
    *,
    split: str,
    vocab_size: int,
    device_batch_size: int,
    max_seq_len: int,
    total_batch_size: int,
    grad_accum_steps: int,
    ddp_world_size: int,
    num_iterations: int,
    buffer_size: int,
    steps: list[dict[str, Any]],
) -> dict[str, Any]:
    version, u_max, grad_accum_u_max = _compute_manifest_u_stats(steps)
    return {
        "version": version,
        "split": split,
        "vocab_size": int(vocab_size),
        "device_batch_size": int(device_batch_size),
        "max_seq_len": int(max_seq_len),
        "total_batch_size": int(total_batch_size),
        "grad_accum_steps": int(grad_accum_steps),
        "ddp_world_size": int(ddp_world_size),
        "num_steps": int(num_iterations),
        "buffer_size": int(buffer_size),
        "u_max": int(u_max),
        "grad_accum_u_max": int(grad_accum_u_max),
        "steps": steps,
    }


def build_manifest_shard_payload(
    *,
    shard_index: int,
    start_step: int,
    steps: list[dict[str, Any]],
) -> dict[str, Any]:
    _, u_max, grad_accum_u_max = _compute_manifest_u_stats(steps)
    return {
        "version": SPARSE_MANIFEST_VERSION,
        "shard_index": int(shard_index),
        "start_step": int(start_step),
        "num_steps": len(steps),
        "u_max": int(u_max),
        "grad_accum_u_max": int(grad_accum_u_max),
        "steps": steps,
    }


def build_sharded_manifest_payload(
    *,
    split: str,
    vocab_size: int,
    device_batch_size: int,
    max_seq_len: int,
    total_batch_size: int,
    grad_accum_steps: int,
    ddp_world_size: int,
    num_iterations: int,
    buffer_size: int,
    u_max: int,
    grad_accum_u_max: int,
    shard_step_count: int,
    shards: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "version": SPARSE_MANIFEST_VERSION,
        "split": split,
        "vocab_size": int(vocab_size),
        "device_batch_size": int(device_batch_size),
        "max_seq_len": int(max_seq_len),
        "total_batch_size": int(total_batch_size),
        "grad_accum_steps": int(grad_accum_steps),
        "ddp_world_size": int(ddp_world_size),
        "num_steps": int(num_iterations),
        "buffer_size": int(buffer_size),
        "u_max": int(u_max),
        "grad_accum_u_max": int(grad_accum_u_max),
        "shard_step_count": int(shard_step_count),
        "num_shards": len(shards),
        "shards": shards,
    }


def build_sequence_manifest_shard_payload(
    *,
    shard_index: int,
    start_sequence_id: int,
    sequence_units: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "version": DUAL_SPARSE_MANIFEST_VERSION,
        "manifest_kind": SPARSE_SEQUENCE_BASE_MANIFEST_KIND,
        "shard_index": int(shard_index),
        "start_sequence_id": int(start_sequence_id),
        "num_sequence_units": len(sequence_units),
        "sequence_units": sequence_units,
    }


def build_sequence_manifest_payload(
    *,
    split: str,
    vocab_size: int,
    device_batch_size: int,
    max_seq_len: int,
    total_batch_size: int,
    grad_accum_steps: int,
    ddp_world_size: int,
    num_iterations: int,
    buffer_size: int,
    num_sequence_units: int,
    shard_sequence_count: int,
    shards: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "version": DUAL_SPARSE_MANIFEST_VERSION,
        "manifest_kind": SPARSE_SEQUENCE_BASE_MANIFEST_KIND,
        "split": split,
        "vocab_size": int(vocab_size),
        "device_batch_size": int(device_batch_size),
        "max_seq_len": int(max_seq_len),
        "total_batch_size": int(total_batch_size),
        "grad_accum_steps": int(grad_accum_steps),
        "ddp_world_size": int(ddp_world_size),
        "num_steps": int(num_iterations),
        "buffer_size": int(buffer_size),
        "num_sequence_units": int(num_sequence_units),
        "shard_sequence_count": int(shard_sequence_count),
        "num_shards": len(shards),
        "shards": shards,
    }


def build_grouping_manifest_payload(
    *,
    split: str,
    vocab_size: int,
    device_batch_size: int,
    max_seq_len: int,
    total_batch_size: int,
    grad_accum_steps: int,
    ddp_world_size: int,
    num_iterations: int,
    buffer_size: int,
    u_max: int,
    grad_accum_u_max: int,
    shard_step_count: int,
    shards: list[dict[str, Any]],
    base_manifest_path: str,
) -> dict[str, Any]:
    return {
        "version": DUAL_SPARSE_MANIFEST_VERSION,
        "manifest_kind": SPARSE_GROUPING_MANIFEST_KIND,
        "base_manifest_path": base_manifest_path,
        "split": split,
        "vocab_size": int(vocab_size),
        "device_batch_size": int(device_batch_size),
        "max_seq_len": int(max_seq_len),
        "total_batch_size": int(total_batch_size),
        "grad_accum_steps": int(grad_accum_steps),
        "ddp_world_size": int(ddp_world_size),
        "num_steps": int(num_iterations),
        "buffer_size": int(buffer_size),
        "u_max": int(u_max),
        "grad_accum_u_max": int(grad_accum_u_max),
        "shard_step_count": int(shard_step_count),
        "num_shards": len(shards),
        "shards": shards,
    }


def save_sparse_manifest(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)


def save_sequence_manifest_shard(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_sequence_manifest_shard(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Sequence manifest shard must deserialize to a dict payload")
    _validate_manifest_version(payload)
    if get_sparse_manifest_kind(payload) != SPARSE_SEQUENCE_BASE_MANIFEST_KIND:
        raise ValueError("Sequence manifest shard payload must have kind 'sequence-base'")
    return payload


def _resolve_shard_path(manifest_path: str | Path, shard_entry: dict[str, Any]) -> Path:
    shard_rel_path = shard_entry.get("path")
    if not isinstance(shard_rel_path, str) or shard_rel_path == "":
        raise ValueError("Sparse manifest shard entry must define a non-empty relative path")
    return Path(manifest_path).parent / shard_rel_path


def load_sparse_manifest_header(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    marker = '"steps"'
    with path.open("r", encoding="utf-8") as f:
        buffer = ""
        while True:
            chunk = f.read(1 << 16)
            if chunk == "":
                payload = json.loads(buffer)
                return _validate_manifest_version(payload)
            buffer += chunk
            marker_idx = buffer.find(marker)
            if marker_idx < 0:
                continue
            colon_idx = buffer.find(":", marker_idx + len(marker))
            bracket_idx = buffer.find("[", colon_idx + 1)
            if colon_idx < 0 or bracket_idx < 0:
                continue
            header_text = buffer[:marker_idx].rstrip()
            if header_text.endswith(","):
                header_text = header_text[:-1]
            payload = json.loads(header_text + "}")
            return _validate_manifest_version(payload)


def resolve_sparse_manifest_grad_accum_u_max(path: str | Path, header: dict[str, Any] | None = None) -> int:
    path = Path(path)
    if header is None:
        header = load_sparse_manifest_header(path)

    if get_sparse_manifest_kind(header) != SPARSE_GROUPING_MANIFEST_KIND:
        raise ValueError("grad_accum_u_max is only defined for grouping sparse manifests")

    u_max = int(header.get("u_max", 0))
    if u_max <= 0:
        raise ValueError("Sparse manifest must define a positive u_max")

    header_grad_accum_u_max = header.get("grad_accum_u_max")
    if header_grad_accum_u_max is not None:
        grad_accum_u_max = int(header_grad_accum_u_max)
        if grad_accum_u_max < u_max:
            raise ValueError(
                f"Sparse manifest grad_accum_u_max must be at least u_max, found grad_accum_u_max={grad_accum_u_max}, u_max={u_max}"
            )
        return grad_accum_u_max

    version = int(header.get("version", 1))
    if version < 2:
        return u_max

    shards = header.get("shards")
    if isinstance(shards, list) and len(shards) > 0:
        shard_grad_accum_u_max = u_max
        found_shard_stat = False
        for shard_entry in shards:
            shard_u_max = int(shard_entry.get("u_max", 0))
            if shard_u_max > 0:
                shard_grad_accum_u_max = max(shard_grad_accum_u_max, shard_u_max)
            shard_grad = shard_entry.get("grad_accum_u_max")
            if shard_grad is not None:
                shard_grad_accum_u_max = max(shard_grad_accum_u_max, int(shard_grad))
                found_shard_stat = True
        if found_shard_stat:
            return shard_grad_accum_u_max

    grad_accum_u_max = u_max
    found_grad_accum_window = False
    for step_idx, step_entry in enumerate(stream_sparse_manifest_steps(path)):
        if "grad_accum_u_size" not in step_entry:
            continue
        grad_accum_u_size = int(step_entry["grad_accum_u_size"])
        grad_accum_ids = step_entry.get("grad_accum_active_ids")
        if isinstance(grad_accum_ids, list) and len(grad_accum_ids) != grad_accum_u_size:
            raise ValueError(
                f"Sparse manifest step {step_idx} grad_accum_u_size mismatch: expected {grad_accum_u_size}, found {len(grad_accum_ids)} ids"
            )
        grad_accum_u_max = max(grad_accum_u_max, grad_accum_u_size)
        found_grad_accum_window = True

    return grad_accum_u_max if found_grad_accum_window else u_max


def _stream_sparse_manifest_file_steps(path: str | Path, start_step: int = 0) -> Iterator[dict[str, Any]]:
    path = Path(path)
    if start_step < 0:
        raise ValueError(f"Sparse manifest start_step must be non-negative, got {start_step}")

    marker = '"steps"'
    decoder = json.JSONDecoder()

    with path.open("r", encoding="utf-8") as f:
        buffer = ""
        while True:
            chunk = f.read(1 << 16)
            if chunk == "":
                raise ValueError("Sparse manifest is missing the steps array")
            buffer += chunk
            marker_idx = buffer.find(marker)
            if marker_idx < 0:
                continue
            colon_idx = buffer.find(":", marker_idx + len(marker))
            bracket_idx = buffer.find("[", colon_idx + 1)
            if colon_idx < 0 or bracket_idx < 0:
                continue
            header_text = buffer[:marker_idx].rstrip()
            if header_text.endswith(","):
                header_text = header_text[:-1]
            _validate_manifest_version(json.loads(header_text + "}"))
            buffer = buffer[bracket_idx + 1:]
            break

        step_idx = 0
        while True:
            while True:
                stripped = buffer.lstrip()
                consumed = len(buffer) - len(stripped)
                if consumed > 0:
                    buffer = stripped
                    continue
                if buffer.startswith(","):
                    buffer = buffer[1:]
                    continue
                if buffer.startswith("]"):
                    return
                if buffer:
                    break
                chunk = f.read(1 << 16)
                if chunk == "":
                    raise ValueError("Sparse manifest ended unexpectedly while reading steps")
                buffer += chunk

            while True:
                try:
                    step_entry, end_idx = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError:
                    chunk = f.read(1 << 16)
                    if chunk == "":
                        raise ValueError("Sparse manifest ended unexpectedly while decoding a step entry")
                    buffer += chunk

            buffer = buffer[end_idx:]
            if step_idx >= start_step:
                yield step_entry
            step_idx += 1


def stream_sparse_manifest_steps(path: str | Path, start_step: int = 0) -> Iterator[dict[str, Any]]:
    path = Path(path)
    if start_step < 0:
        raise ValueError(f"Sparse manifest start_step must be non-negative, got {start_step}")

    header = load_sparse_manifest_header(path)
    if get_sparse_manifest_kind(header) != SPARSE_GROUPING_MANIFEST_KIND:
        raise ValueError("Only grouping sparse manifests contain sparse runtime step entries")
    shards = header.get("shards")
    if not isinstance(shards, list) or len(shards) == 0:
        yield from _stream_sparse_manifest_file_steps(path, start_step=start_step)
        return

    steps_to_skip = int(start_step)
    for shard_entry in shards:
        shard_num_steps = int(shard_entry.get("num_steps", 0))
        if shard_num_steps <= 0:
            raise ValueError("Sparse manifest shard entry must define a positive num_steps")
        if steps_to_skip >= shard_num_steps:
            steps_to_skip -= shard_num_steps
            continue
        shard_path = _resolve_shard_path(path, shard_entry)
        yield from _stream_sparse_manifest_file_steps(shard_path, start_step=steps_to_skip)
        steps_to_skip = 0


def load_sparse_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return _validate_manifest_version(payload)


def validate_sparse_manifest(
    payload: dict[str, Any],
    *,
    split: str,
    vocab_size: int,
    device_batch_size: int,
    max_seq_len: int,
    grad_accum_steps: int,
    ddp_world_size: int,
    num_iterations: int | None = None,
) -> None:
    manifest_kind = get_sparse_manifest_kind(payload)
    if manifest_kind != SPARSE_GROUPING_MANIFEST_KIND:
        raise ValueError("validate_sparse_manifest only accepts grouping sparse manifests")
    expected = {
        "split": split,
        "vocab_size": int(vocab_size),
        "device_batch_size": int(device_batch_size),
        "max_seq_len": int(max_seq_len),
        "grad_accum_steps": int(grad_accum_steps),
        "ddp_world_size": int(ddp_world_size),
    }
    if num_iterations is not None:
        expected["num_steps"] = int(num_iterations)
    for key, value in expected.items():
        found = payload.get(key)
        if isinstance(value, int):
            matches = found is not None and int(found) == value
        else:
            matches = found == value
        if not matches:
            raise ValueError(
                f"Sparse manifest mismatch for {key}: expected {value}, found {found}"
            )
    u_max = int(payload.get("u_max", 0))
    if u_max <= 0:
        raise ValueError("Sparse manifest must define a positive u_max")
    grad_accum_u_max = int(payload.get("grad_accum_u_max", u_max))
    if grad_accum_u_max < u_max:
        raise ValueError(
            f"Sparse manifest grad_accum_u_max must be at least u_max, found grad_accum_u_max={grad_accum_u_max}, u_max={u_max}"
        )
    num_steps = int(payload.get("num_steps", -1))
    if num_steps <= 0:
        raise ValueError("Sparse manifest must define a positive num_steps")
    steps = payload.get("steps")
    shards = payload.get("shards")
    if steps is not None and len(steps) != num_steps:
        raise ValueError("Sparse manifest num_steps does not match the number of stored steps")
    if shards is not None:
        if not isinstance(shards, list) or len(shards) == 0:
            raise ValueError("Sparse manifest shards must be a non-empty list")
        shard_total_steps = 0
        for shard_idx, shard_entry in enumerate(shards):
            shard_num_steps = int(shard_entry.get("num_steps", 0))
            if shard_num_steps <= 0:
                raise ValueError(f"Sparse manifest shard {shard_idx} must define a positive num_steps")
            shard_total_steps += shard_num_steps
            shard_u_max = int(shard_entry.get("u_max", 0))
            if shard_u_max <= 0:
                raise ValueError(f"Sparse manifest shard {shard_idx} must define a positive u_max")
            if shard_u_max > u_max:
                raise ValueError(
                    f"Sparse manifest shard {shard_idx} u_max exceeds manifest u_max: {shard_u_max} > {u_max}"
                )
            shard_grad_accum_u_max = int(shard_entry.get("grad_accum_u_max", shard_u_max))
            if shard_grad_accum_u_max < shard_u_max:
                raise ValueError(
                    f"Sparse manifest shard {shard_idx} grad_accum_u_max must be at least shard u_max, found {shard_grad_accum_u_max} < {shard_u_max}"
                )
            if shard_grad_accum_u_max > grad_accum_u_max:
                raise ValueError(
                    f"Sparse manifest shard {shard_idx} grad_accum_u_max exceeds manifest grad_accum_u_max: {shard_grad_accum_u_max} > {grad_accum_u_max}"
                )
            shard_path = shard_entry.get("path")
            if not isinstance(shard_path, str) or shard_path == "":
                raise ValueError(f"Sparse manifest shard {shard_idx} must define a non-empty path")
        if shard_total_steps != num_steps:
            raise ValueError("Sparse manifest num_steps does not match summed shard num_steps")
        num_shards = payload.get("num_shards")
        if num_shards is not None and int(num_shards) != len(shards):
            raise ValueError("Sparse manifest num_shards does not match the number of shard entries")
    version = int(payload.get("version", 1))
    if version == 2 and steps is not None:
        for step_idx, step_entry in enumerate(steps):
            microsteps = step_entry.get("microsteps")
            if not isinstance(microsteps, list) or len(microsteps) != grad_accum_steps:
                raise ValueError(
                    f"Sparse manifest step {step_idx} must store {grad_accum_steps} microsteps, found {0 if microsteps is None else len(microsteps)}"
                )
            grad_accum_ids = step_entry.get("grad_accum_active_ids")
            if not isinstance(grad_accum_ids, list) or len(grad_accum_ids) == 0:
                raise ValueError(
                    f"Sparse manifest step {step_idx} must define a non-empty grad_accum_active_ids list"
                )
            grad_accum_u_size = int(step_entry.get("grad_accum_u_size", len(grad_accum_ids)))
            if grad_accum_u_size != len(grad_accum_ids):
                raise ValueError(
                    f"Sparse manifest step {step_idx} grad_accum_u_size mismatch: expected {grad_accum_u_size}, found {len(grad_accum_ids)} ids"
                )
            if grad_accum_u_size > grad_accum_u_max:
                raise ValueError(
                    f"Sparse manifest step {step_idx} grad_accum_u_size exceeds grad_accum_u_max: {grad_accum_u_size} > {grad_accum_u_max}"
                )


def validate_sequence_manifest(
    payload: dict[str, Any],
    *,
    split: str,
    vocab_size: int,
    device_batch_size: int,
    max_seq_len: int,
    ddp_world_size: int,
    num_iterations: int | None = None,
) -> None:
    manifest_kind = get_sparse_manifest_kind(payload)
    if manifest_kind != SPARSE_SEQUENCE_BASE_MANIFEST_KIND:
        raise ValueError("validate_sequence_manifest only accepts base sequence manifests")
    expected = {
        "split": split,
        "vocab_size": int(vocab_size),
        "device_batch_size": int(device_batch_size),
        "max_seq_len": int(max_seq_len),
        "ddp_world_size": int(ddp_world_size),
    }
    if num_iterations is not None:
        expected["num_steps"] = int(num_iterations)
    for key, value in expected.items():
        found = payload.get(key)
        if isinstance(value, int):
            matches = found is not None and int(found) == value
        else:
            matches = found == value
        if not matches:
            raise ValueError(
                f"Sequence manifest mismatch for {key}: expected {value}, found {found}"
            )
    num_sequence_units = int(payload.get("num_sequence_units", -1))
    if num_sequence_units <= 0:
        raise ValueError("Sequence manifest must define a positive num_sequence_units")
    num_steps = int(payload.get("num_steps", -1))
    if num_steps <= 0:
        raise ValueError("Sequence manifest must define a positive num_steps")
    shards = payload.get("shards")
    if not isinstance(shards, list) or len(shards) == 0:
        raise ValueError("Sequence manifest shards must be a non-empty list")
    shard_total_units = 0
    for shard_idx, shard_entry in enumerate(shards):
        shard_num_units = int(shard_entry.get("num_sequence_units", 0))
        if shard_num_units <= 0:
            raise ValueError(f"Sequence manifest shard {shard_idx} must define a positive num_sequence_units")
        shard_total_units += shard_num_units
        shard_path = shard_entry.get("path")
        if not isinstance(shard_path, str) or shard_path == "":
            raise ValueError(f"Sequence manifest shard {shard_idx} must define a non-empty path")
    if shard_total_units != num_sequence_units:
        raise ValueError("Sequence manifest num_sequence_units does not match summed shard num_sequence_units")


def resolve_grouping_base_manifest_path(path: str | Path, header: dict[str, Any] | None = None) -> Path:
    path = Path(path)
    if header is None:
        header = load_sparse_manifest_header(path)
    if get_sparse_manifest_kind(header) != SPARSE_GROUPING_MANIFEST_KIND:
        raise ValueError("Only grouping manifests define a base_manifest_path")
    base_manifest_path = header.get("base_manifest_path")
    if not isinstance(base_manifest_path, str) or base_manifest_path == "":
        raise ValueError("Grouping manifest must define a non-empty base_manifest_path")
    return path.parent / base_manifest_path
