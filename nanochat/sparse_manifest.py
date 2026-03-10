from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import torch


SPARSE_MANIFEST_VERSION = 2
SUPPORTED_SPARSE_MANIFEST_VERSIONS = {1, 2}


def _validate_manifest_version(payload: dict[str, Any]) -> dict[str, Any]:
    version = int(payload.get("version", -1))
    if version not in SUPPORTED_SPARSE_MANIFEST_VERSIONS:
        raise ValueError(
            f"Unsupported sparse manifest version {version}; expected one of {sorted(SUPPORTED_SPARSE_MANIFEST_VERSIONS)}"
        )
    return payload


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
    tokenizer_batch_size: int,
    tokenizer_threads: int,
    buffer_size: int,
    steps: list[dict[str, Any]],
) -> dict[str, Any]:
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
        "tokenizer_batch_size": int(tokenizer_batch_size),
        "tokenizer_threads": int(tokenizer_threads),
        "buffer_size": int(buffer_size),
        "u_max": int(u_max),
        "grad_accum_u_max": int(grad_accum_u_max),
        "steps": steps,
    }


def save_sparse_manifest(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)


def load_sparse_manifest_header(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    marker = '"steps"'
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
            payload = json.loads(header_text + "}")
            return _validate_manifest_version(payload)


def stream_sparse_manifest_steps(path: str | Path, start_step: int = 0) -> Iterator[dict[str, Any]]:
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
    if steps is not None and len(steps) != num_steps:
        raise ValueError("Sparse manifest num_steps does not match the number of stored steps")
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
