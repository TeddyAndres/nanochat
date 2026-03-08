from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


SPARSE_MANIFEST_VERSION = 1


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
    u_max = max((int(step["u_size"]) for step in steps), default=0)
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
        "tokenizer_batch_size": int(tokenizer_batch_size),
        "tokenizer_threads": int(tokenizer_threads),
        "buffer_size": int(buffer_size),
        "u_max": int(u_max),
        "steps": steps,
    }


def save_sparse_manifest(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)


def load_sparse_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    version = int(payload.get("version", -1))
    if version != SPARSE_MANIFEST_VERSION:
        raise ValueError(
            f"Unsupported sparse manifest version {version}; expected {SPARSE_MANIFEST_VERSION}"
        )
    return payload


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
    steps = payload.get("steps", [])
    if len(steps) != int(payload.get("num_steps", -1)):
        raise ValueError("Sparse manifest num_steps does not match the number of stored steps")
