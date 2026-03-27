from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq
import torch
from filelock import FileLock

from nanochat.common import print0
from nanochat.dataset import DATA_DIR, list_parquet_files


TOKEN_CACHE_VERSION = 3
TOKEN_CACHE_FORMAT = "parquet-indexed"


def dataset_storage_root() -> Path:
    parquet_paths = list_parquet_files(warn_on_legacy=False)
    dataset_dir = Path(parquet_paths[0]).parent if parquet_paths else Path(DATA_DIR)
    return dataset_dir.resolve().parent


def dataset_storage_dataset_dir() -> Path:
    parquet_paths = list_parquet_files(warn_on_legacy=False)
    dataset_dir = Path(parquet_paths[0]).parent if parquet_paths else Path(DATA_DIR)
    return dataset_dir.resolve()


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def default_token_cache_dir() -> Path:
    dataset_dir = dataset_storage_dataset_dir()
    return dataset_dir.parent / f"{dataset_dir.name}_token_cache_v{TOKEN_CACHE_VERSION}"


def resolve_token_cache_dir(cache_dir: str | Path | None) -> Path:
    storage_root = dataset_storage_root()
    if cache_dir is None:
        return default_token_cache_dir().resolve()
    cache_dir_str = str(cache_dir).strip()
    if cache_dir_str == "":
        return default_token_cache_dir().resolve()
    resolved = Path(cache_dir).expanduser().resolve()
    if not _path_is_within(resolved, storage_root):
        raise ValueError(
            f"Token cache directory must be on dataset-side storage under {storage_root}, got {resolved}"
        )
    return resolved


def _split_parquet_paths(split: str, *, warn_on_legacy: bool = False) -> list[str]:
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files(warn_on_legacy=warn_on_legacy)
    assert len(parquet_paths) != 0, "No dataset parquet files found, did you run dataset.py?"
    return parquet_paths[:-1] if split == "train" else parquet_paths[-1:]


def iter_document_text_batches(
    split: str,
    resume_state_dict: dict | None,
    tokenizer_batch_size: int,
    *,
    ddp_rank: int,
    ddp_world_size: int,
) -> Iterator[tuple[list[str], dict[str, int]]]:
    """Yield document text batches in the same order as the live training loader."""
    warn_on_legacy = ddp_rank == 0 and split == "train"
    parquet_paths = _split_parquet_paths(split, warn_on_legacy=warn_on_legacy)

    resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
    resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
    resume_epoch = resume_state_dict.get("epoch", 1) if resume_state_dict is not None else 1
    resume_text_batch_index = None if resume_state_dict is None else resume_state_dict.get("text_batch_index")
    first_pass = True
    epoch = resume_epoch

    while True:
        pq_idx = resume_pq_idx if first_pass else 0
        while pq_idx < len(parquet_paths):
            filepath = parquet_paths[pq_idx]
            pf = pq.ParquetFile(filepath)
            resume_batch_idx_for_rg = None
            if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                if resume_text_batch_index is not None:
                    rg_idx = resume_rg_idx
                    resume_batch_idx_for_rg = int(resume_text_batch_index) + 1
                    resume_text_batch_index = None
                else:
                    base_idx = resume_rg_idx // ddp_world_size
                    base_idx += 1
                    rg_idx = base_idx * ddp_world_size + ddp_rank
                    if rg_idx >= pf.num_row_groups:
                        pq_idx += 1
                        continue
                    resume_rg_idx = None
            else:
                rg_idx = ddp_rank
            while rg_idx < pf.num_row_groups:
                rg = pf.read_row_group(rg_idx)
                batch = rg.column("text").to_pylist()
                start_batch_idx = 0 if resume_batch_idx_for_rg is None else resume_batch_idx_for_rg
                for batch_idx, i in enumerate(range(0, len(batch), tokenizer_batch_size)):
                    if batch_idx < start_batch_idx:
                        continue
                    state = {
                        "pq_idx": pq_idx,
                        "rg_idx": rg_idx,
                        "epoch": epoch,
                        "text_batch_index": batch_idx,
                    }
                    yield batch[i:i + tokenizer_batch_size], state
                resume_rg_idx = None
                resume_batch_idx_for_rg = None
                rg_idx += ddp_world_size
            pq_idx += 1
        first_pass = False
        epoch += 1


def _cache_split_dir(cache_dir: str | Path, split: str) -> Path:
    return resolve_token_cache_dir(cache_dir) / split


def _cache_metadata_path(cache_dir: str | Path, split: str) -> Path:
    return _cache_split_dir(cache_dir, split) / "metadata.json"


def _cache_lock_path(cache_dir: str | Path, split: str) -> Path:
    return _cache_split_dir(cache_dir, split) / ".build.lock"


def _cache_parquet_data_path(cache_dir: str | Path, split: str, pq_idx: int) -> Path:
    return _cache_split_dir(cache_dir, split) / f"parquet_{int(pq_idx):05d}.pt"


def _cache_parquet_dir(cache_dir: str | Path, split: str, pq_idx: int) -> Path:
    return _cache_split_dir(cache_dir, split) / f"parquet_{int(pq_idx):05d}"


def _cache_row_group_data_path(cache_dir: str | Path, split: str, pq_idx: int, rg_idx: int) -> Path:
    return _cache_parquet_dir(cache_dir, split, pq_idx) / f"row_group_{int(rg_idx):05d}.pt"


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp_path, path)


def _atomic_torch_save(path: Path, payload: dict) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _serialize_token_lists(token_lists: list[torch.Tensor]) -> dict:
    lengths = torch.tensor([int(tokens.numel()) for tokens in token_lists], dtype=torch.int32)
    flat_tokens = torch.cat(token_lists) if token_lists else torch.empty(0, dtype=torch.long)
    return {
        "lengths": lengths,
        "tokens": flat_tokens.to(dtype=torch.long),
    }


def _deserialize_token_lists(payload: dict) -> list[torch.Tensor]:
    lengths = payload["lengths"].to(dtype=torch.int64)
    flat_tokens = payload["tokens"].to(dtype=torch.long)
    token_lists: list[torch.Tensor] = []
    offset = 0
    for length in lengths.tolist():
        next_offset = offset + int(length)
        token_lists.append(flat_tokens[offset:next_offset].clone())
        offset = next_offset
    return token_lists


def _state_key(state: dict[str, int]) -> tuple[int, int, int, int]:
    return (
        int(state.get("epoch", 1)),
        int(state.get("pq_idx", -1)),
        int(state.get("rg_idx", -1)),
        int(state.get("text_batch_index", -1)),
    )


def _normalize_resume_state(resume_state_dict: dict | None) -> dict[str, int] | None:
    if resume_state_dict is None:
        return None
    return {
        key: int(value)
        for key, value in resume_state_dict.items()
        if key in {"pq_idx", "rg_idx", "epoch", "text_batch_index"}
    }


def _resume_state_key(resume_state: dict[str, int]) -> tuple[int, int, int, int]:
    resume_batch = (1 << 60) if "text_batch_index" not in resume_state else int(resume_state["text_batch_index"])
    return (
        int(resume_state.get("epoch", 1)),
        int(resume_state.get("pq_idx", -1)),
        int(resume_state.get("rg_idx", -1)),
        resume_batch,
    )


def _expected_cache_metadata(
    split: str,
    *,
    tokenizer,
    tokenizer_batch_size: int,
    bos_token_id: int,
    parquet_paths: list[str],
) -> dict:
    return {
        "version": TOKEN_CACHE_VERSION,
        "format": TOKEN_CACHE_FORMAT,
        "split": split,
        "tokenizer_batch_size": int(tokenizer_batch_size),
        "vocab_size": int(tokenizer.get_vocab_size()),
        "bos_token_id": int(bos_token_id),
        "parquet_files": [Path(path).name for path in parquet_paths],
    }


def _cache_metadata_matches_expected(metadata: dict | None, expected: dict) -> bool:
    if metadata is None:
        return False
    for key, value in expected.items():
        if metadata.get(key) != value:
            return False
    return True


def _finalize_cache_metadata(metadata: dict, *, cache_dir: str | Path, split: str) -> dict:
    split_dir = _cache_split_dir(cache_dir, split)
    complete_files: list[dict] = []
    for entry in metadata.get("files", []):
        path = split_dir / str(entry.get("path", ""))
        if entry.get("complete") and path.exists():
            complete_files.append(entry)
    complete_files.sort(key=lambda entry: int(entry["pq_idx"]))
    metadata["files"] = complete_files
    metadata["num_parquets"] = len(complete_files)
    metadata["num_row_groups"] = sum(int(entry.get("num_row_groups", 0)) for entry in complete_files)
    metadata["num_batches"] = sum(int(entry.get("num_batches", 0)) for entry in complete_files)
    metadata["num_docs"] = sum(int(entry.get("num_docs", 0)) for entry in complete_files)
    metadata["complete"] = len(complete_files) == len(metadata.get("parquet_files", []))
    return metadata


def load_token_cache_metadata(cache_dir: str | Path, split: str) -> dict | None:
    metadata_path = _cache_metadata_path(cache_dir, split)
    if not metadata_path.exists():
        return None
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def token_cache_is_valid(
    cache_dir: str | Path,
    split: str,
    *,
    tokenizer,
    tokenizer_batch_size: int,
    bos_token_id: int,
    ddp_world_size: int,
) -> bool:
    del ddp_world_size
    parquet_paths = _split_parquet_paths(split, warn_on_legacy=False)
    metadata = load_token_cache_metadata(cache_dir, split)
    expected = _expected_cache_metadata(
        split,
        tokenizer=tokenizer,
        tokenizer_batch_size=tokenizer_batch_size,
        bos_token_id=bos_token_id,
        parquet_paths=parquet_paths,
    )
    if not _cache_metadata_matches_expected(metadata, expected):
        return False
    assert metadata is not None
    finalized = _finalize_cache_metadata(dict(metadata), cache_dir=cache_dir, split=split)
    return bool(finalized.get("complete"))


def _write_parquet_cache_payload(cache_dir: str | Path, split: str, pq_idx: int, payload: dict) -> None:
    data_path = _cache_parquet_data_path(cache_dir, split, pq_idx)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(data_path, payload)


def _write_row_group_cache_payload(cache_dir: str | Path, split: str, pq_idx: int, rg_idx: int, payload: dict) -> str:
    data_path = _cache_row_group_data_path(cache_dir, split, pq_idx, rg_idx)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(data_path, payload)
    return str(data_path.relative_to(_cache_split_dir(cache_dir, split)))


def _build_parquet_cache_entry_with_tokenizer(
    tokenizer,
    *,
    cache_dir: str | Path,
    split: str,
    pq_idx: int,
    filepath: str,
    bos_token_id: int,
    tokenizer_batch_size: int,
    tokenizer_threads: int,
) -> dict:
    pf = pq.ParquetFile(filepath)
    row_groups: list[dict] = []
    num_batches = 0
    num_docs = 0
    for rg_idx in range(pf.num_row_groups):
        rg = pf.read_row_group(rg_idx)
        texts = rg.column("text").to_pylist()
        batch_payloads: list[dict] = []
        for batch_idx, start in enumerate(range(0, len(texts), tokenizer_batch_size)):
            del batch_idx
            text_batch = texts[start:start + tokenizer_batch_size]
            token_lists = [
                torch.tensor(tokens, dtype=torch.long)
                for tokens in tokenizer.encode(text_batch, prepend=bos_token_id, num_threads=tokenizer_threads)
            ]
            batch_payloads.append(_serialize_token_lists(token_lists))
            num_batches += 1
            num_docs += len(token_lists)
        row_group_path = _write_row_group_cache_payload(
            cache_dir,
            split,
            pq_idx,
            rg_idx,
            {
                "version": TOKEN_CACHE_VERSION,
                "format": TOKEN_CACHE_FORMAT,
                "split": split,
                "pq_idx": int(pq_idx),
                "rg_idx": int(rg_idx),
                "num_batches": len(batch_payloads),
                "num_docs": len(texts),
                "batches": batch_payloads,
            },
        )
        row_groups.append(
            {
                "rg_idx": int(rg_idx),
                "num_batches": len(batch_payloads),
                "num_docs": len(texts),
                "path": row_group_path,
            }
        )
    payload = {
        "version": TOKEN_CACHE_VERSION,
        "format": TOKEN_CACHE_FORMAT,
        "split": split,
        "pq_idx": int(pq_idx),
        "source_path": Path(filepath).name,
        "num_row_groups": int(pf.num_row_groups),
        "row_groups": row_groups,
    }
    _write_parquet_cache_payload(cache_dir, split, pq_idx, payload)
    return {
        "pq_idx": int(pq_idx),
        "path": _cache_parquet_data_path(cache_dir, split, pq_idx).name,
        "source_path": Path(filepath).name,
        "num_row_groups": int(pf.num_row_groups),
        "num_batches": int(num_batches),
        "num_docs": int(num_docs),
        "complete": True,
    }


def _build_parquet_cache_entry_worker(args: tuple[str, str, int, str, int, int, int]) -> dict:
    cache_dir, split, pq_idx, filepath, bos_token_id, tokenizer_batch_size, tokenizer_threads = args
    from nanochat.tokenizer import get_tokenizer

    tokenizer = get_tokenizer()
    return _build_parquet_cache_entry_with_tokenizer(
        tokenizer,
        cache_dir=cache_dir,
        split=split,
        pq_idx=pq_idx,
        filepath=filepath,
        bos_token_id=bos_token_id,
        tokenizer_batch_size=tokenizer_batch_size,
        tokenizer_threads=tokenizer_threads,
    )


def _write_split_metadata(cache_dir: str | Path, split: str, metadata: dict) -> dict:
    finalized = _finalize_cache_metadata(metadata, cache_dir=cache_dir, split=split)
    metadata_path = _cache_metadata_path(cache_dir, split)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(metadata_path, finalized)
    return finalized


def _pending_parquet_work(cache_dir: str | Path, split: str, parquet_paths: list[str], metadata: dict) -> list[tuple[int, str]]:
    split_dir = _cache_split_dir(cache_dir, split)
    completed = {
        int(entry["pq_idx"])
        for entry in metadata.get("files", [])
        if entry.get("complete") and (split_dir / str(entry.get("path", ""))).exists()
    }
    return [
        (pq_idx, filepath)
        for pq_idx, filepath in enumerate(parquet_paths)
        if pq_idx not in completed
    ]


def _build_serial_cache_entries(
    cache_dir: str | Path,
    split: str,
    *,
    tokenizer,
    tokenizer_threads: int,
    tokenizer_batch_size: int,
    bos_token_id: int,
    pending_pq_indices: set[int],
) -> list[dict]:
    entries: list[dict] = []
    current_pq_idx: int | None = None
    current_filepath: str | None = None
    current_row_groups: dict[int, list[dict]] = {}
    current_num_batches = 0
    current_num_docs = 0
    parquet_paths = _split_parquet_paths(split, warn_on_legacy=False)

    def flush_current() -> None:
        nonlocal current_pq_idx, current_filepath, current_row_groups, current_num_batches, current_num_docs
        if current_pq_idx is None or current_filepath is None:
            return
        ordered_row_groups = []
        for rg_idx in sorted(current_row_groups):
            batch_payloads = current_row_groups[rg_idx]
            row_group_path = _write_row_group_cache_payload(
                cache_dir,
                split,
                current_pq_idx,
                rg_idx,
                {
                    "version": TOKEN_CACHE_VERSION,
                    "format": TOKEN_CACHE_FORMAT,
                    "split": split,
                    "pq_idx": int(current_pq_idx),
                    "rg_idx": int(rg_idx),
                    "num_batches": len(batch_payloads),
                    "num_docs": sum(int(payload["lengths"].numel()) for payload in batch_payloads),
                    "batches": batch_payloads,
                },
            )
            ordered_row_groups.append(
                {
                    "rg_idx": int(rg_idx),
                    "num_batches": len(batch_payloads),
                    "num_docs": sum(int(payload["lengths"].numel()) for payload in batch_payloads),
                    "path": row_group_path,
                }
            )
        payload = {
            "version": TOKEN_CACHE_VERSION,
            "format": TOKEN_CACHE_FORMAT,
            "split": split,
            "pq_idx": int(current_pq_idx),
            "source_path": Path(current_filepath).name,
            "num_row_groups": len(ordered_row_groups),
            "row_groups": ordered_row_groups,
        }
        _write_parquet_cache_payload(cache_dir, split, current_pq_idx, payload)
        entries.append(
            {
                "pq_idx": int(current_pq_idx),
                "path": _cache_parquet_data_path(cache_dir, split, current_pq_idx).name,
                "source_path": Path(current_filepath).name,
                "num_row_groups": len(ordered_row_groups),
                "num_batches": int(current_num_batches),
                "num_docs": int(current_num_docs),
                "complete": True,
            }
        )
        current_pq_idx = None
        current_filepath = None
        current_row_groups = {}
        current_num_batches = 0
        current_num_docs = 0

    for text_batch, state in iter_document_text_batches(
        split,
        resume_state_dict=None,
        tokenizer_batch_size=tokenizer_batch_size,
        ddp_rank=0,
        ddp_world_size=1,
    ):
        if int(state.get("epoch", 1)) > 1:
            break
        pq_idx = int(state["pq_idx"])
        if pq_idx not in pending_pq_indices:
            continue
        if current_pq_idx is None:
            current_pq_idx = pq_idx
            current_filepath = parquet_paths[pq_idx]
        elif pq_idx != current_pq_idx:
            flush_current()
            current_pq_idx = pq_idx
            current_filepath = parquet_paths[pq_idx]
        rg_idx = int(state["rg_idx"])
        token_lists = [
            torch.tensor(tokens, dtype=torch.long)
            for tokens in tokenizer.encode(text_batch, prepend=bos_token_id, num_threads=tokenizer_threads)
        ]
        current_row_groups.setdefault(rg_idx, []).append(_serialize_token_lists(token_lists))
        current_num_batches += 1
        current_num_docs += len(token_lists)
    flush_current()
    return entries


def prepare_token_cache_writer(
    cache_dir: str | Path,
    split: str,
    *,
    tokenizer,
    tokenizer_batch_size: int,
    bos_token_id: int,
    ddp_world_size: int,
    shard_batch_count: int = 256,
):
    del tokenizer, tokenizer_batch_size, bos_token_id, ddp_world_size, shard_batch_count
    resolved_dir = resolve_token_cache_dir(cache_dir)
    split_dir = _cache_split_dir(resolved_dir, split)
    split_dir.mkdir(parents=True, exist_ok=True)
    existing_metadata = load_token_cache_metadata(resolved_dir, split)
    existing_artifacts = [path for path in split_dir.iterdir() if path.name != ".build.lock"]
    if existing_metadata is not None or existing_artifacts:
        raise ValueError(
            "Refusing to overwrite existing token cache contents because they are not compatible with the current run. "
            f"Use a different --token-cache-dir or remove the cache manually if replacement is intentional: {split_dir}"
        )
    raise ValueError("Incremental token-cache writing is no longer supported; call ensure_token_cache() instead")


def ensure_token_cache(
    cache_dir: str | Path,
    split: str,
    *,
    tokenizer,
    tokenizer_threads: int,
    tokenizer_batch_size: int,
    bos_token_id: int,
    ddp_rank: int,
    ddp_world_size: int,
    shard_batch_count: int = 256,
    num_workers: int = 1,
) -> dict:
    del ddp_rank, ddp_world_size, shard_batch_count
    resolved_dir = resolve_token_cache_dir(cache_dir)
    split_dir = _cache_split_dir(resolved_dir, split)
    split_dir.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(_cache_lock_path(resolved_dir, split)))
    parquet_paths = _split_parquet_paths(split, warn_on_legacy=False)
    expected = _expected_cache_metadata(
        split,
        tokenizer=tokenizer,
        tokenizer_batch_size=tokenizer_batch_size,
        bos_token_id=bos_token_id,
        parquet_paths=parquet_paths,
    )

    with lock:
        metadata = load_token_cache_metadata(resolved_dir, split)
        if metadata is not None and not _cache_metadata_matches_expected(metadata, expected):
            raise ValueError(
                "Refusing to overwrite existing token cache contents because they are not compatible with the current run. "
                f"Use a different --token-cache-dir or remove the cache manually if replacement is intentional: {split_dir}"
            )
        if metadata is None:
            metadata = {
                **expected,
                "files": [],
                "num_parquets": 0,
                "num_row_groups": 0,
                "num_batches": 0,
                "num_docs": 0,
                "complete": False,
            }
            metadata = _write_split_metadata(resolved_dir, split, metadata)
        else:
            metadata = _write_split_metadata(resolved_dir, split, metadata)
        if metadata.get("complete"):
            return metadata
        pending = _pending_parquet_work(resolved_dir, split, parquet_paths, metadata)

    if not pending:
        with lock:
            metadata = load_token_cache_metadata(resolved_dir, split) or metadata
            return _write_split_metadata(resolved_dir, split, metadata)

    worker_count = max(1, int(num_workers))
    print0(
        f"Building token cache for split='{split}' at {split_dir} "
        f"with {worker_count} worker{'s' if worker_count != 1 else ''} across {len(pending)} parquet file(s)"
    )
    new_entries: list[dict] = []
    if worker_count == 1:
        new_entries = _build_serial_cache_entries(
            resolved_dir,
            split,
            tokenizer=tokenizer,
            tokenizer_threads=tokenizer_threads,
            tokenizer_batch_size=tokenizer_batch_size,
            bos_token_id=bos_token_id,
            pending_pq_indices={pq_idx for pq_idx, _ in pending},
        )
    else:
        work_items = [
            (
                str(resolved_dir),
                split,
                int(pq_idx),
                filepath,
                int(bos_token_id),
                int(tokenizer_batch_size),
                int(tokenizer_threads),
            )
            for pq_idx, filepath in pending
        ]
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(_build_parquet_cache_entry_worker, item) for item in work_items]
            for future in as_completed(futures):
                new_entries.append(future.result())

    with lock:
        metadata = load_token_cache_metadata(resolved_dir, split) or {
            **expected,
            "files": [],
        }
        if not _cache_metadata_matches_expected(metadata, expected):
            raise ValueError(
                "Token cache metadata changed during build; refusing to merge results into an incompatible cache."
            )
        files_by_pq = {int(entry["pq_idx"]): entry for entry in metadata.get("files", [])}
        for entry in new_entries:
            files_by_pq[int(entry["pq_idx"])] = entry
        metadata["files"] = [files_by_pq[pq_idx] for pq_idx in sorted(files_by_pq)]
        return _write_split_metadata(resolved_dir, split, metadata)


def _iter_cached_epoch_batches(
    cache_dir: str | Path,
    split: str,
    *,
    ddp_rank: int,
    ddp_world_size: int,
) -> Iterator[tuple[list[torch.Tensor], dict[str, int]]]:
    metadata = load_token_cache_metadata(cache_dir, split)
    if metadata is None:
        return
    split_dir = _cache_split_dir(cache_dir, split)
    for file_entry in sorted(metadata.get("files", []), key=lambda entry: int(entry["pq_idx"])):
        pq_idx = int(file_entry["pq_idx"])
        data_path = split_dir / str(file_entry["path"])
        payload = torch.load(data_path, map_location="cpu", weights_only=False)
        for row_group in payload.get("row_groups", []):
            rg_idx = int(row_group["rg_idx"])
            if rg_idx % int(ddp_world_size) != int(ddp_rank):
                continue
            row_group_path = split_dir / str(row_group["path"])
            row_group_payload = torch.load(row_group_path, map_location="cpu", weights_only=False)
            for batch_idx, batch_payload in enumerate(row_group_payload.get("batches", [])):
                state = {
                    "pq_idx": pq_idx,
                    "rg_idx": rg_idx,
                    "epoch": 1,
                    "text_batch_index": int(batch_idx),
                }
                yield _deserialize_token_lists(batch_payload), state


def load_cached_token_batches(cache_dir: str | Path, split: str) -> list[tuple[list[torch.Tensor], dict[str, int]]]:
    return list(iter_cached_token_batches(cache_dir, split, ddp_rank=0, ddp_world_size=1, repeat=False))


def iter_cached_token_batches(
    cache_dir: str | Path,
    split: str,
    *,
    resume_state_dict: dict | None = None,
    ddp_rank: int = 0,
    ddp_world_size: int = 1,
    repeat: bool = False,
) -> Iterator[tuple[list[torch.Tensor], dict[str, int]]]:
    resume_state = _normalize_resume_state(resume_state_dict)
    resume_key = None if resume_state is None else _resume_state_key(resume_state)
    first_pass = True
    current_epoch = 1 if resume_state is None else int(resume_state.get("epoch", 1))
    while True:
        started = resume_state is None or not first_pass
        for token_lists, state in _iter_cached_epoch_batches(
            cache_dir,
            split,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
        ):
            state["epoch"] = current_epoch
            if not started and resume_state is not None:
                assert resume_key is not None
                if _state_key(state) <= resume_key:
                    continue
                started = True
            yield token_lists, state
        if not repeat:
            break
        first_pass = False
        current_epoch += 1


def iter_token_batches_from_cache(
    cache_dir: str | Path,
    split: str,
    *,
    resume_state_dict: dict | None = None,
    ddp_rank: int = 0,
    ddp_world_size: int = 1,
) -> Iterator[tuple[list[torch.Tensor], dict[str, int]]]:
    iterator = iter_cached_token_batches(
        cache_dir,
        split,
        resume_state_dict=resume_state_dict,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
        repeat=False,
    )
    try:
        first_item = next(iterator)
    except StopIteration:
        raise ValueError(f"Token cache for split='{split}' contains no batches")
    yield first_item
    yield from iterator
