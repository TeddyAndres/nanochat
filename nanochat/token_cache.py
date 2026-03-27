from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq
import torch
from filelock import FileLock

from nanochat.common import print0
from nanochat.dataset import DATA_DIR, list_parquet_files


TOKEN_CACHE_VERSION = 1


def default_token_cache_dir() -> Path:
    parquet_paths = list_parquet_files(warn_on_legacy=False)
    dataset_dir = Path(parquet_paths[0]).parent if parquet_paths else Path(DATA_DIR)
    return dataset_dir.parent / f"{dataset_dir.name}_token_cache"


def resolve_token_cache_dir(cache_dir: str | Path | None) -> Path:
    if cache_dir is None:
        return default_token_cache_dir()
    cache_dir_str = str(cache_dir).strip()
    if cache_dir_str == "":
        return default_token_cache_dir()
    return Path(cache_dir)


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
    parquet_paths = list_parquet_files(warn_on_legacy=warn_on_legacy)
    assert len(parquet_paths) != 0, "No dataset parquet files found, did you run dataset.py?"
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]

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


def _serialize_token_batch(token_lists: list[torch.Tensor], state: dict[str, int]) -> dict:
    lengths = torch.tensor([int(tokens.numel()) for tokens in token_lists], dtype=torch.int32)
    flat_tokens = torch.cat(token_lists) if token_lists else torch.empty(0, dtype=torch.long)
    return {
        "state": {key: int(value) for key, value in state.items()},
        "lengths": lengths,
        "tokens": flat_tokens.to(dtype=torch.long),
    }


def _deserialize_token_batch(payload: dict) -> tuple[list[torch.Tensor], dict[str, int]]:
    lengths = payload["lengths"].to(dtype=torch.int64)
    flat_tokens = payload["tokens"].to(dtype=torch.long)
    token_lists: list[torch.Tensor] = []
    offset = 0
    for length in lengths.tolist():
        next_offset = offset + int(length)
        token_lists.append(flat_tokens[offset:next_offset].clone())
        offset = next_offset
    state = {key: int(value) for key, value in payload["state"].items()}
    return token_lists, state


def load_cached_token_batches(cache_dir: str | Path, split: str) -> list[tuple[list[torch.Tensor], dict[str, int]]]:
    metadata = load_token_cache_metadata(cache_dir, split)
    if metadata is None:
        return []
    split_dir = _cache_split_dir(cache_dir, split)
    cached_batches: list[tuple[list[torch.Tensor], dict[str, int]]] = []
    for shard_entry in metadata.get("shards", []):
        shard_path = split_dir / str(shard_entry["path"])
        payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        for batch_payload in payload["batches"]:
            cached_batches.append(_deserialize_token_batch(batch_payload))
    return cached_batches


def _iter_cached_token_batches_from_shards(
    cache_dir: str | Path,
    split: str,
) -> Iterator[tuple[list[torch.Tensor], dict[str, int]]]:
    metadata = load_token_cache_metadata(cache_dir, split)
    if metadata is None:
        return
    split_dir = _cache_split_dir(cache_dir, split)
    for shard_entry in metadata.get("shards", []):
        shard_path = split_dir / str(shard_entry["path"])
        payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        for batch_payload in payload["batches"]:
            yield _deserialize_token_batch(batch_payload)


def iter_cached_token_batches(
    cache_dir: str | Path,
    split: str,
    *,
    resume_state_dict: dict | None = None,
) -> Iterator[tuple[list[torch.Tensor], dict[str, int]]]:
    resume_state = None if resume_state_dict is None else {
        key: int(value)
        for key, value in resume_state_dict.items()
        if key in {"pq_idx", "rg_idx", "epoch", "text_batch_index"}
    }
    started = resume_state is None
    if resume_state is not None:
        resume_epoch = int(resume_state.get("epoch", 1))
        resume_pq = int(resume_state.get("pq_idx", -1))
        resume_rg = int(resume_state.get("rg_idx", -1))
        resume_batch = int(resume_state.get("text_batch_index", -1))

    for token_lists, state in _iter_cached_token_batches_from_shards(cache_dir, split):
        if not started:
            state_epoch = int(state["epoch"])
            state_pq = int(state["pq_idx"])
            state_rg = int(state["rg_idx"])
            state_batch = int(state.get("text_batch_index", -1))
            if (state_epoch, state_pq, state_rg, state_batch) <= (resume_epoch, resume_pq, resume_rg, resume_batch):
                continue
            started = True
        yield token_lists, state


class TokenCacheWriter:
    def __init__(self, cache_dir: str | Path, split: str, *, metadata: dict, shard_batch_count: int):
        self.cache_dir = resolve_token_cache_dir(cache_dir)
        self.split = split
        self.split_dir = _cache_split_dir(self.cache_dir, split)
        self.split_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = metadata
        self.shard_batch_count = int(shard_batch_count)
        self.shard_payloads: list[dict] = []
        self.lock = FileLock(str(_cache_lock_path(self.cache_dir, split)))
        self.closed = False

    def _write_metadata(self) -> None:
        metadata_path = _cache_metadata_path(self.cache_dir, self.split)
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(self.metadata, f)

    def _flush_locked(self) -> None:
        if not self.shard_payloads:
            return
        shard_index = int(self.metadata.get("next_shard_index", 0))
        shard_name = f"token_cache.{shard_index:05d}.pt"
        shard_path = self.split_dir / shard_name
        torch.save({"batches": self.shard_payloads}, shard_path)
        self.metadata.setdefault("shards", []).append(
            {
                "path": shard_name,
                "num_batches": len(self.shard_payloads),
            }
        )
        self.metadata["next_shard_index"] = shard_index + 1
        self.shard_payloads = []
        self._write_metadata()

    def append(self, token_lists: list[torch.Tensor], state: dict[str, int]) -> None:
        if self.closed:
            raise ValueError("TokenCacheWriter is already closed")
        self.shard_payloads.append(_serialize_token_batch(token_lists, state))
        self.metadata["num_batches"] = int(self.metadata.get("num_batches", 0)) + 1
        self.metadata["num_docs"] = int(self.metadata.get("num_docs", 0)) + len(token_lists)
        self.metadata["last_state"] = {key: int(value) for key, value in state.items()}
        if len(self.shard_payloads) >= self.shard_batch_count:
            with self.lock:
                self._flush_locked()

    def close(self) -> None:
        if self.closed:
            return
        with self.lock:
            self._flush_locked()
            self._write_metadata()
        self.closed = True


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
    metadata = load_token_cache_metadata(cache_dir, split)
    if metadata is None:
        return False
    expected = {
        "version": TOKEN_CACHE_VERSION,
        "split": split,
        "tokenizer_batch_size": int(tokenizer_batch_size),
        "vocab_size": int(tokenizer.get_vocab_size()),
        "bos_token_id": int(bos_token_id),
        "ddp_world_size": int(ddp_world_size),
    }
    for key, value in expected.items():
        if isinstance(value, int):
            if int(metadata.get(key, -1)) != value:
                return False
        elif metadata.get(key) != value:
            return False
    shard_entries = metadata.get("shards")
    if not isinstance(shard_entries, list) or len(shard_entries) == 0:
        return False
    split_dir = _cache_split_dir(cache_dir, split)
    return all((split_dir / str(entry.get("path", ""))).exists() for entry in shard_entries)


def prepare_token_cache_writer(
    cache_dir: str | Path,
    split: str,
    *,
    tokenizer,
    tokenizer_batch_size: int,
    bos_token_id: int,
    ddp_world_size: int,
    shard_batch_count: int = 256,
) -> TokenCacheWriter | None:
    resolved_dir = resolve_token_cache_dir(cache_dir)
    existing_metadata = load_token_cache_metadata(resolved_dir, split)
    if token_cache_is_valid(
        resolved_dir,
        split,
        tokenizer=tokenizer,
        tokenizer_batch_size=tokenizer_batch_size,
        bos_token_id=bos_token_id,
        ddp_world_size=ddp_world_size,
    ):
        existing_metadata = existing_metadata or {}
        existing_metadata.setdefault("next_shard_index", len(existing_metadata.get("shards", [])))
        existing_metadata.setdefault("num_batches", 0)
        existing_metadata.setdefault("num_docs", 0)
        print0(f"Extending token cache for split='{split}' at {_cache_split_dir(resolved_dir, split)}")
        return TokenCacheWriter(
            resolved_dir,
            split,
            metadata=existing_metadata,
            shard_batch_count=shard_batch_count,
        )

    split_dir = _cache_split_dir(resolved_dir, split)
    split_dir.mkdir(parents=True, exist_ok=True)
    existing_shards = sorted(split_dir.glob("*.pt"))
    if existing_metadata is not None or existing_shards:
        raise ValueError(
            "Refusing to overwrite existing token cache contents because they are not compatible with the current run. "
            f"Use a different --token-cache-dir or remove the cache manually if replacement is intentional: {split_dir}"
        )
    metadata = {
        "version": TOKEN_CACHE_VERSION,
        "split": split,
        "tokenizer_batch_size": int(tokenizer_batch_size),
        "vocab_size": int(tokenizer.get_vocab_size()),
        "bos_token_id": int(bos_token_id),
        "ddp_world_size": int(ddp_world_size),
        "num_batches": 0,
        "num_docs": 0,
        "next_shard_index": 0,
        "shards": [],
    }
    print0(f"Caching tokenized batches for split='{split}' at {split_dir}")
    return TokenCacheWriter(
        resolved_dir,
        split,
        metadata=metadata,
        shard_batch_count=shard_batch_count,
    )


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
) -> dict:
    """Compatibility helper that eagerly fills a cache through the first dataset epoch."""
    resolved_dir = resolve_token_cache_dir(cache_dir)
    if token_cache_is_valid(
        resolved_dir,
        split,
        tokenizer=tokenizer,
        tokenizer_batch_size=tokenizer_batch_size,
        bos_token_id=bos_token_id,
        ddp_world_size=ddp_world_size,
    ):
        return load_token_cache_metadata(resolved_dir, split) or {}

    writer = prepare_token_cache_writer(
        resolved_dir,
        split,
        tokenizer=tokenizer,
        tokenizer_batch_size=tokenizer_batch_size,
        bos_token_id=bos_token_id,
        ddp_world_size=ddp_world_size,
        shard_batch_count=shard_batch_count,
    )
    assert writer is not None
    try:
        for text_batch, state in iter_document_text_batches(
            split,
            resume_state_dict=None,
            tokenizer_batch_size=tokenizer_batch_size,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
        ):
            if state["epoch"] > 1:
                break
            token_lists = [torch.tensor(tokens, dtype=torch.long) for tokens in tokenizer.encode(text_batch, prepend=bos_token_id, num_threads=tokenizer_threads)]
            writer.append(token_lists, state)
    finally:
        writer.close()
    return load_token_cache_metadata(resolved_dir, split) or {}


def iter_token_batches_from_cache(
    cache_dir: str | Path,
    split: str,
    *,
    resume_state_dict: dict | None = None,
) -> Iterator[tuple[list[torch.Tensor], dict[str, int]]]:
    iterator = iter_cached_token_batches(cache_dir, split, resume_state_dict=resume_state_dict)
    try:
        first_item = next(iterator)
    except StopIteration:
        raise ValueError(f"Token cache for split='{split}' contains no batches")
    yield first_item
    yield from iterator
