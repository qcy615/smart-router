from __future__ import annotations

import base64
import hashlib
import json
from typing import Any, Iterable, Sequence

LocalBlockHash = bytes
ExternalBlockHash = bytes | int | str


def canonicalize(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"__bytes__": base64.b64encode(value).decode("ascii")}
    if isinstance(value, tuple):
        return [canonicalize(item) for item in value]
    if isinstance(value, list):
        return [canonicalize(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): canonicalize(value[key])
            for key in sorted(value.keys(), key=lambda item: str(item))
        }
    return value


def hash_block_tokens(
    parent_hash: LocalBlockHash | None,
    token_ids: Sequence[int],
    extra_keys: Any = None,
) -> LocalBlockHash:
    payload = {
        "parent": canonicalize(parent_hash),
        "tokens": list(token_ids),
        "extra": canonicalize(extra_keys),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).digest()


def block_hashes_for_tokens(
    token_ids: Sequence[int],
    block_size: int,
    extra_keys: Sequence[Any] | None = None,
    parent_hash: LocalBlockHash | None = None,
) -> list[LocalBlockHash]:
    block_count = len(token_ids) // block_size
    hashes: list[LocalBlockHash] = []
    previous = parent_hash

    for block_idx in range(block_count):
        start = block_idx * block_size
        end = start + block_size
        block_extra_keys = extra_keys[block_idx] if extra_keys and block_idx < len(extra_keys) else None
        block_hash = hash_block_tokens(previous, token_ids[start:end], block_extra_keys)
        hashes.append(block_hash)
        previous = block_hash

    return hashes


def normalize_external_hash(value: Any, group_idx: int | None = None) -> tuple[Any, int | None]:
    if isinstance(value, bytes):
        key: Any = ("bytes", value)
    elif isinstance(value, bytearray):
        key = ("bytes", bytes(value))
    elif isinstance(value, list):
        key = ("list", tuple(canonicalize(value)))
    else:
        key = value
    return key, group_idx


def derive_request_extra_keys(body: dict[str, Any] | None, block_count: int) -> list[tuple[Any, ...] | None]:
    body = body or {}
    lora_keys: list[Any] = []
    for key in ("lora_name", "lora_id"):
        value = body.get(key)
        if value is not None:
            lora_keys.append(value)

    cache_salt = body.get("cache_salt", body.get("kv_cache_salt"))

    explicit_extra_keys = body.get("kv_extra_keys")
    if isinstance(explicit_extra_keys, list):
        if len(explicit_extra_keys) == block_count:
            return [tuple(item) if isinstance(item, list) else item for item in explicit_extra_keys]
        lora_keys.extend(explicit_extra_keys)

    extra_keys: list[tuple[Any, ...] | None] = []
    for block_idx in range(block_count):
        keys = list(lora_keys)
        if block_idx == 0 and cache_salt is not None:
            keys.append(cache_salt)
        extra_keys.append(tuple(keys) if keys else None)
    return extra_keys


def split_full_blocks(token_ids: Sequence[int], block_size: int, max_blocks: int | None = None) -> list[int]:
    full_len = (len(token_ids) // block_size) * block_size
    if max_blocks is not None:
        full_len = min(full_len, max_blocks * block_size)
    return list(token_ids[:full_len])
