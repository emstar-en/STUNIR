#!/usr/bin/env python3
"""Deterministic CBOR encoder (canonical-ish) for JSON-like data.

Goal:
  - produce deterministic bytes for a Python object tree
  - support JSON types: null/bool/int/float/str/list/dict

Notes:
  - This implements Canonical CBOR map ordering (RFC 8949) for map keys.
  - Floats are encoded as IEEE-754 float64 (deterministic, not minimal).
  - This is intentionally small and dependency-free.

If you need strict dCBOR compliance (additional restrictions + shortest float),
tighten the allowed types/ranges and float encoding policy.
"""

from __future__ import annotations

import struct
from typing import Any


def _encode_uint(ai: int, n: int) -> bytes:
    if n < 0:
        raise ValueError('uint must be non-negative')
    if n < 24:
        return bytes([(ai << 5) | n])
    if n < 256:
        return bytes([(ai << 5) | 24, n])
    if n < 65536:
        return bytes([(ai << 5) | 25]) + struct.pack('>H', n)
    if n < 4294967296:
        return bytes([(ai << 5) | 26]) + struct.pack('>I', n)
    if n < 18446744073709551616:
        return bytes([(ai << 5) | 27]) + struct.pack('>Q', n)
    raise ValueError('uint too large')


def _encode_int(n: int) -> bytes:
    if n >= 0:
        return _encode_uint(0, n)
    # CBOR negative integer is encoded as (-1 - n)
    return _encode_uint(1, -1 - n)


def _encode_bytes(b: bytes) -> bytes:
    return _encode_uint(2, len(b)) + b


def _encode_text(s: str) -> bytes:
    b = s.encode('utf-8')
    return _encode_uint(3, len(b)) + b


def _encode_array(items: list[Any]) -> bytes:
    out = bytearray()
    out += _encode_uint(4, len(items))
    for it in items:
        out += dumps(it)
    return bytes(out)


def _encode_map(m: dict[Any, Any]) -> bytes:
    # Canonical CBOR: sort map keys by (len(encoded_key), encoded_key)
    enc_items = []
    for k, v in m.items():
        k_enc = dumps(k)
        v_enc = dumps(v)
        enc_items.append((k_enc, v_enc))
    enc_items.sort(key=lambda kv: (len(kv[0]), kv[0]))

    out = bytearray()
    out += _encode_uint(5, len(enc_items))
    for k_enc, v_enc in enc_items:
        out += k_enc
        out += v_enc
    return bytes(out)


def dumps(obj: Any) -> bytes:
    """Encode obj into deterministic CBOR bytes."""
    if obj is None:
        return b'ö'  # null
    if obj is False:
        return b'ô'
    if obj is True:
        return b'õ'

    if isinstance(obj, int) and not isinstance(obj, bool):
        return _encode_int(obj)

    if isinstance(obj, bytes):
        return _encode_bytes(obj)

    if isinstance(obj, str):
        return _encode_text(obj)

    if isinstance(obj, (list, tuple)):
        return _encode_array(list(obj))

    if isinstance(obj, dict):
        return _encode_map(obj)

    if isinstance(obj, float):
        # Deterministic float policy: always float64.
        return b'û' + struct.pack('>d', obj)

    raise TypeError(f'Unsupported type for CBOR encoding: {type(obj).__name__}')


def sha256_hex(b: bytes) -> str:
    import hashlib
    return hashlib.sha256(b).hexdigest()
