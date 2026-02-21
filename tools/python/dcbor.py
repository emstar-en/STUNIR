#!/usr/bin/env python3
"""Deterministic CBOR encoder for a JSON-like data model.

Goals:
  - deterministic bytes for a Python object tree
  - support JSON-ish types: null/bool/int/float/str/list/dict and bytes
  - canonical-ish map ordering (RFC 8949 Canonical CBOR map key ordering)

Float encoding is configurable via an explicit policy enum:
  - forbid_floats: reject float values
  - float64_fixed: always encode floats as IEEE-754 float64 (deterministic, simple)
  - dcbor_shortest: dCBOR-style numeric reduction + shortest float width

Notes:
  - Map ordering uses (len(encoded_key), encoded_key) sorting.
  - This is intentionally small and dependency-free.

References:
  - dCBOR draft: float shortest form + numeric reduction + canonical NaN/inf/-0.
"""

from __future__ import annotations

import math
import struct
from enum import Enum
from typing import Any


class FloatPolicy(str, Enum):
    FORBID_FLOATS = 'forbid_floats'
    FLOAT64_FIXED = 'float64_fixed'
    DCBOR_SHORTEST = 'dcbor_shortest'

    @classmethod
    def parse(cls, v: 'FloatPolicy | str | None') -> 'FloatPolicy':
        if v is None:
            return cls.FLOAT64_FIXED
        if isinstance(v, cls):
            return v
        s = str(v).strip().lower().replace('-', '_')
        aliases = {
            'forbid': cls.FORBID_FLOATS,
            'forbid_float': cls.FORBID_FLOATS,
            'forbid_floats': cls.FORBID_FLOATS,
            'no_floats': cls.FORBID_FLOATS,
            'float64': cls.FLOAT64_FIXED,
            'float64_fixed': cls.FLOAT64_FIXED,
            'fixed64': cls.FLOAT64_FIXED,
            'dcbor': cls.DCBOR_SHORTEST,
            'dcbor_shortest': cls.DCBOR_SHORTEST,
            'shortest': cls.DCBOR_SHORTEST,
        }
        if s in aliases:
            return aliases[s]
        raise ValueError(
            f'Unknown float policy: {v!r}. Expected one of: ' + ', '.join([p.value for p in cls])
        )


def _b(*ints: int) -> bytes:
    return bytes(ints)


def _encode_uint(major_type: int, n: int) -> bytes:
    if n < 0:
        raise ValueError('uint must be non-negative')
    if n < 24:
        return bytes([(major_type << 5) | n])
    if n < 256:
        return bytes([(major_type << 5) | 24, n])
    if n < 65536:
        return bytes([(major_type << 5) | 25]) + struct.pack('>H', n)
    if n < 4294967296:
        return bytes([(major_type << 5) | 26]) + struct.pack('>I', n)
    if n < 18446744073709551616:
        return bytes([(major_type << 5) | 27]) + struct.pack('>Q', n)
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


def _encode_array(items: list[Any], *, float_policy: FloatPolicy) -> bytes:
    out = bytearray()
    out += _encode_uint(4, len(items))
    for it in items:
        out += dumps(it, float_policy=float_policy)
    return bytes(out)


def _encode_map(m: dict[Any, Any], *, float_policy: FloatPolicy) -> bytes:
    # Canonical CBOR: sort map keys by (len(encoded_key), encoded_key)
    enc_items: list[tuple[bytes, bytes]] = []
    for k, v in m.items():
        k_enc = dumps(k, float_policy=float_policy)
        v_enc = dumps(v, float_policy=float_policy)
        enc_items.append((k_enc, v_enc))
    enc_items.sort(key=lambda kv: (len(kv[0]), kv[0]))

    out = bytearray()
    out += _encode_uint(5, len(enc_items))
    for k_enc, v_enc in enc_items:
        out += k_enc
        out += v_enc
    return bytes(out)


def _is_negative_zero(x: float) -> bool:
    return x == 0.0 and math.copysign(1.0, x) < 0.0


def _encode_float_float64_fixed(x: float) -> bytes:
    # Major type 7, AI=27 (float64): 0xfb
    return _b(0xFB) + struct.pack('>d', x)


def _encode_float_dcbor_shortest(x: float) -> bytes:
    """dCBOR-style float handling.

    - NaN -> float16 qNaN 0x7e00
    - +/-inf -> float16 +/-inf
    - +/-0.0 -> integer 0
    - floats with no fractional part -> integer (within supported range)
    - other floats -> shortest width float16/float32/float64 that preserves value

    This encoder does not implement CBOR tags (bignums, decimals); such values are rejected.
    """
    if math.isnan(x):
        return _b(0xF9, 0x7E, 0x00)
    if math.isinf(x):
        return _b(0xF9, 0xFC, 0x00) if x < 0 else _b(0xF9, 0x7C, 0x00)

    # Reduce negative zero and positive zero to integer 0
    if x == 0.0 or _is_negative_zero(x):
        return _b(0x00)

    # Numeric reduction: exact integers become CBOR ints
    if x.is_integer():
        n = int(x)
        # dCBOR narrowing: forbid 65-bit negative integers
        if n < -(2**63):
            raise ValueError('dCBOR policy forbids integers < -2**63')
        if n > (2**64 - 1) or n < -(2**64):
            raise ValueError('dCBOR policy cannot encode integers outside CBOR 64-bit range')
        return _encode_int(n)

    # Fractional: choose shortest IEEE width that round-trips exactly.
    try:
        b16 = struct.pack('>e', x)
        if struct.unpack('>e', b16)[0] == x:
            return _b(0xF9) + b16
    except Exception:
        pass

    try:
        b32 = struct.pack('>f', x)
        if struct.unpack('>f', b32)[0] == x:
            return _b(0xFA) + b32
    except Exception:
        pass

    return _b(0xFB) + struct.pack('>d', x)


def dumps(obj: Any, *, float_policy: FloatPolicy | str | None = None) -> bytes:
    """Encode obj into deterministic CBOR bytes."""
    fp = FloatPolicy.parse(float_policy)

    if obj is None:
        return _b(0xF6)  # null
    if obj is False:
        return _b(0xF4)
    if obj is True:
        return _b(0xF5)

    if isinstance(obj, int) and not isinstance(obj, bool):
        return _encode_int(obj)

    if isinstance(obj, bytes):
        return _encode_bytes(obj)

    if isinstance(obj, str):
        return _encode_text(obj)

    if isinstance(obj, (list, tuple)):
        return _encode_array(list(obj), float_policy=fp)

    if isinstance(obj, dict):
        return _encode_map(obj, float_policy=fp)

    if isinstance(obj, float):
        if fp == FloatPolicy.FORBID_FLOATS:
            raise TypeError('Float values are forbidden by float policy')
        if fp == FloatPolicy.FLOAT64_FIXED:
            return _encode_float_float64_fixed(obj)
        if fp == FloatPolicy.DCBOR_SHORTEST:
            return _encode_float_dcbor_shortest(obj)
        raise ValueError(f'Unhandled float policy: {fp}')

    raise TypeError(f'Unsupported type for CBOR encoding: {type(obj).__name__}')


def sha256_hex(b: bytes) -> str:
    import hashlib
    return hashlib.sha256(b).hexdigest()
