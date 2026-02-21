"""STUNIR IR bundle v1 (authoritative) reference implementation.

This implements the authoritative spec where Canonical IR Content (CIR) is:
- a JSON array of JSON objects ("units"), fully normalized;
- encoded to canonical DCBOR (RFC 8949 canonical CBOR; definite lengths only);
- with floats forbidden, and all strings NFC-normalized.

The IR bundle bytes are canonical DCBOR encoding of:
  ["stunir.ir_bundle.v1", cir_units]

Byte-exactness applies ONLY to the IR bundle bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


class ValidationError(Exception):
    pass


class _NoDupesObjectPairs:
    def __init__(self, pairs: List[Tuple[str, Any]]):
        """Store JSON object pairs for duplicate detection."""
        self.pairs = pairs


def _nfc(s: str) -> str:
    """Normalize a string to NFC and validate code points."""
    if not isinstance(s, str):
        raise ValidationError(f"expected string, got {type(s).__name__}")
    for ch in s:
        oc = ord(ch)
        if 0xD800 <= oc <= 0xDFFF:
            raise ValidationError("string contains surrogate code point")
    return unicodedata.normalize("NFC", s)


def _json_load_no_dupes(fp) -> Any:
    """Load JSON while rejecting duplicate keys.

    Note: this rejects exact duplicates at parse time.
    Duplicates introduced by NFC normalization are rejected during normalization.
    """

    def hook(pairs):
        seen = set()
        for k, _ in pairs:
            if k in seen:
                raise ValidationError(f"duplicate key in JSON object: {k!r}")
            seen.add(k)
        return _NoDupesObjectPairs(pairs)

    return json.load(fp, object_pairs_hook=hook)


def normalize_json_value(v: Any) -> Any:
    """Normalize a CIR JSON value under the v1 profile.

    Allowed types:
      - null, boolean, integer
      - string (NFC)
      - array (recursive)
      - object (keys are NFC, recursive values)

    Forbidden:
      - floats
      - duplicate keys (including duplicates introduced by NFC)
      - surrogate code points in strings
    """

    if isinstance(v, _NoDupesObjectPairs):
        norm_items: List[Tuple[str, Any]] = []
        for k, val in v.pairs:
            if not isinstance(k, str):
                raise ValidationError("JSON object key is not a string")
            nk = _nfc(k)
            norm_items.append((nk, normalize_json_value(val)))

        seen = set()
        for nk, _ in norm_items:
            if nk in seen:
                raise ValidationError(f"duplicate key after NFC normalization: {nk!r}")
            seen.add(nk)

        return {k: val for k, val in norm_items}

    if v is None:
        return None

    if isinstance(v, bool):
        return v

    if isinstance(v, int):
        return v

    if isinstance(v, float):
        raise ValidationError("floats are forbidden in CIR JSON profile")

    if isinstance(v, str):
        return _nfc(v)

    if isinstance(v, list):
        return [normalize_json_value(x) for x in v]

    if isinstance(v, dict):
        # Defensive: should not happen when using _json_load_no_dupes.
        norm: Dict[str, Any] = {}
        for k, val in v.items():
            if not isinstance(k, str):
                raise ValidationError("JSON object key is not a string")
            nk = _nfc(k)
            if nk in norm:
                raise ValidationError(f"duplicate key after NFC normalization: {nk!r}")
            norm[nk] = normalize_json_value(val)
        return norm

    raise ValidationError(f"unsupported JSON type: {type(v).__name__}")


def sha256_hex(data: bytes) -> str:
    """Compute a SHA-256 digest for bytes."""
    return hashlib.sha256(data).hexdigest()


def _encode_major_int(major: int, value: int) -> bytes:
    """Encode an unsigned integer value for a CBOR major type."""
    if value < 0:
        raise ValidationError("internal error: negative additional integer")

    if value < 24:
        return bytes([(major << 5) | value])

    if value < 256:
        return bytes([(major << 5) | 24, value])

    if value < 65536:
        return bytes([(major << 5) | 25]) + value.to_bytes(2, "big")

    if value < 4294967296:
        return bytes([(major << 5) | 26]) + value.to_bytes(4, "big")

    if value < 18446744073709551616:
        return bytes([(major << 5) | 27]) + value.to_bytes(8, "big")

    raise ValidationError("integer out of uint64 range for direct encoding")


def _encode_uint(n: int) -> bytes:
    """Encode a non-negative integer as CBOR unsigned integer."""
    return _encode_major_int(0, n)


def _encode_nint(n: int) -> bytes:
    """Encode a negative integer as CBOR negative integer."""
    return _encode_major_int(1, (-1 - n))


def _encode_bytes(b: bytes) -> bytes:
    """Encode a bytes value as CBOR byte string."""
    b = bytes(b)
    return _encode_major_int(2, len(b)) + b


def _encode_text(s: str) -> bytes:
    """Encode a string as CBOR text with NFC normalization."""
    s = _nfc(s)
    try:
        b = s.encode("utf-8")
    except UnicodeEncodeError as e:
        raise ValidationError(f"invalid UTF-8 string: {e}")
    return _encode_major_int(3, len(b)) + b


def _encode_array(items: List[Any]) -> bytes:
    """Encode a list as a CBOR array."""
    out = bytearray()
    out += _encode_major_int(4, len(items))
    for x in items:
        out += dcbor_encode(x)
    return bytes(out)


def _canonical_map_sort_key(encoded_key: bytes):
    """Return the canonical map sort key for an encoded CBOR key."""
    # RFC 8949 canonical order: sort by length then lexicographic bytes of *encoded key*.
    return (len(encoded_key), encoded_key)


def _encode_map(m: Dict[str, Any]) -> bytes:
    """Encode a dict as a CBOR map with canonical key ordering."""
    encoded_items: List[Tuple[bytes, bytes]] = []

    for k, v in m.items():
        if not isinstance(k, str):
            raise ValidationError("map key is not a string")
        ek = _encode_text(k)
        ev = dcbor_encode(v)
        encoded_items.append((ek, ev))

    encoded_items.sort(key=lambda kv: _canonical_map_sort_key(kv[0]))

    out = bytearray()
    out += _encode_major_int(5, len(encoded_items))
    for ek, ev in encoded_items:
        out += ek
        out += ev
    return bytes(out)


def _encode_tag(tag_num: int, tagged_bytes: bytes) -> bytes:
    """Encode a CBOR tag and its tagged bytes."""
    return _encode_major_int(6, tag_num) + tagged_bytes


def _encode_bignum(n: int) -> bytes:
    """Encode an integer using CBOR bignum tags 2 or 3."""
    # RFC 8949: tag(2) for positive bignum, tag(3) for negative bignum.
    if n >= 0:
        tag = 2
        mag = n
    else:
        tag = 3
        mag = -1 - n

    if mag == 0:
        b = b""
    else:
        blen = (mag.bit_length() + 7) // 8
        b = mag.to_bytes(blen, "big")

    return _encode_tag(tag, _encode_bytes(b))


def dcbor_encode(v: Any) -> bytes:
    """Canonical DCBOR encoding for the IR bundle v1 subset.

    Supported types:
      - None, bool, int, str, list, dict

    Forbidden:
      - float (always)
      - indefinite-length items (not representable via this encoder)

    Notes:
      - Integers outside uint64/nint64 are encoded via bignum tags 2/3.
      - Map key canonical ordering follows RFC 8949 canonical CBOR ordering.
    """

    if v is None:
        return b"\xf6"  # null

    if v is False:
        return b"\xf4"

    if v is True:
        return b"\xf5"

    if isinstance(v, int):
        if 0 <= v < 18446744073709551616:
            return _encode_uint(v)
        if -18446744073709551616 <= v < 0:
            return _encode_nint(v)
        return _encode_bignum(v)

    if isinstance(v, float):
        raise ValidationError("floats are forbidden in DCBOR for IR bundle v1")

    if isinstance(v, str):
        return _encode_text(v)

    if isinstance(v, list):
        return _encode_array(v)

    if isinstance(v, dict):
        return _encode_map(v)

    raise ValidationError(f"unsupported type for CBOR encoding: {type(v).__name__}")


def load_cir_units_json(path: str) -> List[Dict[str, Any]]:
    """Load and normalize CIR units from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        raw = _json_load_no_dupes(f)

    norm = normalize_json_value(raw)

    if not isinstance(norm, list):
        raise ValidationError("top-level CIR must be a JSON array (list) of unit objects")

    for i, unit in enumerate(norm):
        if not isinstance(unit, dict):
            raise ValidationError(f"unit[{i}] must be a JSON object")

    return norm


def make_ir_bundle_bytes(cir_units: List[Dict[str, Any]]) -> bytes:
    """Encode CIR units into IR bundle bytes."""
    bundle_value = ["stunir.ir_bundle.v1", cir_units]
    return dcbor_encode(bundle_value)


def make_receipt(*, mode: str, engine_type: str, engine_version: str, cir_sha256: str, ir_bundle_sha256: str) -> Dict[str, Any]:
    """Create a receipt record for an IR bundle emission."""
    return {
        "ir_bundle.mode": mode,
        "ir_bundle.id": "stunir.ir_bundle.v1",
        "ir_bundle.format_version": 1,
        "ir_bundle.sha256": ir_bundle_sha256,
        "ir_bundle.cir_sha256": cir_sha256,
        "ir_bundle.engine.type": engine_type,
        "ir_bundle.engine.version_or_build": engine_version,
        "ir_bundle.encoding.family": "dcbor",
        "ir_bundle.float_policy": "forbid_floats",
        "ir_bundle.unicode_normalization": "NFC",
    }


def _main(argv: List[str] | None = None) -> int:
    """CLI entry point for emitting an IR bundle v1 and receipt."""
    ap = argparse.ArgumentParser(description="Build STUNIR IR bundle v1 from CIR units JSON")
    ap.add_argument("--units-json", required=True, help="Path to cir_units JSON file (array of objects)")
    ap.add_argument("--out-bundle", required=True, help="Path to write IR bundle bytes")
    ap.add_argument("--out-receipt", required=True, help="Path to write receipt JSON")
    ap.add_argument("--mode", default="byte_exact", choices=["byte_exact", "semantic"], help="Receipt mode label")
    ap.add_argument("--engine-type", default="python_ref", help="Receipt engine type")
    ap.add_argument("--engine-version", default="stunir.tools.ir_bundle_v1:ref", help="Receipt engine version/build")

    args = ap.parse_args(argv)

    cir_units = load_cir_units_json(args.units_json)

    cir_bytes = dcbor_encode(cir_units)
    cir_sha = sha256_hex(cir_bytes)

    bundle_bytes = make_ir_bundle_bytes(cir_units)
    bundle_sha = sha256_hex(bundle_bytes)

    with open(args.out_bundle, "wb") as f:
        f.write(bundle_bytes)

    receipt = make_receipt(
        mode=args.mode,
        engine_type=args.engine_type,
        engine_version=args.engine_version,
        cir_sha256=cir_sha,
        ir_bundle_sha256=bundle_sha,
    )

    with open(args.out_receipt, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(_main())