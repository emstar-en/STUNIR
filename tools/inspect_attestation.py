#!/usr/bin/env python3
"""Inspect attestation documents and classify them deterministically."""

import argparse
import hashlib
import json
import os


def sha256_file(path: str) -> str:
    """Compute the SHA-256 digest for a file path."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: str):
    """Load JSON from a file path."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def canon_bytes(obj) -> bytes:
    """Serialize an object to canonical JSON bytes."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')


def sha256_bytes(b: bytes) -> str:
    """Compute the SHA-256 digest for bytes."""
    return hashlib.sha256(b).hexdigest()


def guess_kind(obj):
    """Guess attestation kind from a parsed JSON object."""
    if isinstance(obj, dict):
        if obj.get('spdxVersion') is not None:
            return 'spdx'
        if obj.get('bomFormat') == 'CycloneDX':
            return 'cyclonedx'
        if obj.get('_type') and 'in-toto.io/Statement' in str(obj.get('_type')):
            return 'in_toto_statement'
        if obj.get('predicateType') and 'slsa.dev/provenance' in str(obj.get('predicateType')):
            return 'slsa_provenance'
        if 'payload' in obj and 'signatures' in obj:
            return 'dsse_envelope'
    return 'unknown_json'


def main() -> None:
    """Inspect attestation files and emit a canonical summary."""
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    out = {'attestations': []}
    for p in args.paths:
        rec = {'path': p, 'sha256': None, 'content_type': None, 'kind_guess': None, 'canonical_digest': None}
        if os.path.exists(p) and os.path.isfile(p):
            rec['sha256'] = sha256_file(p)
            try:
                obj = load_json(p)
                rec['content_type'] = 'json'
                rec['kind_guess'] = guess_kind(obj)
                rec['canonical_digest'] = sha256_bytes(canon_bytes(obj))
            except Exception:
                rec['content_type'] = 'opaque'
                rec['kind_guess'] = 'opaque'
        out['attestations'].append(rec)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'wb') as f:
        f.write(canon_bytes(out))


if __name__ == '__main__':
    main()
