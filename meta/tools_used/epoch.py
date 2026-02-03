#!/usr/bin/env python3
"""STUNIR epoch selection.

This tool chooses a single integer epoch and writes a small manifest.

Design goals:
- Deterministic by default (no wall-clock time unless explicitly permitted)
- Transparent: record all candidate sources + derivation details

Selection priority (first available wins):
1) STUNIR_BUILD_EPOCH (explicit override)
2) SOURCE_DATE_EPOCH (standard reproducible-build knob)
3) DERIVED_SPEC_DIGEST_V1 (deterministic; derived from spec/ tree digest)
4) GIT_COMMIT_EPOCH (git commit timestamp when available)
5) ZERO (0) unless --allow-current-time is set

The derived epoch is *not* intended to represent wall-clock time. It is a
stable, content-addressed epoch in a conventional Unix-epoch range.
"""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


def to_int(v):
    try:
        return int(str(v).strip())
    except Exception:
        return None


def sha256_of_dir(root: Path) -> str:
    """Deterministic directory digest: sha256 over (relpath + bytes) in sorted traversal."""
    if not root.exists():
        return '0' * 64
    h = hashlib.sha256()
    for p in sorted(root.rglob('*')):
        if p.is_file():
            h.update(p.relative_to(root).as_posix().encode('utf-8'))
            with p.open('rb') as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b''):
                    h.update(chunk)
    return h.hexdigest()


def derive_epoch_from_spec_digest_v1(spec_digest_hex: str) -> int:
    """Map a spec tree sha256 into a conventional Unix-epoch range deterministically.

    Formula:
        base + (u64(first16hex(spec_digest)) % span)

    Where:
      - base = 2000-01-01T00:00:00Z = 946684800
      - span = 100 * 365 * 86400 = 3153600000 seconds

    This yields an epoch in [2000-01-01, ~2099-12-07], deterministic per spec digest.
    """
    base = 946684800
    span = 3153600000
    word = int(spec_digest_hex[:16], 16)
    return base + (word % span)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec-root', default='spec', help='Spec root used for deterministic epoch derivation')
    ap.add_argument('--out-json', help='Write full epoch manifest to this path')
    ap.add_argument('--print-epoch', action='store_true', help='Print only the selected epoch to stdout')
    ap.add_argument(
        '--allow-current-time',
        action='store_true',
        help='If set and no deterministic sources are available, fall back to wall-clock time (non-deterministic).',
    )
    args = ap.parse_args()

    inputs = {
        'STUNIR_BUILD_EPOCH': to_int(os.environ.get('STUNIR_BUILD_EPOCH')),
        'SOURCE_DATE_EPOCH': to_int(os.environ.get('SOURCE_DATE_EPOCH')),
        'DERIVED_SPEC_DIGEST_V1': None,
        'GIT_COMMIT_EPOCH': None,
        'SPEC_DIGEST_SHA256': None,
    }

    # Try to compute GIT_COMMIT_EPOCH if in a repo
    try:
        out = subprocess.run(['git', 'log', '-1', '--format=%ct'], capture_output=True, text=True, check=False)
        inputs['GIT_COMMIT_EPOCH'] = to_int(out.stdout.strip())
    except Exception:
        inputs['GIT_COMMIT_EPOCH'] = None

    # Deterministic derivation from spec/ digest
    try:
        spec_root = Path(args.spec_root)
        spec_digest = sha256_of_dir(spec_root)
        inputs['SPEC_DIGEST_SHA256'] = spec_digest
        # Even if spec/ is missing, sha256_of_dir returns 0*64 deterministically.
        inputs['DERIVED_SPEC_DIGEST_V1'] = derive_epoch_from_spec_digest_v1(spec_digest)
    except Exception:
        inputs['SPEC_DIGEST_SHA256'] = None
        inputs['DERIVED_SPEC_DIGEST_V1'] = None

    selected_epoch = None
    source = None
    for key in ('STUNIR_BUILD_EPOCH', 'SOURCE_DATE_EPOCH', 'DERIVED_SPEC_DIGEST_V1', 'GIT_COMMIT_EPOCH'):
        if inputs.get(key) is not None:
            selected_epoch = inputs[key]
            source = key
            break

    if selected_epoch is None:
        if args.allow_current_time:
            selected_epoch = int(time.time())
            source = 'CURRENT_TIME'
        else:
            selected_epoch = 0
            source = 'ZERO'

    manifest = {
        'selected_epoch': int(selected_epoch),
        'source': str(source),
        'inputs': inputs,
        'derivation': {
            'version': 1,
            'method': 'spec_digest_v1',
            'spec_root': args.spec_root,
            'spec_digest_sha256': inputs.get('SPEC_DIGEST_SHA256'),
            'mapping': {
                'base_epoch': 946684800,
                'span_seconds': 3153600000,
                'formula': 'base + (u64(first16hex(spec_digest_sha256)) % span)',
            },
        },
    }

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(manifest, sort_keys=True, separators=(',', ':'), ensure_ascii=False) + '\n',
            encoding='utf-8',
        )

    if args.print_epoch:
        print(int(selected_epoch))


if __name__ == '__main__':
    main()
