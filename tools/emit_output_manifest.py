#!/usr/bin/env python3
"""Emit a deterministic output manifest for a directory tree.

Manifest schema: stunir.output_manifest.v1
- root: repo-relative directory path
- files[]: {file, sha256, bytes_len}

The manifest is canonical JSON (stunir-json-c14n-v1 + trailing newline).
"""

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_canon_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n',
        encoding='utf-8',
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--manifest-out', required=True)
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists() or not root.is_dir():
        raise SystemExit(f'root must exist and be a directory: {root}')

    files = [p for p in root.rglob('*') if p.is_file()]
    files.sort(key=lambda p: p.relative_to(root).as_posix())

    out = {
        'schema': 'stunir.output_manifest.v1',
        'root': root.as_posix(),
        'files': [],
    }

    for p in files:
        rel = p.relative_to(root).as_posix()
        bsz = p.stat().st_size
        out['files'].append({'file': rel, 'sha256': sha256_file(p), 'bytes_len': int(bsz)})

    write_canon_json(Path(args.manifest_out), out)


if __name__ == '__main__':
    main()
