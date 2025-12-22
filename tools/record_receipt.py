#!/usr/bin/env python3
"""Record a build receipt.

Backwards-compatible with legacy positional invocation:

    record_receipt.py <target_path> <receipt_path> <status> <build_epoch> [epoch_json|-] [exception_reason]

New CLI:
    record_receipt.py --target <path> --receipt <path> --status <STATUS> --build-epoch <int>
                     [--epoch-json <path>|-] [--exception-reason <text>]
                     [--inputs <path> ...] [--input-dirs <dir> ...]
                     [--tool <path>] [--argv <arg> ...] [--include-platform 0|1]

Design goals:
  - receipts are machine-checkable
  - include a deterministic `receipt_core_id_sha256` that *excludes* platform noise
  - optionally include platform information for auditing
"""

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canon_bytes(obj) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')


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


def load_epoch_manifest(epoch_json: Path | None, build_epoch: int):
    if epoch_json and epoch_json != Path('-') and epoch_json.exists():
        try:
            return json.loads(epoch_json.read_text(encoding='utf-8'))
        except Exception:
            return None
    return None


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n',
        encoding='utf-8',
    )


def parse_legacy(argv: list[str]):
    if len(argv) < 5:
        return None
    if argv[1].startswith('-'):
        return None
    target = argv[1]
    receipt = argv[2]
    status = argv[3]
    build_epoch = int(argv[4])
    epoch_json = argv[5] if len(argv) > 5 else None
    exception_reason = argv[6] if len(argv) > 6 else None
    return {
        'target': target,
        'receipt': receipt,
        'status': status,
        'build_epoch': build_epoch,
        'epoch_json': epoch_json,
        'exception_reason': exception_reason,
        'inputs': [],
        'input_dirs': [],
        'tool': None,
        'argv': None,
        'include_platform': None,
    }


def parse_args(argv: list[str]):
    legacy = parse_legacy(argv)
    if legacy is not None:
        return legacy

    ap = argparse.ArgumentParser()
    ap.add_argument('--target', required=True)
    ap.add_argument('--receipt', required=True)
    ap.add_argument('--status', required=True)
    ap.add_argument('--build-epoch', required=True, type=int)
    ap.add_argument('--epoch-json', default=None)
    ap.add_argument('--exception-reason', default=None)
    ap.add_argument('--inputs', nargs='*', default=[])
    ap.add_argument('--input-dirs', nargs='*', default=[])
    ap.add_argument('--tool', default=None)
    ap.add_argument('--argv', nargs='*', default=None)
    ap.add_argument('--include-platform', type=int, choices=[0, 1], default=None)
    ns = ap.parse_args(argv[1:])
    return {
        'target': ns.target,
        'receipt': ns.receipt,
        'status': ns.status,
        'build_epoch': ns.build_epoch,
        'epoch_json': ns.epoch_json,
        'exception_reason': ns.exception_reason,
        'inputs': ns.inputs,
        'input_dirs': ns.input_dirs,
        'tool': ns.tool,
        'argv': ns.argv,
        'include_platform': ns.include_platform,
    }


def main():
    a = parse_args(sys.argv)

    target_path = Path(a['target'])
    receipt_path = Path(a['receipt'])
    status = str(a['status'])
    build_epoch = int(a['build_epoch'])

    epoch_json = None
    if a.get('epoch_json') and a['epoch_json'] != '-':
        epoch_json = Path(a['epoch_json'])

    include_platform = a.get('include_platform')
    if include_platform is None:
        include_platform = 1
        if os.environ.get('STUNIR_STRICT', '0') == '1':
            include_platform = int(os.environ.get('STUNIR_INCLUDE_PLATFORM', '0'))
        else:
            include_platform = int(os.environ.get('STUNIR_INCLUDE_PLATFORM', '1'))

    epoch_manifest = load_epoch_manifest(epoch_json, build_epoch) or {
        'selected_epoch': build_epoch,
        'source': 'UNKNOWN',
        'inputs': {},
    }

    target_sha256 = sha256_file(target_path)

    inputs = []
    for p in a.get('inputs') or []:
        pp = Path(p)
        inputs.append({'path': str(pp), 'kind': 'file', 'sha256': sha256_file(pp)})

    for d in a.get('input_dirs') or []:
        dd = Path(d)
        inputs.append({'path': str(dd), 'kind': 'dir', 'sha256': sha256_of_dir(dd)})

    tool_obj = None
    if a.get('tool'):
        tp = Path(a['tool'])
        tool_obj = {'path': str(tp), 'sha256': sha256_file(tp)}

    argv_obj = None
    if a.get('argv'):
        argv_obj = list(a['argv'])

    core = {
        'schema': 'stunir.receipt.build.v1',
        'target': str(target_path),
        'status': status,
        'build_epoch': build_epoch,
        'sha256': target_sha256,
        'epoch': {
            'selected_epoch': epoch_manifest.get('selected_epoch', build_epoch),
            'source': epoch_manifest.get('source', 'UNKNOWN'),
            'inputs': epoch_manifest.get('inputs', {}),
        },
        'inputs': sorted(inputs, key=lambda x: (x['kind'], x['path'])),
        'tool': tool_obj,
        'argv': argv_obj,
    }

    receipt = dict(core)
    receipt['receipt_core_id_sha256'] = sha256_bytes(canon_bytes(core))

    if a.get('exception_reason'):
        receipt['epoch_exception'] = True
        receipt['exception_reason'] = str(a['exception_reason'])

    if include_platform == 1:
        receipt['platform'] = {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'python': platform.python_version(),
        }

    write_json(receipt_path, receipt)
    print(f"Wrote {receipt_path}")


if __name__ == '__main__':
    main()
