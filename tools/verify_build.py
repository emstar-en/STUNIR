#!/usr/bin/env python3
"""Verify STUNIR build artifacts and receipts.

This is intended to be a *small checker*.

Verifies (when present):
  - build/provenance.json matches recomputed digests of spec/ and asm/
  - build/provenance.h matches recomputed provenance
  - receipts/spec_ir.json sha256 matches asm/spec_ir.txt
  - receipts/ir_manifest.json matches asm/ir/* file sha256s (exact file set in --strict mode)
  - receipts/prov_emit.json sha256 matches bin/prov_emit when status=BINARY_EMITTED
  - each receipt epoch.selected_epoch matches build/epoch.json selected_epoch

Exit codes:
  0 OK
  2 missing required files
  3 mismatch
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def sha256_of_dir(root: Path) -> str:
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


def load_json(p: Path):
    return json.loads(p.read_text(encoding='utf-8'))


def fail(msg: str):
    print('FAIL:', msg)
    raise SystemExit(3)


def run_py(repo: Path, tool_rel: str, argv: list[str]) -> subprocess.CompletedProcess:
    cmd = ['python3', '-B', str(repo / tool_rel)] + argv
    return subprocess.run(cmd, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default='.')
    ap.add_argument('--strict', action='store_true')
    args = ap.parse_args()

    repo = Path(args.repo).resolve()

    epoch_json = repo / 'build' / 'epoch.json'
    prov_json = repo / 'build' / 'provenance.json'
    prov_h = repo / 'build' / 'provenance.h'

    if not epoch_json.exists():
        fail('missing build/epoch.json (run scripts/build.sh first)')
    if not prov_json.exists():
        fail('missing build/provenance.json (run scripts/build.sh first)')

    epoch = load_json(epoch_json)
    selected_epoch = epoch.get('selected_epoch')

    prov = load_json(prov_json)

    if selected_epoch is not None and prov.get('build_epoch') is not None and int(prov.get('build_epoch')) != int(selected_epoch):
        fail('build/provenance.json build_epoch does not match build/epoch.json selected_epoch')

    spec_digest = sha256_of_dir(repo / 'spec')
    asm_digest = sha256_of_dir(repo / 'asm')

    if prov.get('spec_digest') != spec_digest:
        fail(f"spec_digest mismatch: expected {prov.get('spec_digest')} got {spec_digest}")
    if prov.get('asm_digest') != asm_digest:
        fail(f"asm_digest mismatch: expected {prov.get('asm_digest')} got {asm_digest}")

    # Recompute provenance artifacts and compare bytes
    tmp = repo / 'build' / '.verify_tmp'
    if tmp.exists():
        for p in sorted(tmp.rglob('*'), reverse=True):
            if p.is_file():
                p.unlink()
            else:
                try:
                    p.rmdir()
                except Exception:
                    pass
        try:
            tmp.rmdir()
        except Exception:
            pass
    tmp.mkdir(parents=True, exist_ok=True)
    tmp_h = tmp / 'provenance.h'
    tmp_j = tmp / 'provenance.json'

    r = run_py(repo, 'tools/gen_provenance.py', [
        '--epoch', str(prov.get('build_epoch', 0)),
        '--spec-root', 'spec',
        '--asm-root', 'asm',
        '--out-header', str(tmp_h),
        '--out-json', str(tmp_j),
        '--epoch-source', str(prov.get('epoch_source', 'UNKNOWN')),
    ])
    if r.returncode != 0:
        fail('gen_provenance.py failed during verification')

    if prov_json.read_bytes() != tmp_j.read_bytes():
        fail('build/provenance.json is not reproducible from inputs')
    if prov_h.exists() and tmp_h.exists() and prov_h.read_bytes() != tmp_h.read_bytes():
        fail('build/provenance.h is not reproducible from inputs')

    # Verify spec_ir receipt
    spec_ir = repo / 'asm' / 'spec_ir.txt'
    spec_ir_receipt = repo / 'receipts' / 'spec_ir.json'
    if spec_ir_receipt.exists():
        rr = load_json(spec_ir_receipt)
        if rr.get('sha256') and spec_ir.exists():
            got = sha256_file(spec_ir)
            if rr.get('sha256') != got:
                fail('receipts/spec_ir.json sha256 does not match asm/spec_ir.txt')
        e = (rr.get('epoch') or {}).get('selected_epoch')
        if selected_epoch is not None and e is not None and int(e) != int(selected_epoch):
            fail('spec_ir receipt epoch.selected_epoch does not match build/epoch.json')

    # Verify IR manifest and asm/ir set
    man_path = repo / 'receipts' / 'ir_manifest.json'
    ir_root = repo / 'asm' / 'ir'
    if man_path.exists():
        man = load_json(man_path)
        listed = {f['file']: f['sha256'] for f in (man.get('files') or [])}
        actual_files = []
        if ir_root.exists():
            for p in sorted(ir_root.rglob('*.json')):
                if p.is_file():
                    rel = p.relative_to(ir_root).as_posix()
                    actual_files.append(rel)
                    got = sha256_file(p)
                    exp = listed.get(rel)
                    if exp != got:
                        fail(f"IR file sha256 mismatch for {rel}: expected {exp} got {got}")
        if args.strict:
            if set(actual_files) != set(listed.keys()):
                fail('IR manifest file set does not match asm/ir contents')
        e = (man.get('epoch') or {}).get('selected_epoch')
        if selected_epoch is not None and e is not None and int(e) != int(selected_epoch):
            fail('ir_manifest epoch.selected_epoch does not match build/epoch.json')

    # Verify prov_emit receipt
    prov_emit_receipt = repo / 'receipts' / 'prov_emit.json'
    prov_emit_bin = repo / 'bin' / 'prov_emit'
    if prov_emit_receipt.exists():
        rr = load_json(prov_emit_receipt)
        st = rr.get('status')
        if st == 'BINARY_EMITTED':
            if not prov_emit_bin.exists():
                fail('prov_emit receipt says BINARY_EMITTED but bin/prov_emit is missing')
            got = sha256_file(prov_emit_bin)
            if rr.get('sha256') != got:
                fail('prov_emit receipt sha256 does not match bin/prov_emit')
        e = (rr.get('epoch') or {}).get('selected_epoch')
        if selected_epoch is not None and e is not None and int(e) != int(selected_epoch):
            fail('prov_emit receipt epoch.selected_epoch does not match build/epoch.json')

    print('OK')


if __name__ == '__main__':
    main()
