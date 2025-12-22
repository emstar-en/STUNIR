#!/usr/bin/env python3
"""Verify that dependency acceptance receipts satisfy resolved requirements."""

import argparse
import hashlib
import json
import os


def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def canon_bytes(obj) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def recompute_dependency_id(receipt: dict) -> str:
    dep_core = {
        'contract_name': receipt.get('contract_name'),
        'tool': receipt.get('tool'),
        'identity': receipt.get('identity'),
        'attestations': ((receipt.get('inputs_presented') or {}).get('attestations') or []),
        'platform': receipt.get('platform'),
    }
    return sha256_bytes(canon_bytes(dep_core))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--requirements', required=True)
    ap.add_argument('--deps_dir', default='receipts/deps')
    args = ap.parse_args()

    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    req_path = args.requirements
    if not os.path.isabs(req_path):
        req_path = os.path.join(repo, req_path)

    deps_dir = args.deps_dir
    if not os.path.isabs(deps_dir):
        deps_dir = os.path.join(repo, deps_dir)

    req = load_json(req_path)
    if req.get('requirements_type') != 'backend_requirements':
        print('FAIL: requirements_type mismatch')
        raise SystemExit(2)

    missing = []
    bad = []

    for item in req.get('required_contracts', []):
        cname = item.get('contract_name')
        if not cname:
            continue
        rpath = os.path.join(deps_dir, f'{cname}.json')
        if not os.path.exists(rpath):
            missing.append(cname)
            continue
        r = load_json(rpath)
        if r.get('receipt_type') != 'dependency_acceptance':
            bad.append((cname, 'wrong_receipt_type'))
            continue
        if r.get('contract_name') != cname:
            bad.append((cname, 'contract_name_mismatch'))
            continue
        if not r.get('accepted'):
            bad.append((cname, 'not_accepted'))
            continue
        if r.get('dependency_id') != recompute_dependency_id(r):
            bad.append((cname, 'dependency_id_inconsistent'))
            continue

    if missing or bad:
        if missing:
            print('MISSING:', ','.join(missing))
        if bad:
            print('BAD:', ';'.join([f'{c}:{reason}' for c, reason in bad]))
        raise SystemExit(3)

    print('OK')


if __name__ == '__main__':
    main()
