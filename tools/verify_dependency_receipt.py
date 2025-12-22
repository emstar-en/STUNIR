#!/usr/bin/env python3
"""Verify a dependency acceptance receipt.

Checks identity-level evidence and receipt integrity. For full determinism
re-probing, generate a fresh receipt with probe_dependency.py.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def canon_bytes(obj) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')


def run_cmd(argv, cwd=None, timeout=60):
    p = subprocess.run(argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False)
    return {
        'argv': argv,
        'exit_code': p.returncode,
        'stdout_sha256': sha256_bytes(p.stdout),
        'stderr_sha256': sha256_bytes(p.stderr),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--receipt', required=True)
    ap.add_argument('--contract', required=True)
    ap.add_argument('--tool', default=None)
    args = ap.parse_args()

    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    contract_path = args.contract
    if not os.path.isabs(contract_path):
        contract_path = os.path.join(repo, contract_path)

    receipt = load_json(args.receipt)
    contract = load_json(contract_path)

    tool = args.tool or (receipt.get('tool') or {}).get('resolved')
    if tool and not (os.path.sep in tool):
        tool = shutil.which(tool) or tool

    if not tool:
        print('FAIL: no tool specified')
        raise SystemExit(2)

    tool_sha = None
    if os.path.exists(tool) and os.path.isfile(tool):
        tool_sha = sha256_file(tool)

    expected_sha = (receipt.get('tool') or {}).get('sha256')
    if expected_sha and tool_sha and expected_sha != tool_sha:
        print('FAIL: tool executable sha256 mismatch')
        raise SystemExit(3)

    mapping = {'tool': tool, 'repo': repo}
    def apply(argv):
        out=[]
        for a in argv:
            for k,v in mapping.items():
                a=a.replace('{'+k+'}', v)
            out.append(a)
        return out

    id_cmds = contract.get('identity', {}).get('commands', [])
    got = []
    for cmd in id_cmds:
        r = run_cmd(apply(cmd), cwd=repo)
        got.append({k: r[k] for k in ('exit_code','stdout_sha256','stderr_sha256')})

    exp = []
    for r in (receipt.get('identity', {}) or {}).get('commands', []):
        exp.append({k: r.get(k) for k in ('exit_code','stdout_sha256','stderr_sha256')})

    if exp and got and exp != got:
        print('FAIL: identity probe mismatch')
        raise SystemExit(4)

    dep_core = {
        'contract_name': receipt.get('contract_name'),
        'tool': receipt.get('tool'),
        'identity': receipt.get('identity'),
        'attestations': ((receipt.get('inputs_presented') or {}).get('attestations') or []),
        'platform': receipt.get('platform'),
    }
    dep_id = sha256_bytes(canon_bytes(dep_core))
    if receipt.get('dependency_id') != dep_id:
        print('FAIL: dependency_id mismatch')
        raise SystemExit(5)

    print('OK')


if __name__ == '__main__':
    main()
