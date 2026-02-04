#!/usr/bin/env python3
"""STUNIR dependency probe.
Accept a user-provided dependency (tool and/or attestations), evaluate it
against a capability contract, and emit a checkable acceptance receipt.
No downloads/installs. No hard pinned manifest.

Bundle C extension:
- tests can treat stdout/stderr as output artifacts by listing '@stdout' and/or '@stderr' in a test's outputs.
- invariants can include: stdout_nonempty, stdout_empty, stderr_nonempty, stderr_empty.
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple


def sha256_bytes(b: bytes) -> str:
    """Compute SHA-256 digest for bytes."""
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str) -> str:
    """Compute SHA-256 digest for a file."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: str) -> Any:
    """Load JSON from a file path."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def canonical_json_bytes(obj: Any) -> bytes:
    """Serialize an object to canonical JSON bytes."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')


def run_cmd(argv: Sequence[str], cwd: Optional[str] = None, env: Optional[Dict[str, str]] = None, timeout: int = 60) -> Dict[str, Any]:
    """Run a command and return exit code and hashed outputs."""
    p = subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    return {
        'argv': list(argv),
        'exit_code': p.returncode,
        'stdout_sha256': sha256_bytes(p.stdout),
        'stderr_sha256': sha256_bytes(p.stderr),
        'stdout_len': len(p.stdout),
        'stderr_len': len(p.stderr),
    }


def resolve_tool(locator: Dict[str, Any], explicit_tool: Optional[str] = None) -> Optional[str]:
    """Resolve a tool path using a locator or explicit tool name."""
    if explicit_tool:
        if os.path.sep in explicit_tool or (os.path.altsep and os.path.altsep in explicit_tool):
            return explicit_tool
        found = shutil.which(explicit_tool)
        return found or explicit_tool

    if locator.get('kind') == 'path_or_command':
        for cand in locator.get('candidates', []):
            found = shutil.which(cand)
            if found:
                return found
        return None

    return None


def apply_placeholders(argv: Sequence[str], mapping: Dict[str, str]) -> List[str]:
    """Substitute placeholders in command arguments using a mapping."""
    out = []
    for a in argv:
        for k, v in mapping.items():
            a = a.replace('{' + k + '}', v)
        out.append(a)
    return out


def eval_invariants(invariants: Optional[List[Dict[str, Any]]], mapping: Dict[str, str], run_record: Optional[Dict[str, Any]] = None) -> Tuple[bool, List[Dict[str, Any]]]:
    """Evaluate invariant checks for a dependency probe run."""
    results = []
    ok = True

    for inv in invariants or []:
        inv_type = inv.get('type')

        if inv_type in ('stdout_nonempty', 'stdout_empty', 'stderr_nonempty', 'stderr_empty'):
            if run_record is None:
                passed = False
            else:
                if inv_type == 'stdout_nonempty':
                    passed = int(run_record.get('stdout_len') or 0) > 0
                elif inv_type == 'stdout_empty':
                    passed = int(run_record.get('stdout_len') or 0) == 0
                elif inv_type == 'stderr_nonempty':
                    passed = int(run_record.get('stderr_len') or 0) > 0
                elif inv_type == 'stderr_empty':
                    passed = int(run_record.get('stderr_len') or 0) == 0
                else:
                    passed = False
            results.append({'type': inv_type, 'passed': passed})
            ok = ok and passed
            continue

        # Default: file/path invariants
        path = inv.get('path', '')
        for k, v in mapping.items():
            path = path.replace('{' + k + '}', v)

        if inv_type == 'file_exists':
            passed = os.path.exists(path)
        elif inv_type == 'file_nonempty':
            passed = os.path.exists(path) and os.path.getsize(path) > 0
        else:
            passed = False

        results.append({'type': inv_type, 'path': path, 'passed': passed})
        ok = ok and passed

    return ok, results


def main() -> None:
    """Run dependency probe and emit an acceptance receipt."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--contract', required=True)
    ap.add_argument('--tool', default=None)
    ap.add_argument('--attestation', action='append', default=[])
    ap.add_argument('--out', required=True)
    ap.add_argument('--require', action='store_true')
    args = ap.parse_args()

    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    contract_path = args.contract
    if not os.path.isabs(contract_path):
        contract_path = os.path.join(repo, contract_path)
    contract = load_json(contract_path)

    tool = resolve_tool(contract.get('tool_locator', {}), explicit_tool=args.tool)

    receipt = {
        'receipt_type': 'dependency_acceptance',
        'contract_name': contract.get('contract_name'),
        'platform': {
            'system': platform.system(),
            'machine': platform.machine(),
            'python': platform.python_version(),
        },
        'inputs_presented': {
            'tool_arg': args.tool,
            'attestations': [],
        },
        'tool': None,
        'identity': {'commands': []},
        'tests': [],
        'accepted': False,
        'acceptance_reason': None,
    }

    for apath in args.attestation:
        rec = {'path': apath, 'sha256': None, 'parsed_type': None, 'parsed_digest': None}
        if os.path.exists(apath) and os.path.isfile(apath):
            rec['sha256'] = sha256_file(apath)
            try:
                obj = load_json(apath)
                rec['parsed_type'] = 'json'
                rec['parsed_digest'] = sha256_bytes(canonical_json_bytes(obj))
            except Exception:
                rec['parsed_type'] = 'opaque'
        receipt['inputs_presented']['attestations'].append(rec)

    if not tool:
        receipt['accepted'] = False
        receipt['acceptance_reason'] = 'tool_not_found'
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'wb') as f:
            f.write(canonical_json_bytes(receipt))
        if args.require:
            raise SystemExit(2)
        return

    tool_sha = None
    if os.path.exists(tool) and os.path.isfile(tool):
        try:
            tool_sha = sha256_file(tool)
        except Exception:
            tool_sha = None

    receipt['tool'] = {
        'resolved': tool,
        'resolved_abs': os.path.abspath(tool) if os.path.exists(tool) else tool,
        'sha256': tool_sha,
    }

    id_cmds = contract.get('identity', {}).get('commands', [])
    accept_codes = set(contract.get('identity', {}).get('accept_exit_codes', [0]))
    mapping = {'tool': tool, 'repo': repo}

    for cmd in id_cmds:
        argv = apply_placeholders(cmd, mapping)
        try:
            r = run_cmd(argv, cwd=repo)
        except Exception:
            r = {'argv': argv, 'exit_code': None, 'stdout_sha256': None, 'stderr_sha256': None, 'stdout_len': None, 'stderr_len': None}
        r['accepted_exit_code'] = (r['exit_code'] in accept_codes) if r['exit_code'] is not None else False
        receipt['identity']['commands'].append(r)

    all_tests_ok = True

    with tempfile.TemporaryDirectory(prefix='stunir_dep_') as td:
        for test in contract.get('tests', []):
            tname = test.get('name')
            trepeat = int(test.get('repeat', 1))
            tcmd = test.get('command')
            touts = test.get('outputs', [])
            invs = test.get('invariants', [])
            tworkdir = test.get('workdir', '{repo}')

            t_mapping = {'tool': tool, 'repo': repo, 'temp': td}
            cwd = tworkdir
            for k, v in t_mapping.items():
                cwd = cwd.replace('{' + k + '}', v)

            runs = []
            for _i in range(trepeat):
                # Clear file outputs from prior run (stdout/stderr pseudo outputs ignored)
                for op in touts:
                    if op in ('@stdout', '@stderr'):
                        continue
                    op2 = op
                    for k, v in t_mapping.items():
                        op2 = op2.replace('{' + k + '}', v)
                    if os.path.exists(op2):
                        try:
                            os.remove(op2)
                        except Exception:
                            pass

                argv = apply_placeholders(tcmd, t_mapping)
                r = run_cmd(argv, cwd=cwd)

                outs_digest_map = {}
                for op in touts:
                    if op == '@stdout':
                        outs_digest_map[op] = r.get('stdout_sha256')
                        continue
                    if op == '@stderr':
                        outs_digest_map[op] = r.get('stderr_sha256')
                        continue

                    op2 = op
                    for k, v in t_mapping.items():
                        op2 = op2.replace('{' + k + '}', v)
                    if os.path.exists(op2) and os.path.isfile(op2):
                        outs_digest_map[op] = sha256_file(op2)
                    else:
                        outs_digest_map[op] = None

                inv_ok, inv_results = eval_invariants(invs, t_mapping, run_record=r)
                r.update({'outputs_sha256': outs_digest_map, 'invariants_ok': inv_ok, 'invariants': inv_results})
                runs.append(r)

            det_ok = True
            first = runs[0]['outputs_sha256'] if runs else {}
            for rr in runs:
                det_ok = det_ok and rr.get('invariants_ok', False)
                det_ok = det_ok and (rr.get('outputs_sha256') == first)
                det_ok = det_ok and (rr.get('exit_code') == 0)

            receipt['tests'].append({
                'name': tname,
                'kind': test.get('kind'),
                'repeat': trepeat,
                'command_template': tcmd,
                'outputs': touts,
                'runs': runs,
                'deterministic': det_ok,
            })

            all_tests_ok = all_tests_ok and det_ok

    receipt['accepted'] = bool(all_tests_ok)
    receipt['acceptance_reason'] = 'all_tests_passed' if receipt['accepted'] else 'tests_failed'

    dep_core = {
        'contract_name': receipt.get('contract_name'),
        'tool': receipt.get('tool'),
        'identity': receipt.get('identity'),
        'attestations': receipt.get('inputs_presented', {}).get('attestations', []),
        'platform': receipt.get('platform'),
    }
    receipt['dependency_id'] = sha256_bytes(canonical_json_bytes(dep_core))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'wb') as f:
        f.write(canonical_json_bytes(receipt))

    if args.require and not receipt['accepted']:
        raise SystemExit(3)


if __name__ == '__main__':
    main()
