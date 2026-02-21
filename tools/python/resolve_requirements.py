#!/usr/bin/env python3
"""Resolve required dependency contracts for requested output targets."""

import argparse
import json
import os
from typing import Any, Dict, List


def load_json(path: str) -> Any:
    """Load JSON from a file path."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def dump_canon(obj: Any) -> bytes:
    """Serialize an object to canonical JSON bytes."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')


def main() -> None:
    """Resolve target requirements and write a requirements JSON file."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', action='append', default=[])
    ap.add_argument('--map', dest='map_path', default='contracts/target_requirements.json')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    map_path = args.map_path
    if not os.path.isabs(map_path):
        map_path = os.path.join(repo, map_path)

    m = load_json(map_path)
    aliases = m.get('target_aliases', {})
    targets_def = m.get('targets', {})

    raw = []
    for t in args.targets:
        if not t:
            continue
        raw.extend([x.strip() for x in t.split(',') if x.strip()])

    normalized = [aliases.get(t, t) for t in raw]

    unknown = [t for t in normalized if t not in targets_def]
    if unknown:
        out = {
            'requirements_type': 'backend_requirements',
            'requested_targets': raw,
            'normalized_targets': normalized,
            'unknown_targets': unknown,
            'required_contracts': [],
            'optional_contracts': [],
            'notes': ['Unknown targets present; no requirements emitted.'],
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'wb') as f:
            f.write(dump_canon(out))
        raise SystemExit(2)

    required = []
    optional = []
    notes = []
    for t in normalized:
        td = targets_def[t]
        for c in td.get('required_contracts', []):
            required.append({'contract_name': c, 'reason': f'required_by_target:{t}'})
        for c in td.get('optional_contracts', []):
            optional.append({'contract_name': c, 'reason': f'optional_for_target:{t}'})
        notes.extend(td.get('notes', []) or [])

    def dedupe(lst: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        out = []
        for x in lst:
            k = x.get('contract_name')
            if k in seen:
                continue
            seen.add(k)
            out.append(x)
        return out

    out = {
        'requirements_type': 'backend_requirements',
        'requested_targets': raw,
        'normalized_targets': normalized,
        'unknown_targets': [],
        'required_contracts': dedupe(required),
        'optional_contracts': dedupe(optional),
        'notes': [{'note': n} for n in dict.fromkeys(notes)],
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'wb') as f:
        f.write(dump_canon(out))


if __name__ == '__main__':
    main()