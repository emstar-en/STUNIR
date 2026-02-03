#!/usr/bin/env python3
"""Small helper to extract the accepted tool path from a dependency acceptance receipt."""

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--receipt', required=True)
    ap.add_argument('--require-accepted', action='store_true')
    ap.add_argument('--print-path', action='store_true')
    args = ap.parse_args()

    p = Path(args.receipt)
    if not p.exists():
        return 2

    obj = json.loads(p.read_text(encoding='utf-8'))
    if obj.get('receipt_type') != 'dependency_acceptance':
        return 2

    accepted = bool(obj.get('accepted', False))
    if args.require_accepted and not accepted:
        return 3

    tool = obj.get('tool') or {}
    path = tool.get('resolved_abs') or tool.get('resolved')
    if args.print_path:
        if isinstance(path, str) and path:
            print(path)
            return 0
        return 4

    return 0 if accepted else 3


if __name__ == '__main__':
    raise SystemExit(main())
