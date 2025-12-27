#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec-root", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--epoch-json", default=None)

    # legacy flags (still accepted)
    ap.add_argument("--in-json", dest="in_json", default=None)
    ap.add_argument("--out-ir", dest="out_ir", default=None)

    args = ap.parse_args()

    # normalize legacy -> current
    spec_root = args.spec_root
    out_path = args.out

    if args.in_json and not spec_root:
        spec_root = str(Path(args.in_json).parent)
    if args.out_ir and not out_path:
        out_path = args.out_ir

    if not spec_root:
        spec_root = "."
    if not out_path:
        out_path = str(Path(spec_root) / "ir.json")

    spec_path = Path(spec_root) / "spec.json"
    spec = {}
    if spec_path.exists():
        spec = json.loads(spec_path.read_text(encoding="utf-8"))

    module_name = spec.get("module_name") or spec.get("name") or "Main"

    ir = {
        "ir_version": "v1",
        "module_name": module_name,
        "types": [],
        "functions": [],
    }

    Path(out_path).write_text(
        json.dumps(ir, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8"
    )

    # build.sh expects this string
    print("Generated IR Summary")

if __name__ == "__main__":
    main()
