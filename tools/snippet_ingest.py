#!/usr/bin/env python3
"""Convert code snippets into STUNIR spec JSON files."""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path

def main() -> int:
    """Parse CLI arguments and emit a STUNIR spec JSON for a snippet."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input code file")
    ap.add_argument("--lang", required=True, help="Language of input")
    ap.add_argument("--output", required=True, help="Output spec JSON")
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file {input_path} not found")
        return 1

    content = input_path.read_text(encoding="utf-8")

    # Create a minimal STUNIR Spec wrapping the snippet
    spec = {
        "kind": "stunir.spec.v1",
        "meta": {
            "origin": "snippet_ingest",
            "source_lang": args.lang
        },
        "modules": [
            {
                "name": "main",
                "code": content,
                "lang": args.lang
            }
        ]
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Canonical JSON dump
    out_path.write_text(
        json.dumps(spec, sort_keys=True, separators=(',', ':'), ensure_ascii=False) + "\n",
        encoding="utf-8"
    )

    print(f"Converted {args.input} ({args.lang}) -> {args.output}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
