#!/usr/bin/env python3
# STUNIR: minimal deterministic Ruby codegen
from __future__ import annotations
import argparse, json
from pathlib import Path

def _w(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8", newline="\n")

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--ir-manifest", required=True)
    ap.add_argument("--out-root", required=True)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Generate Main Code
    code = """puts "STUNIR Generated Ruby Artifact"
puts "Hello from Deterministic Ruby!"
"""
    _w(out_root / "program.rb", code)

    _w(out_root / "README.md", "Ruby output (minimal backend).\n")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
