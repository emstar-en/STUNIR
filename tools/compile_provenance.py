#!/usr/bin/env python3
# STUNIR: Compile Provenance Tool
# Compiles tools/prov_emit.c if available, or emits a skipped receipt.
from __future__ import annotations
import argparse, json, subprocess, shutil, sys
from pathlib import Path

def _w(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8", newline="\n")

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", required=True)
    ap.add_argument("--epoch-json", required=True)
    ap.add_argument("--provenance-json", required=True)
    ap.add_argument("--ir-manifest", required=True)
    ap.add_argument("--bundle-manifest", required=True)
    args = ap.parse_args()

    # Check for C compiler
    cc = shutil.which("gcc") or shutil.which("clang") or shutil.which("cc")

    src = Path("tools/prov_emit.c")
    bin_out = Path("bin/prov_emit")
    bin_out.parent.mkdir(parents=True, exist_ok=True)

    status = "SKIPPED_NO_COMPILER"

    if cc and src.exists():
        try:
            # Compile
            cmd = [cc, str(src), "-o", str(bin_out), "-O2"]
            subprocess.check_call(cmd)
            status = "BINARY_EMITTED"
        except subprocess.CalledProcessError:
            status = "COMPILATION_FAILED"
    elif not src.exists():
        status = "SKIPPED_NO_SOURCE"

    # Record Receipt (Mocking record_receipt logic for simplicity)
    receipt = {
        "kind": "stunir.receipt.v1",
        "target": str(bin_out),
        "status": status,
        "epoch": args.epoch
    }

    _w(Path("receipts/prov_emit.json"), json.dumps(receipt, indent=2))
    print(f"Compile Provenance: {status}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
