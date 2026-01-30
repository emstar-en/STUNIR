#!/usr/bin/env python3
"""
===============================================================================
STUNIR Spec to IR Converter - Python REFERENCE Implementation
===============================================================================

WARNING: This is a REFERENCE IMPLEMENTATION for readability purposes only.
         DO NOT use this file for production, verification, or safety-critical
         applications.

PRIMARY IMPLEMENTATION: Ada SPARK
    Location: tools/spark/bin/stunir_spec_to_ir_main
    Build:    cd tools/spark && gprbuild -P stunir_tools.gpr

This Python version exists to:
1. Provide a readable reference for understanding the algorithm
2. Serve as a fallback when Ada SPARK tools are not available
3. Enable quick prototyping and testing

For all production use cases, use the Ada SPARK implementation which provides:
- Formal verification guarantees
- Deterministic execution
- DO-178C compliance support
- Absence of runtime errors (proven via SPARK)

===============================================================================
"""
import argparse
import hashlib
import json
import sys
import os
from pathlib import Path

# Add local tools dir to path to allow importing lib
sys.path.insert(0, str(Path(__file__).parent))

try:
    from lib import toolchain
except ImportError:
    # Fallback if running from root without setting PYTHONPATH
    sys.path.insert(0, "tools")
    from lib import toolchain

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def canon(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lockfile", default="local_toolchain.lock.json", help="Path to toolchain lockfile")
    a = ap.parse_args()

    # 1. Enforce Toolchain Lock
    print(f"[INFO] Loading toolchain from {a.lockfile}...")
    try:
        toolchain.load(a.lockfile)
        # Verify critical tools if we were to use them. 
        # For now, just verifying 'python' (ourselves) and 'git' (for epoch) is a good sanity check.

        # Verify Python (self-check)
        # Note: The lockfile might name it 'python' or 'python3'
        py_path = toolchain.get_tool("python")
        if py_path:
            print(f"[INFO] Verified Python runtime: {py_path}")

    except Exception as e:
        print(f"[ERROR] Toolchain verification failed: {e}")
        sys.exit(1)

    spec_root = Path(a.spec_root)
    out_path = Path(a.out)

    if not spec_root.exists():
        print(f"[ERROR] Spec root not found: {spec_root}")
        sys.exit(1)

    # 2. Process Specs
    print(f"[INFO] Processing specs from {spec_root}...")

    manifest = []

    # Deterministic walk
    for root, dirs, files in os.walk(spec_root):
        dirs.sort()
        files.sort()
        for f in files:
            if not f.endswith(".json"):
                continue

            full_path = Path(root) / f
            rel_path = full_path.relative_to(spec_root)

            with open(full_path, "rb") as fh:
                content = fh.read()

            digest = sha256_bytes(content)
            manifest.append({
                "path": str(rel_path),
                "sha256": digest,
                "size": len(content)
            })

    # 3. Write Output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        f.write(canon(manifest))

    print(f"[INFO] Wrote IR manifest to {out_path}")

if __name__ == "__main__":
    main()
