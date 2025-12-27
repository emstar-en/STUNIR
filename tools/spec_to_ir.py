#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def canonical_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spec-root", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    spec_root = Path(args.spec_root)
    spec_path = spec_root / "spec.json"
    spec_bytes = spec_path.read_bytes()
    spec_sha256 = sha256_bytes(spec_bytes)

    spec = json.loads(spec_bytes.decode("utf-8"))

    module_name = spec.get("module_name") or spec.get("name") or "stunir_module"

    ir = {
        "ir_version": "v1",
        "module_name": module_name,
        "types": [],
        "functions": [],
        # Cryptographic binding to the exact bytes of build/spec.json
        "spec_sha256": spec_sha256,
    }

    # Optional: carry the imported module list forward for traceability
    if isinstance(spec, dict) and "modules" in spec:
        ir["source_modules"] = spec["modules"]

    Path(args.out).write_text(canonical_json(ir), encoding="utf-8")

if __name__ == "__main__":
    main()
