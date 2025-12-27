#!/usr/bin/env python3
import argparse
import json
import hashlib
import sys
from pathlib import Path

def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()

def canonical_json_bytes(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    spec_root = Path(args.spec_root)
    spec_path = spec_root / "spec.json"
    
    if not spec_path.exists():
        sys.exit(f"Spec not found at {spec_path}")

    # Read the Spec
    with open(spec_path, "rb") as f:
        spec_content = f.read()
        spec_data = json.loads(spec_content)

    # Calculate Spec Hash for binding
    spec_hash = sha256_bytes(spec_content)

    # Construct Valid IR (Bootstrap Mode)
    # We create a minimal valid IR that represents "The project defined by this spec"
    ir = {
        "ir_version": "v1",
        "module_name": "stunir_bootstrap",
        "docstring": "Bootstrap IR generated from source manifest.",
        "types": [],
        "functions": [],
        "source": {
            "spec_sha256": spec_hash,
            "spec_logical_path": "spec.json"
        },
        "determinism": {
            "requires_utf8": True,
            "requires_lf_newlines": True,
            "requires_stable_ordering": True
        }
    }

    # Calculate Self-Hash (Canonical)
    ir_bytes = canonical_json_bytes(ir)
    ir_hash = sha256_bytes(ir_bytes)
    
    # Inject self-hash into source.ir_canonical_sha256 (Optional but good practice)
    ir["source"]["ir_canonical_sha256"] = ir_hash

    # Output Canonical IR
    with open(args.out, "wb") as f:
        f.write(canonical_json_bytes(ir))
        f.write(b"\n") # Trailing newline for shell compatibility

if __name__ == "__main__":
    main()
