#!/usr/bin/env python3
import argparse, hashlib, json
from pathlib import Path

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def canon(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec-root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    spec_root = Path(a.spec_root)
    spec_path = spec_root / "spec.json"
    spec_bytes = spec_path.read_bytes()
    spec_sha256 = sha256_bytes(spec_bytes)
    spec = json.loads(spec_bytes.decode("utf-8"))

    module_name = (spec.get("module_name") or spec.get("name") or "stunir_module") if isinstance(spec, dict) else "stunir_module"

    ir = {
        "ir_version": "v1",
        "module_name": module_name,
        "types": [],
        "functions": [],
        "spec_sha256": spec_sha256,
        "source": {
            "spec_sha256": spec_sha256,
            "spec_path": str(spec_path).replace("\\", "/"),
        },
    }

    if isinstance(spec, dict) and "modules" in spec:
        ir["source_modules"] = spec["modules"]

    Path(a.out).write_text(canon(ir), encoding="utf-8")

if __name__ == "__main__":
    main()
