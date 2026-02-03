#!/usr/bin/env python3
import argparse, json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec-root")
    ap.add_argument("--out-root")
    ap.add_argument("--epoch-json")
    ap.add_argument("--manifest-out")
    ap.add_argument("--bundle-out")
    ap.add_argument("--bundle-manifest-out")
    args = ap.parse_args()

    # Mock IR Files
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Create a dummy IR file
    (out_root / "dummy.dcbor").write_bytes(b"dummy")

    # Manifests
    if args.manifest_out:
        Path(args.manifest_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.manifest_out).write_text(json.dumps({"files": []}), encoding="utf-8")

    if args.bundle_manifest_out:
        Path(args.bundle_manifest_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.bundle_manifest_out).write_text(json.dumps({"bundle": "mock"}), encoding="utf-8")

    print("Generated IR Files")

if __name__ == "__main__":
    main()
