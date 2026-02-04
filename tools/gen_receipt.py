#!/usr/bin/env python3
"""Generate build receipts for produced artifacts."""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def canonical_json_bytes(obj):
    """Serialize an object to canonical JSON bytes."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(b):
    """Compute a SHA-256 digest for bytes."""
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    """Compute a SHA-256 digest for a file path."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    """Write a canonical build receipt for a target artifact."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--toolchain-lock")
    parser.add_argument("--epoch-json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    target_path = Path(args.target)
    if not target_path.exists():
        # If target doesn't exist, we can't receipt it.
        sys.exit(f"Target not found: {target_path}")

    # 1. Basic Fields
    receipt = {
        "schema": "stunir.receipt.build.v1",
        "target": str(target_path),
        "status": "success",
        "sha256": sha256_file(target_path),
        "inputs": [], # Shell mode doesn't track inputs yet
        "argv": ["shell_generated"],
        "tool": None
    }

    # 2. Epoch
    if args.epoch_json:
        with open(args.epoch_json, "rb") as f:
            ej = json.load(f)
        receipt["epoch"] = ej
        receipt["build_epoch"] = ej.get("selected_epoch", 0)
    else:
        receipt["epoch"] = {"selected_epoch": 0, "source": "unknown"}
        receipt["build_epoch"] = 0

    # 3. Toolchain
    if args.toolchain_lock:
        tc_path = Path(args.toolchain_lock)
        if tc_path.exists():
            receipt["toolchain_sha256"] = sha256_file(tc_path)

    # 4. Core ID (The critical binding hash)
    core = {
        "schema": receipt["schema"],
        "target": receipt["target"],
        "status": receipt["status"],
        "build_epoch": receipt["build_epoch"],
        "sha256": receipt["sha256"],
        "epoch": receipt["epoch"],
        "inputs": receipt["inputs"],
        "tool": receipt["tool"],
        "argv": receipt["argv"],
    }
    receipt["receipt_core_id_sha256"] = sha256_bytes(canonical_json_bytes(core))

    # 5. Write
    with open(args.out, "wb") as f:
        f.write(canonical_json_bytes(receipt))
        f.write(b"\n")

if __name__ == "__main__":
    main()
