#!/usr/bin/env python3
import json
import sys
import time
import os

# STUNIR Receipt Emitter (Python Fallback)

def main():
    # Usage: emit_receipt.py <target> <status> <epoch> <tool_name> <tool_path> <tool_hash> <tool_ver> [args...]
    if len(sys.argv) < 8:
        print("Usage: emit_receipt.py <target> <status> <epoch> <tool_name> <tool_path> <tool_hash> <tool_ver> [args...]")
        sys.exit(1)

    target = sys.argv[1]
    status = sys.argv[2]
    epoch = int(sys.argv[3])
    tool_name = sys.argv[4]
    tool_path = sys.argv[5]
    tool_hash = sys.argv[6]
    tool_ver = sys.argv[7]
    argv = sys.argv[8:]

    receipt = {
        "schema": "stunir.receipt.build.v1",
        "receipt_target": target,
        "receipt_status": status,
        "receipt_build_epoch": epoch,
        "receipt_epoch_json": "build/epoch.json", # Default
        "receipt_inputs": {}, # Placeholder
        "receipt_tool": {
            "name": tool_name,
            "path": tool_path,
            "hash": tool_hash,
            "version": tool_ver
        },
        "receipt_argv": argv
    }

    # Canonicalize output
    print(json.dumps(receipt, sort_keys=True, separators=(',', ':'), ensure_ascii=False))

if __name__ == "__main__":
    main()
