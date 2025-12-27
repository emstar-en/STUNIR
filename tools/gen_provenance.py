#!/usr/bin/env python3
import argparse, json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch")
    ap.add_argument("--spec-root")
    ap.add_argument("--asm-root")
    ap.add_argument("--out-header")
    ap.add_argument("--out-json")
    args = ap.parse_args()

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps({"provenance": "mock"}), encoding="utf-8")

    if args.out_header:
        Path(args.out_header).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_header).write_text("#define PROVENANCE_MOCK 1", encoding="utf-8")

    print("Generated Provenance")

if __name__ == "__main__":
    main()
