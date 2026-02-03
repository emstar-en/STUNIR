#!/usr/bin/env python3
import argparse, json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target")
    ap.add_argument("--receipt")
    ap.add_argument("--status")
    ap.add_argument("--build-epoch")
    ap.add_argument("--epoch-json")
    args = ap.parse_args()

    if args.receipt:
        Path(args.receipt).parent.mkdir(parents=True, exist_ok=True)
        Path(args.receipt).write_text(json.dumps({
            "target": args.target,
            "status": args.status,
            "epoch": args.build_epoch
        }, indent=2), encoding="utf-8")

    print(f"Recorded Receipt for {args.target}")

if __name__ == "__main__":
    main()
