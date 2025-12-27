#!/usr/bin/env python3
import argparse, json, time
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json")
    ap.add_argument("--set-epoch")
    ap.add_argument("--print-epoch", action="store_true")
    args = ap.parse_args()

    epoch = args.set_epoch or "0"

    if args.print_epoch:
        print(epoch)

    if args.out_json:
        p = Path(args.out_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"selected_epoch": epoch}, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
