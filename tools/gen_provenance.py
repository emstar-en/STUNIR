#!/usr/bin/env python3
import argparse
import json
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--epoch-source", default="UNKNOWN")
    parser.add_argument("--spec-root")
    parser.add_argument("--asm-root")
    parser.add_argument("--out-header")
    parser.add_argument("--out-json")
    args = parser.parse_args()

    # Output canonical JSON with required fields
    if args.out_json:
        data = {
            "build_epoch": args.epoch,
            "epoch_source": args.epoch_source
        }
        # Use canonical formatting (no spaces, sorted keys)
        with open(args.out_json, "w") as f:
            json.dump(data, f, separators=(',', ':'), sort_keys=True)
    
    if args.out_header:
        with open(args.out_header, "w") as f:
            f.write(f"#define STUNIR_EPOCH {args.epoch}\n")

if __name__ == "__main__":
    main()
