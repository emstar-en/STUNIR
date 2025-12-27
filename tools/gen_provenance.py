#!/usr/bin/env python3
import argparse
import json
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch")
    parser.add_argument("--epoch-source")
    parser.add_argument("--spec-root")
    parser.add_argument("--asm-root")
    parser.add_argument("--out-header")
    parser.add_argument("--out-json")
    args = parser.parse_args()

    # Output canonical empty JSON for now (Shell Mode compatibility)
    # This matches what dispatch.sh produces.
    if args.out_json:
        with open(args.out_json, "w") as f:
            f.write('{}') 
    
    if args.out_header:
        with open(args.out_header, "w") as f:
            f.write("")

if __name__ == "__main__":
    main()
