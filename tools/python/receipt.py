#!/usr/bin/env python3
"""Emit a stub receipt JSON for a binary path."""
import argparse, json


def main() -> None:
    """Parse arguments and write a stub receipt file."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-bin")
    parser.add_argument("--out-receipt")
    args, _ = parser.parse_known_args()
    
    # Generate dummy Receipt
    print(f"Signing receipt for {args.in_bin}...")
    with open(args.out_receipt, "w") as f:
        json.dump({"kind": "receipt", "signature": "valid_stub"}, f, indent=2)

if __name__ == "__main__":
    main()
