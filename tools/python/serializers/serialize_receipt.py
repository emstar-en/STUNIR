#!/usr/bin/env python3
"""STUNIR Receipt Serializer - Serializes receipts to canonical format.

This tool is part of the tools → serializers pipeline stage.
It converts receipts to canonical JSON format.

Usage:
    serialize_receipt.py <receipt.json> [--output=<file>]
"""

import json
import sys
import hashlib
from typing import Any

def canonical_json(data: Any) -> str:
    """Generate canonical JSON output."""
    return json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=False)

def compute_sha256(data: Any) -> str:
    """Compute SHA-256 hash."""
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.sha256(data).hexdigest()

def serialize_receipt(receipt_data: Any) -> bytes:
    """Serialize receipt to canonical JSON bytes."""
    return canonical_json(receipt_data).encode('utf-8')

def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <receipt.json> [--output=<file>]", file=sys.stderr)
        print("\nSTUNIR Receipt Serializer - Serializes receipts to canonical format.", file=sys.stderr)
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = None
    
    for arg in sys.argv[2:]:
        if arg.startswith('--output='):
            output_path = arg.split('=', 1)[1]
    
    try:
        # Read receipt file
        with open(input_path, 'r') as f:
            receipt_data = json.load(f)
        
        # Serialize
        serialized = serialize_receipt(receipt_data)
        content_hash = compute_sha256(serialized)
        
        # Output
        if output_path:
            with open(output_path, 'wb') as f:
                f.write(serialized)
            print(f"Serialized to {output_path}", file=sys.stderr)
        else:
            sys.stdout.buffer.write(serialized)
            print()  # newline
        
        print(f"Size: {len(serialized)} bytes", file=sys.stderr)
        print(f"SHA256: {content_hash}", file=sys.stderr)
        
    except json.JSONDecodeError as e:
        print(f"JSON Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
