#!/usr/bin/env python3
"""STUNIR JSON Canonicalizer - RFC 8785 / JCS subset implementation.

This tool is part of the tools → canonicalizers pipeline stage.
It produces deterministic JSON output suitable for hashing.

Usage:
    canonicalize_json.py <input.json> [output.json]
"""

import json
import sys
import hashlib

def canonical_json(data):
    """Generate canonical JSON output (RFC 8785 subset).
    
    Rules:
    1. Keys are sorted alphabetically (Unicode code point order)
    2. No whitespace between tokens
    3. No trailing newline
    4. UTF-8 encoded
    5. Numbers: no leading zeros, no trailing zeros in fractions
    6. Strings: minimal escape sequences
    """
    return json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=False)

def compute_sha256(data):
    """Compute SHA-256 hash of bytes."""
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.sha256(data).hexdigest()

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input.json> [output.json]", file=sys.stderr)
        print("\nSTUNIR JSON Canonicalizer - RFC 8785 / JCS subset.", file=sys.stderr)
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) >= 3 else None
    
    try:
        # Read input
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        # Canonicalize
        canonical_output = canonical_json(data)
        canonical_bytes = canonical_output.encode('utf-8')
        digest = compute_sha256(canonical_bytes)
        
        # Write output
        if output_path:
            with open(output_path, 'wb') as f:
                f.write(canonical_bytes)
            print(f"Canonicalized to {output_path}", file=sys.stderr)
        else:
            print(canonical_output)
        
        print(f"SHA256: {digest}", file=sys.stderr)
        
    except json.JSONDecodeError as e:
        print(f"JSON Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
