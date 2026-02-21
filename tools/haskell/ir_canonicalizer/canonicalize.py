#!/usr/bin/env python3
import json
import sys
import hashlib

# STUNIR Profile 3 Canonicalizer (Python Fallback)
# Implements JCS (RFC 8785) subset for STUNIR IR

def canonicalize(data):
    # 1. Sort keys
    # 2. No whitespace
    # 3. UTF-8 encoding
    return json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=False)

def main():
    if len(sys.argv) < 2:
        print("Usage: canonicalize.py <input.json> [output.json]")
        sys.exit(1)

    input_path = sys.argv[1]

    try:
        with open(input_path, 'r') as f:
            data = json.load(f)

        # Enforce Schema for IR if possible, or just canonicalize generic JSON
        # For STUNIR, we might want to ensure specific fields exist

        canonical_str = canonicalize(data)
        canonical_bytes = canonical_str.encode('utf-8')
        digest = hashlib.sha256(canonical_bytes).hexdigest()

        # Output
        if len(sys.argv) >= 3:
            output_path = sys.argv[2]
            with open(output_path, 'wb') as f:
                f.write(canonical_bytes)
            print(f"Canonicalized to {output_path}")
        else:
            print(canonical_str)

        print(f"SHA256: {digest}", file=sys.stderr)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
