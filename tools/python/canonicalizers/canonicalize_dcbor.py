#!/usr/bin/env python3
"""STUNIR dCBOR Canonicalizer - Deterministic CBOR output.

This tool is part of the tools → canonicalizers pipeline stage.
It produces deterministic CBOR output suitable for hashing.

Usage:
    canonicalize_dcbor.py <input.json> [output.dcbor]
"""

import json
import sys
import os

# Import the dCBOR encoder from tools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from dcbor import encode, FloatPolicy
    DCBOR_AVAILABLE = True
except ImportError:
    DCBOR_AVAILABLE = False

import hashlib

def compute_sha256(data):
    """Compute SHA-256 hash of bytes."""
    return hashlib.sha256(data).hexdigest()

def json_to_dcbor(data):
    """Convert JSON data to dCBOR bytes."""
    if DCBOR_AVAILABLE:
        return encode(data, float_policy=FloatPolicy.FLOAT64_FIXED)
    else:
        # Fallback: use canonical JSON as bytes
        canonical = json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
        return canonical.encode('utf-8')

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input.json> [output.dcbor]", file=sys.stderr)
        print("\nSTUNIR dCBOR Canonicalizer - Deterministic CBOR output.", file=sys.stderr)
        if not DCBOR_AVAILABLE:
            print("Note: dCBOR encoder not available, using JSON fallback.", file=sys.stderr)
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) >= 3 else None
    
    try:
        # Read input
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        # Convert to dCBOR
        dcbor_bytes = json_to_dcbor(data)
        digest = compute_sha256(dcbor_bytes)
        
        # Write output
        if output_path:
            with open(output_path, 'wb') as f:
                f.write(dcbor_bytes)
            print(f"dCBOR written to {output_path}", file=sys.stderr)
        else:
            # Hex output for terminal
            print(dcbor_bytes.hex())
        
        print(f"SHA256: {digest}", file=sys.stderr)
        print(f"Size: {len(dcbor_bytes)} bytes", file=sys.stderr)
        
    except json.JSONDecodeError as e:
        print(f"JSON Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
