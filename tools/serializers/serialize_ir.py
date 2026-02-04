#!/usr/bin/env python3
"""STUNIR IR Serializer - Serializes IR to various formats.

This tool is part of the tools → serializers pipeline stage.
It converts IR to canonical JSON or dCBOR format.

Usage:
    serialize_ir.py <ir.json> [--format=json|dcbor] [--output=<file>]
"""

import json
import sys
import hashlib
import os
from typing import Any, Dict, Optional

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dcbor import encode, FloatPolicy
    DCBOR_AVAILABLE = True
except ImportError:
    DCBOR_AVAILABLE = False

def canonical_json(data: Any) -> str:
    """Generate canonical JSON output."""
    return json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=False)

def compute_sha256(data: Any) -> str:
    """Compute SHA-256 hash."""
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.sha256(data).hexdigest()

def serialize_to_json(ir_data: Any) -> bytes:
    """Serialize IR to canonical JSON bytes."""
    return canonical_json(ir_data).encode('utf-8')

def serialize_to_dcbor(ir_data: Any) -> bytes:
    """Serialize IR to dCBOR bytes."""
    if DCBOR_AVAILABLE:
        return encode(ir_data, float_policy=FloatPolicy.FLOAT64_FIXED)
    else:
        # Fallback to canonical JSON
        return serialize_to_json(ir_data)

def parse_args(argv: list[str]) -> Dict[str, Optional[str]]:
    """Parse command line arguments."""
    args = {'format': 'json', 'output': None, 'input': None}

    for arg in argv[1:]:
        if arg.startswith('--format='):
            args['format'] = arg.split('=', 1)[1]
        elif arg.startswith('--output='):
            args['output'] = arg.split('=', 1)[1]
        elif not arg.startswith('--'):
            args['input'] = arg

    return args

def main() -> None:
    args = parse_args(sys.argv)
    
    if not args['input']:
        print(f"Usage: {sys.argv[0]} <ir.json> [--format=json|dcbor] [--output=<file>]", file=sys.stderr)
        print("\nSTUNIR IR Serializer - Serializes IR to various formats.", file=sys.stderr)
        if not DCBOR_AVAILABLE:
            print("Note: dCBOR not available, will fallback to JSON.", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Read IR file
        with open(args['input'], 'r') as f:
            ir_data = json.load(f)
        
        # Serialize based on format
        if args['format'] == 'dcbor':
            serialized = serialize_to_dcbor(ir_data)
            ext = '.dcbor'
        else:
            serialized = serialize_to_json(ir_data)
            ext = '.json'
        
        # Compute hash
        content_hash = compute_sha256(serialized)
        
        # Output
        if args['output']:
            with open(args['output'], 'wb') as f:
                f.write(serialized)
            print(f"Serialized to {args['output']}", file=sys.stderr)
        else:
            if args['format'] == 'dcbor':
                print(serialized.hex())
            else:
                sys.stdout.buffer.write(serialized)
                print()  # newline
        
        print(f"Format: {args['format']}", file=sys.stderr)
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
