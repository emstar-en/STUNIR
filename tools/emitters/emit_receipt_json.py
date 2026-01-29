#!/usr/bin/env python3
"""STUNIR Receipt JSON Emitter - Emits receipts in JSON format.

This tool is part of the tools → emitters pipeline stage.
It generates receipt JSON with canonical formatting.

Usage:
    emit_receipt_json.py --target=<name> --status=<status> [options]
"""

import json
import sys
import time
import hashlib
import os

def canonical_json(data):
    """Generate canonical JSON output."""
    return json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=False)

def compute_sha256(data):
    """Compute SHA-256 hash."""
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.sha256(data).hexdigest()

def create_receipt(target, status, epoch=None, tool_name=None, tool_path=None, tool_hash=None, tool_version=None, inputs=None, argv=None):
    """Create a STUNIR receipt."""
    receipt = {
        "schema": "stunir.receipt.build.v1",
        "receipt_target": target,
        "receipt_status": status,
        "receipt_build_epoch": epoch or int(time.time()),
        "receipt_epoch_json": "build/epoch.json"
    }
    
    if inputs:
        receipt["receipt_inputs"] = inputs
    else:
        receipt["receipt_inputs"] = {}
    
    if tool_name:
        receipt["receipt_tool"] = {
            "name": tool_name,
            "path": tool_path or "",
            "hash": tool_hash or "",
            "version": tool_version or "1.0.0"
        }
    
    if argv:
        receipt["receipt_argv"] = argv
    
    return receipt

def parse_args(argv):
    """Parse command line arguments."""
    args = {
        'target': None,
        'status': 'success',
        'epoch': None,
        'tool_name': None,
        'tool_path': None,
        'tool_hash': None,
        'tool_version': None,
        'output': None
    }
    
    for arg in argv[1:]:
        if arg.startswith('--target='):
            args['target'] = arg.split('=', 1)[1]
        elif arg.startswith('--status='):
            args['status'] = arg.split('=', 1)[1]
        elif arg.startswith('--epoch='):
            args['epoch'] = int(arg.split('=', 1)[1])
        elif arg.startswith('--tool-name='):
            args['tool_name'] = arg.split('=', 1)[1]
        elif arg.startswith('--tool-path='):
            args['tool_path'] = arg.split('=', 1)[1]
        elif arg.startswith('--tool-hash='):
            args['tool_hash'] = arg.split('=', 1)[1]
        elif arg.startswith('--tool-version='):
            args['tool_version'] = arg.split('=', 1)[1]
        elif arg.startswith('--output='):
            args['output'] = arg.split('=', 1)[1]
    
    return args

def main():
    args = parse_args(sys.argv)
    
    if not args['target']:
        print(f"Usage: {sys.argv[0]} --target=<name> --status=<status> [options]", file=sys.stderr)
        print("\nOptions:", file=sys.stderr)
        print("  --target=<name>       Receipt target name", file=sys.stderr)
        print("  --status=<status>     Status (success/failure)", file=sys.stderr)
        print("  --epoch=<timestamp>   Build epoch", file=sys.stderr)
        print("  --tool-name=<name>    Tool name", file=sys.stderr)
        print("  --tool-path=<path>    Tool path", file=sys.stderr)
        print("  --tool-hash=<hash>    Tool hash", file=sys.stderr)
        print("  --tool-version=<ver>  Tool version", file=sys.stderr)
        print("  --output=<file>       Output file", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Create receipt
        receipt = create_receipt(
            target=args['target'],
            status=args['status'],
            epoch=args['epoch'],
            tool_name=args['tool_name'],
            tool_path=args['tool_path'],
            tool_hash=args['tool_hash'],
            tool_version=args['tool_version']
        )
        
        # Canonicalize
        canonical_output = canonical_json(receipt)
        
        # Output
        if args['output']:
            with open(args['output'], 'w') as f:
                f.write(canonical_output)
            print(f"Receipt emitted to {args['output']}", file=sys.stderr)
        else:
            print(canonical_output)
        
        print(f"Target: {args['target']}", file=sys.stderr)
        print(f"Status: {args['status']}", file=sys.stderr)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
