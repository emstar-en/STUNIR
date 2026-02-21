#!/usr/bin/env python3
"""STUNIR IR Emitter - Converts spec to deterministic IR format.

This tool is part of the tools → ir_emitter pipeline stage.
It reads a STUNIR spec file and emits canonical IR in JSON format.

Usage:
    emit_ir.py <spec.json> [output.json]

The output is deterministic (RFC 8785 / JCS subset):
- Keys are sorted alphabetically
- No unnecessary whitespace
- UTF-8 encoded
"""

import json
import sys
import hashlib
import time
import os
from typing import Any, Dict

# STUNIR IR Schema Version
IR_SCHEMA_VERSION = "stunir.ir.v1"

def canonical_json(data: Any) -> str:
    """Generate canonical JSON output (RFC 8785 subset)."""
    return json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=False)

def compute_sha256(data: Any) -> str:
    """Compute SHA-256 hash of bytes."""
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.sha256(data).hexdigest()

def spec_to_ir(spec_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert spec data to IR format."""
    # Extract module info from spec
    module_name = spec_data.get('module', spec_data.get('name', 'unnamed'))
    
    ir = {
        "schema": IR_SCHEMA_VERSION,
        "ir_module": module_name,
        "ir_epoch": int(time.time()),
        "ir_spec_hash": compute_sha256(canonical_json(spec_data)),
        "ir_functions": [],
        "ir_types": [],
        "ir_imports": [],
        "ir_exports": []
    }
    
    # Convert spec functions to IR functions
    functions = spec_data.get('functions', spec_data.get('ops', []))
    for i, func in enumerate(functions):
        if isinstance(func, dict):
            ir_func = {
                "name": func.get('name', f'func_{i}'),
                "params": func.get('params', func.get('inputs', [])),
                "returns": func.get('returns', func.get('output', 'void')),
                "body": func.get('body', [])
            }
        else:
            ir_func = {
                "name": f'func_{i}',
                "params": [],
                "returns": 'void',
                "body": [str(func)]
            }
        ir["ir_functions"].append(ir_func)
    
    # Convert spec types to IR types
    types = spec_data.get('types', [])
    for t in types:
        if isinstance(t, dict):
            ir["ir_types"].append(t)
        else:
            ir["ir_types"].append({"name": str(t)})
    
    # Handle imports/exports
    ir["ir_imports"] = spec_data.get('imports', [])
    ir["ir_exports"] = spec_data.get('exports', [module_name])

    return ir

def main() -> None:
    """Read a spec JSON file and emit canonical IR JSON."""
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <spec.json> [output.json]", file=sys.stderr)
        print("\nSTUNIR IR Emitter - Converts spec to deterministic IR format.", file=sys.stderr)
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) >= 3 else None
    
    try:
        # Read spec
        with open(input_path, 'r') as f:
            spec_data = json.load(f)
        
        # Convert to IR
        ir_data = spec_to_ir(spec_data)
        
        # Generate canonical output
        canonical_output = canonical_json(ir_data)
        canonical_bytes = canonical_output.encode('utf-8')
        ir_hash = compute_sha256(canonical_bytes)
        
        # Write output
        if output_path:
            with open(output_path, 'wb') as f:
                f.write(canonical_bytes)
            print(f"IR emitted to {output_path}", file=sys.stderr)
        else:
            print(canonical_output)
        
        print(f"IR SHA256: {ir_hash}", file=sys.stderr)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
