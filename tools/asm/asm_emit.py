#!/usr/bin/env python3
"""STUNIR ASM Emitter - Emits assembly-level artifacts.

This tool is part of the tools → asm pipeline stage.
It generates ASM/IR bundle artifacts.

Usage:
    asm_emit.py <ir.json> [--output-dir=<dir>]
"""

import json
import sys
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

def emit_asm_artifact(ir_data, output_dir):
    """Emit ASM artifact from IR data."""
    module_name = ir_data.get('ir_module', 'module')
    
    # Create ASM bundle
    asm_bundle = {
        "schema": "stunir.asm.v1",
        "asm_module": module_name,
        "asm_ir_hash": compute_sha256(canonical_json(ir_data)),
        "asm_instructions": [],
        "asm_labels": [],
        "asm_data": []
    }
    
    # Convert IR functions to ASM-like representation
    functions = ir_data.get('ir_functions', [])
    for func in functions:
        func_name = func.get('name', 'unnamed')
        # Add label
        asm_bundle['asm_labels'].append({
            "name": func_name,
            "offset": len(asm_bundle['asm_instructions'])
        })
        # Add pseudo-instructions
        asm_bundle['asm_instructions'].append({
            "op": "FUNC_BEGIN",
            "label": func_name
        })
        for stmt in func.get('body', []):
            asm_bundle['asm_instructions'].append({
                "op": "STMT",
                "data": str(stmt)
            })
        asm_bundle['asm_instructions'].append({
            "op": "FUNC_END",
            "label": func_name
        })
    
    # Write output
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{module_name}.asm.json")
    
    canonical_output = canonical_json(asm_bundle)
    with open(output_path, 'w') as f:
        f.write(canonical_output)
    
    return output_path, compute_sha256(canonical_output.encode('utf-8'))

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <ir.json> [--output-dir=<dir>]", file=sys.stderr)
        print("\nSTUNIR ASM Emitter - Emits assembly-level artifacts.", file=sys.stderr)
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_dir = 'asm/ir'
    
    for arg in sys.argv[2:]:
        if arg.startswith('--output-dir='):
            output_dir = arg.split('=', 1)[1]
    
    try:
        # Read IR file
        with open(input_path, 'r') as f:
            ir_data = json.load(f)
        
        # Emit ASM artifact
        output_path, content_hash = emit_asm_artifact(ir_data, output_dir)
        
        print(f"ASM artifact emitted to {output_path}")
        print(f"SHA256: {content_hash}")
        
    except json.JSONDecodeError as e:
        print(f"JSON Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
