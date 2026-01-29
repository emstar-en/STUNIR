#!/usr/bin/env python3
"""STUNIR ASM Verifier - Verifies assembly-level artifacts.

This tool is part of the tools → asm pipeline stage.
It verifies ASM artifacts against their manifest.

Usage:
    asm_verify.py <asm.json> [--manifest=<manifest.json>]
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

def verify_asm_artifact(asm_data, manifest_data=None):
    """Verify ASM artifact.
    
    Returns:
        tuple: (is_valid, errors, warnings)
    """
    errors = []
    warnings = []
    
    # Check schema
    schema = asm_data.get('schema', '')
    if not schema:
        errors.append("Missing 'schema' field")
    elif not schema.startswith('stunir.asm.'):
        errors.append(f"Invalid schema: {schema}")
    
    # Check module name
    module = asm_data.get('asm_module', '')
    if not module:
        errors.append("Missing 'asm_module' field")
    
    # Check IR hash
    ir_hash = asm_data.get('asm_ir_hash', '')
    if not ir_hash:
        warnings.append("Missing 'asm_ir_hash' field")
    
    # Verify against manifest if provided
    if manifest_data:
        manifest_entries = manifest_data.get('entries', [])
        found = False
        for entry in manifest_entries:
            if entry.get('module') == module:
                found = True
                if entry.get('hash') != ir_hash:
                    errors.append(f"Hash mismatch for module {module}")
                break
        if not found:
            warnings.append(f"Module {module} not found in manifest")
    
    is_valid = len(errors) == 0
    return is_valid, errors, warnings

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <asm.json> [--manifest=<manifest.json>]", file=sys.stderr)
        print("\nSTUNIR ASM Verifier - Verifies assembly-level artifacts.", file=sys.stderr)
        sys.exit(1)
    
    input_path = sys.argv[1]
    manifest_path = None
    
    for arg in sys.argv[2:]:
        if arg.startswith('--manifest='):
            manifest_path = arg.split('=', 1)[1]
    
    try:
        # Read ASM file
        with open(input_path, 'r') as f:
            asm_data = json.load(f)
        
        # Read manifest if provided
        manifest_data = None
        if manifest_path:
            with open(manifest_path, 'r') as f:
                manifest_data = json.load(f)
        
        # Verify
        is_valid, errors, warnings = verify_asm_artifact(asm_data, manifest_data)
        
        # Output results
        print(f"File: {input_path}")
        print(f"Schema: {asm_data.get('schema', 'unknown')}")
        print(f"Module: {asm_data.get('asm_module', 'unknown')}")
        print(f"Valid: {is_valid}")
        
        if warnings:
            print("\nWarnings:")
            for w in warnings:
                print(f"  ⚠ {w}")
        
        if errors:
            print("\nErrors:")
            for e in errors:
                print(f"  ✗ {e}")
            sys.exit(1)
        
        print("\n✓ ASM verification passed")
        
    except json.JSONDecodeError as e:
        print(f"JSON Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
