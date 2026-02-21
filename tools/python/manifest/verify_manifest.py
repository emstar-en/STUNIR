#!/usr/bin/env python3
"""STUNIR Toolchain Manifest Verifier.

This tool is part of the tools → manifest pipeline stage.
It verifies toolchain manifests against actual tools.

Usage:
    verify_manifest.py <manifest.json> [--tools-dir=<dir>]
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

def compute_file_hash(filepath):
    """Compute SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def verify_manifest(manifest_data, tools_dir=None):
    """Verify manifest against actual tools.
    
    Returns:
        tuple: (is_valid, errors, warnings, stats)
    """
    errors = []
    warnings = []
    stats = {'verified': 0, 'missing': 0, 'hash_mismatch': 0}
    
    # Check schema
    schema = manifest_data.get('schema', '')
    if not schema:
        errors.append("Missing 'schema' field")
    elif not schema.startswith('stunir.manifest.'):
        warnings.append(f"Unexpected schema: {schema}")
    
    # Verify manifest hash
    manifest_copy = dict(manifest_data)
    stored_hash = manifest_copy.pop('manifest_hash', None)
    computed_hash = compute_sha256(canonical_json(manifest_copy))
    
    if stored_hash and stored_hash != computed_hash:
        errors.append(f"Manifest hash mismatch: stored={stored_hash}, computed={computed_hash}")
    
    # Verify tools if directory provided
    if tools_dir and os.path.isdir(tools_dir):
        tools = manifest_data.get('manifest_tools', [])
        
        for tool in tools:
            tool_path = os.path.join(tools_dir, tool.get('path', ''))
            
            if not os.path.exists(tool_path):
                errors.append(f"Tool not found: {tool.get('name')} at {tool.get('path')}")
                stats['missing'] += 1
                continue
            
            # Verify hash
            actual_hash = compute_file_hash(tool_path)
            expected_hash = tool.get('hash', '')
            
            if expected_hash and actual_hash != expected_hash:
                errors.append(f"Hash mismatch for {tool.get('name')}: expected={expected_hash}, actual={actual_hash}")
                stats['hash_mismatch'] += 1
            else:
                stats['verified'] += 1
    
    is_valid = len(errors) == 0
    return is_valid, errors, warnings, stats

def main():
    """Validate a toolchain manifest from the command line."""
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <manifest.json> [--tools-dir=<dir>]", file=sys.stderr)
        print("\nSTUNIR Toolchain Manifest Verifier.", file=sys.stderr)
        sys.exit(1)
    
    input_path = sys.argv[1]
    tools_dir = None
    
    for arg in sys.argv[2:]:
        if arg.startswith('--tools-dir='):
            tools_dir = arg.split('=', 1)[1]
    
    try:
        # Read manifest
        with open(input_path, 'r') as f:
            manifest_data = json.load(f)
        
        # Verify
        is_valid, errors, warnings, stats = verify_manifest(manifest_data, tools_dir)
        
        # Output results
        print(f"Manifest: {input_path}")
        print(f"Schema: {manifest_data.get('schema', 'unknown')}")
        print(f"Tools in manifest: {manifest_data.get('manifest_count', 0)}")
        
        if tools_dir:
            print(f"\nVerification against {tools_dir}:")
            print(f"  Verified: {stats['verified']}")
            print(f"  Missing: {stats['missing']}")
            print(f"  Hash mismatch: {stats['hash_mismatch']}")
        
        print(f"\nValid: {is_valid}")
        
        if warnings:
            print("\nWarnings:")
            for w in warnings:
                print(f"  ⚠ {w}")
        
        if errors:
            print("\nErrors:")
            for e in errors:
                print(f"  ✗ {e}")
            sys.exit(1)
        
        print("\n✓ Manifest verification passed")
        
    except json.JSONDecodeError as e:
        print(f"JSON Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
