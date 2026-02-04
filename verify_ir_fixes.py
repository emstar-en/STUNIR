#!/usr/bin/env python3
"""Verify IR schema fixes"""
import json
import glob

print("=" * 60)
print("VERIFYING IR SCHEMA FIXES")
print("=" * 60)

files = glob.glob('examples/semantic_ir/*.json')
all_valid = True

for filepath in files:
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        schema = data.get('schema', 'MISSING')
        expected = 'stunir_ir_v1'
        
        if schema == expected:
            print(f"✓ {filepath}: schema = '{schema}'")
        else:
            print(f"✗ {filepath}: schema = '{schema}' (expected '{expected}')")
            all_valid = False
    except Exception as e:
        print(f"✗ {filepath}: ERROR - {e}")
        all_valid = False

print("=" * 60)
if all_valid:
    print("✓ ALL IR FILES VALID - Phase 1 Complete!")
else:
    print("✗ SOME IR FILES INVALID")
print("=" * 60)
