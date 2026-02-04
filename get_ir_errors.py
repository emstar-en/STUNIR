#!/usr/bin/env python3
"""Extract IR format errors from analysis report"""
import json

with open('analysis_report.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print("=" * 60)
print("IR FORMAT ERRORS (pass13_ir_format)")
print("=" * 60)

ir_errors = data['findings'].get('pass13_ir_format', [])
print(f"\nTotal IR errors: {len(ir_errors)}\n")

for i, error in enumerate(ir_errors, 1):
    print(f"Error #{i}")
    print(f"  File: {error.get('file', 'unknown')}")
    print(f"  Line: {error.get('line', 'N/A')}")
    print(f"  Severity: {error.get('severity', 'unknown')}")
    print(f"  Message: {error.get('message', 'N/A')}")
    if error.get('metadata'):
        print(f"  Metadata: {error['metadata']}")
    print("-" * 60)
