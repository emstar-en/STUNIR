#!/usr/bin/env python3
"""Convert extraction.json to IR format"""
import json

# Read extraction.json
with open('extraction.json', 'r') as f:
    extraction = json.load(f)

# Convert to IR format
functions = []
for func in extraction['functions']:
    args = []
    for param in func['parameters']:
        args.append({
            'name': param['name'],
            'type': param['type']
        })
    functions.append({
        'name': func['name'],
        'args': args,
        'return_type': func['return_type'],
        'steps': [{'op': 'noop'}]
    })

ir = {
    'schema': 'stunir_flat_ir_v1',
    'ir_version': 'v1',
    'module_name': 'bc_batch_01',
    'docstring': '',
    'types': [],
    'functions': functions
}

with open('ir.json', 'w') as f:
    json.dump(ir, f, separators=(',', ':'))

print('Created ir.json with', len(functions), 'functions')
for func in functions:
    print(f"  {func['name']}: {func['args']}")
