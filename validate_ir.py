import json

with open('stunir_runs/ardupilot_full/ir_fixed.json', 'r') as f:
    data = json.load(f)

funcs = data['functions']
print(f"Module: {data['module_name']}")
print(f"Total functions: {len(funcs)}")
print()

for f in funcs:
    args = ', '.join([p['type'] + ' ' + p['name'] for p in f['args']])
    print(f"  {f['return_type']} {f['name']}({args})")