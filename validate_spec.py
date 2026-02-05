import json

with open('stunir_runs/ardupilot_full/spec.json', 'r') as f:
    data = json.load(f)

funcs = data['modules'][0]['functions']
print(f'Total functions: {len(funcs)}')

for f in funcs:
    params = ', '.join([p['type'] + ' ' + p['name'] for p in f['parameters']])
    print(f"  {f['return_type']} {f['name']}({params})")