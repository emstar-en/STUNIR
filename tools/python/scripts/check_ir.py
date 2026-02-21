import json

with open('stunir_runs/ardupilot_full/ir.json', 'r') as f:
    content = f.read()

print('File size:', len(content))
print('First 200 chars:', repr(content[:200]))
print('Has schema:', 'schema' in content)
print('Schema value:', json.loads(content).get('schema'))