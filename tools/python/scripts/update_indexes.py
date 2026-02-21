import os
import json
from datetime import datetime

# Create index for tools/spark/src
src_dir = 'STUNIR-main/tools/spark/src'
files = []

for root, dirs, filenames in os.walk(src_dir):
    for f in filenames:
        full_path = os.path.join(root, f)
        rel_path = os.path.relpath(full_path, src_dir)
        files.append({
            'path': rel_path.replace('\\', '/'),
            'size': os.path.getsize(full_path)
        })

index = {
    'kind': 'stunir.code_index.v1',
    'root_path': 'tools/spark/src',
    'generated_at': datetime.now().isoformat(),
    'file_count': len(files),
    'files': sorted(files, key=lambda x: x['path'])
}

with open('STUNIR-main/tools/spark/index.machine.json', 'w') as f:
    json.dump(index, f, indent=2)

print(f'Generated index with {len(files)} files')