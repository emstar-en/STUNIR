#!/usr/bin/env python3
import json, os, sys, hashlib, platform
from pathlib import Path

# Args: binary receipt status build_epoch [epoch_json] [exception_reason]
if len(sys.argv) < 5:
    print("Usage: record_receipt.py <binary_path> <receipt_path> <status> <build_epoch> [epoch_json|-] [exception_reason]", file=sys.stderr)
    sys.exit(2)

binary_path = Path(sys.argv[1])
receipt_path = Path(sys.argv[2])
status = sys.argv[3]
build_epoch = int(sys.argv[4]) if len(sys.argv) > 4 else 0
epoch_json = Path(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] != '-' else None
exception_reason = sys.argv[6] if len(sys.argv) > 6 else None

sha256 = None
if binary_path.exists() and binary_path.is_file():
    h = hashlib.sha256()
    with open(binary_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    sha256 = h.hexdigest()

epoch_manifest = None
if epoch_json and epoch_json.exists():
    try:
        with open(epoch_json, 'r', encoding='utf-8') as ef:
            epoch_manifest = json.load(ef)
    except Exception:
        epoch_manifest = None

receipt = {
    'target': binary_path.name,
    'status': status,
    'build_epoch': build_epoch,
    'sha256': sha256,
    'epoch': epoch_manifest or {'selected_epoch': build_epoch, 'source': 'UNKNOWN', 'inputs': {}},
    'platform': {
        'system': platform.system(),
        'release': platform.release(),
        'version': platform.version(),
        'machine': platform.machine(),
        'python': platform.python_version(),
    },
}
if exception_reason:
    receipt['epoch_exception'] = True
    receipt['exception_reason'] = exception_reason

os.makedirs(receipt_path.parent, exist_ok=True)
with open(receipt_path, 'w', encoding='utf-8') as f:
    json.dump(receipt, f, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
print(f"Wrote {receipt_path}")
