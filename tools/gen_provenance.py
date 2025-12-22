#!/usr/bin/env python3
import argparse, hashlib, json, os, sys, time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', required=True)
parser.add_argument('--epoch-source', default='UNKNOWN')
parser.add_argument('--spec-root', default='spec')
parser.add_argument('--asm-root', default='asm')
parser.add_argument('--out-header', required=True)
parser.add_argument('--out-json', required=True)
args = parser.parse_args()

def sha256_of_dir(root: Path) -> str:
    if not root.exists():
        return '0'*64
    h = hashlib.sha256()
    for p in sorted(root.rglob('*')):
        if p.is_file():
            h.update(p.relative_to(root).as_posix().encode())
            with open(p, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    h.update(chunk)
    return h.hexdigest()

spec_digest = sha256_of_dir(Path(args.spec_root))
asm_digest = sha256_of_dir(Path(args.asm_root))

# Canonicalize epoch to int
try:
    epoch = int(str(args.epoch).strip())
except Exception:
    epoch = 0

meta = {
    'build_epoch': epoch,
    'epoch_source': args.epoch_source,
    'spec_digest': spec_digest,
    'asm_digest': asm_digest,
    'provenance_version': 1
}

os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
with open(args.out_json, 'w', encoding='utf-8') as f:
    json.dump(meta, f, sort_keys=True, separators=(',', ':'), ensure_ascii=False)

# Emit a tiny header used by the runtime tool
header = f