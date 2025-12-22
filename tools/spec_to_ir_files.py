#!/usr/bin/env python3
import argparse, hashlib, json, os
from pathlib import Path

def normalize_json_bytes(b):
    try:
        obj = json.loads(b.decode('utf-8'))
        return json.dumps(obj, sort_keys=True, separators=(',',':'), ensure_ascii=False).encode('utf-8') + b"\n"
    except Exception:
        return b

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--spec-root', default='spec')
    p.add_argument('--out-root', default='asm/ir')
    p.add_argument('--epoch-json', default=None)
    p.add_argument('--manifest-out', default='receipts/ir_manifest.json')
    args = p.parse_args()

    spec_root = Path(args.spec_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in spec_root.rglob('*.json') if p.is_file()])
    manifest = { 'files': [], 'epoch': None }
    if args.epoch_json and os.path.isfile(args.epoch_json):
        try:
            with open(args.epoch_json, 'r', encoding='utf-8') as f:
                manifest['epoch'] = json.load(f)
        except Exception:
            manifest['epoch'] = None

    for src in files:
        rel = src.relative_to(spec_root)
        dst = out_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        data = src.read_bytes()
        norm = normalize_json_bytes(data)
        dst.write_bytes(norm)
        sha = hashlib.sha256(norm).hexdigest()
        manifest['files'].append({'file': rel.as_posix(), 'sha256': sha})

    manifest['files'].sort(key=lambda x: x['file'])
    man_path = Path(args.manifest_out)
    man_path.parent.mkdir(parents=True, exist_ok=True)
    with open(man_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, sort_keys=True, separators=(',',':'), ensure_ascii=False)
    print(f"Wrote IR files to {out_root} and manifest {man_path}")

if __name__ == '__main__':
    main()
