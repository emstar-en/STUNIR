#!/usr/bin/env python3
import argparse, hashlib, json, os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec-root', default='spec')
    parser.add_argument('--out', required=True)
    parser.add_argument('--epoch-json', default=None)
    args = parser.parse_args()

    spec_root = Path(args.spec_root)
    # Sort files for deterministic order
    files = sorted([p for p in spec_root.rglob('*.json') if p.is_file()])

    entries = []
    for p in files:
        b = p.read_bytes()
        sha = hashlib.sha256(b).hexdigest()
        meta = {'file': p.relative_to(spec_root).as_posix(), 'sha256': sha, 'id': None, 'name': None}

        try:
            obj = json.loads(b.decode('utf-8', errors='ignore'))
            if isinstance(obj, dict):
                # Extract identity
                meta['id'] = obj.get('id')
                # Fallback chain for name: name -> title -> original_filename (for blobs)
                meta['name'] = obj.get('name') or obj.get('title')

                # Extra robustness for blobs if 'name' wasn't hoisted (backward compatibility)
                if meta['name'] is None and obj.get('kind') == 'stunir.blob':
                     meta['name'] = obj.get('metadata', {}).get('original_filename')

        except Exception:
            pass

        entries.append(meta)

    lines = []
    lines.append('; STUNIR IR SUMMARY v1')

    if args.epoch_json and os.path.isfile(args.epoch_json):
        try:
            with open(args.epoch_json, 'r', encoding='utf-8') as ef:
                ej = json.load(ef)
                lines.append(f"; epoch.selected={ej.get('selected_epoch')} source={ej.get('source')}")
        except Exception:
            lines.append('; epoch.selected=? source=?')
    else:
        lines.append('; epoch.selected=? source=?')

    for m in entries:
        ident = m['id'] if m['id'] is not None else '-'
        name = m['name'] if m['name'] is not None else '-'
        lines.append(f"FILE {m['file']} SHA256 {m['sha256']} ID {ident} NAME {name}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, 'w', encoding='utf-8') as outf:
        outf.write('\n'.join(lines) + '\n')

    print(f"Wrote IR {out_path} with {len(entries)} entries")

if __name__ == '__main__':
    main()
