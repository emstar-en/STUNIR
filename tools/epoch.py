#!/usr/bin/env python3
import argparse, json, os, subprocess, time

def to_int(v):
    try:
        return int(str(v).strip())
    except Exception:
        return None

parser = argparse.ArgumentParser()
parser.add_argument('--out-json', help='Write full epoch manifest to this path')
parser.add_argument('--print-epoch', action='store_true', help='Print only the selected epoch to stdout')
args = parser.parse_args()

inputs = {
    'STUNIR_BUILD_EPOCH': to_int(os.environ.get('STUNIR_BUILD_EPOCH')),
    'SOURCE_DATE_EPOCH': to_int(os.environ.get('SOURCE_DATE_EPOCH')),
    'GIT_COMMIT_EPOCH': None,
}

# Try to compute GIT_COMMIT_EPOCH if in a repo
try:
    out = subprocess.run(['git','log','-1','--format=%ct'], capture_output=True, text=True, check=False)
    val = to_int(out.stdout.strip())
    inputs['GIT_COMMIT_EPOCH'] = val
except Exception:
    inputs['GIT_COMMIT_EPOCH'] = None

selected_epoch = None
source = None
for key in ('STUNIR_BUILD_EPOCH','SOURCE_DATE_EPOCH','GIT_COMMIT_EPOCH'):
    if inputs.get(key) is not None:
        selected_epoch = inputs[key]
        source = key
        break

if selected_epoch is None:
    selected_epoch = int(time.time())
    source = 'CURRENT_TIME'

manifest = {
    'selected_epoch': selected_epoch,
    'source': source,
    'inputs': inputs,
}

if args.out_json:
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, sort_keys=True, separators=(',',':'), ensure_ascii=False)

if args.print_epoch:
    print(selected_epoch)
