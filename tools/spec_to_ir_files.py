#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from pathlib import Path

# When run as: python3 tools/spec_to_ir_files.py
# sys.path[0] == 'tools', so sibling import works.
import dcbor


def _load_epoch(epoch_json_path: str | None):
    if not epoch_json_path:
        return None
    if epoch_json_path == '-':
        return None
    if os.path.isfile(epoch_json_path):
        try:
            with open(epoch_json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _resolve_float_policy(arg: str | None) -> dcbor.FloatPolicy:
    if arg:
        return dcbor.FloatPolicy.parse(arg)
    env = os.environ.get('STUNIR_CBOR_FLOAT_POLICY')
    if env:
        return dcbor.FloatPolicy.parse(env)
    return dcbor.FloatPolicy.FLOAT64_FIXED


def _encode_spec_bytes_to_dcbor(b: bytes, *, float_policy: dcbor.FloatPolicy) -> tuple[bytes, bool]:
    """Return (dcbor_bytes, parsed_json_ok).

    If the input is not valid JSON, we still produce deterministic bytes by
    encoding the raw bytes as a CBOR byte-string.
    """
    try:
        obj = json.loads(b.decode('utf-8'))
        return (dcbor.dumps(obj, float_policy=float_policy), True)
    except Exception:
        return (dcbor.dumps(b, float_policy=float_policy), False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--spec-root', default='spec')
    p.add_argument('--out-root', default='asm/ir')
    p.add_argument('--epoch-json', default=None)
    p.add_argument('--manifest-out', default='receipts/ir_manifest.json')
    p.add_argument(
        '--float-policy',
        default=None,
        choices=[pp.value for pp in dcbor.FloatPolicy],
        help='Float encoding policy; overrides STUNIR_CBOR_FLOAT_POLICY when set.',
    )
    # Optional: emit a single bundle file + offset index for simplicity.
    p.add_argument('--bundle-out', default=None)
    p.add_argument('--bundle-manifest-out', default=None)
    args = p.parse_args()

    float_policy = _resolve_float_policy(args.float_policy)

    spec_root = Path(args.spec_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    epoch = _load_epoch(args.epoch_json)

    src_files = sorted([pp for pp in spec_root.rglob('*.json') if pp.is_file()])

    manifest = {
        'schema': 'stunir.ir_manifest.v2',
        'ir_format': 'dcbor',
        'dcbor': {
            'float_policy': float_policy.value,
        },
        'files': [],
        'epoch': epoch,
    }

    # Bundle support
    bundle_entries = []
    bundle_bytes_parts: list[bytes] = []
    bundle_offset = 0

    for src in src_files:
        rel_src = src.relative_to(spec_root)
        # Output path: replace .json -> .dcbor for clarity.
        rel_out = rel_src.with_suffix('.dcbor')
        dst = out_root / rel_out
        dst.parent.mkdir(parents=True, exist_ok=True)

        raw = src.read_bytes()
        ir_bytes, parsed_ok = _encode_spec_bytes_to_dcbor(raw, float_policy=float_policy)
        dst.write_bytes(ir_bytes)

        sha = hashlib.sha256(ir_bytes).hexdigest()
        rec = {
            'file': rel_out.as_posix(),
            'source_file': rel_src.as_posix(),
            'sha256': sha,
            'json_parse': bool(parsed_ok),
            'bytes_len': len(ir_bytes),
        }
        manifest['files'].append(rec)

        if args.bundle_out:
            bundle_entries.append({
                'file': rel_out.as_posix(),
                'source_file': rel_src.as_posix(),
                'offset': bundle_offset,
                'length': len(ir_bytes),
                'sha256': sha,
                'json_parse': bool(parsed_ok),
            })
            bundle_bytes_parts.append(ir_bytes)
            bundle_offset += len(ir_bytes)

    manifest['files'].sort(key=lambda x: x['file'])

    man_path = Path(args.manifest_out)
    man_path.parent.mkdir(parents=True, exist_ok=True)
    man_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(',', ':'), ensure_ascii=False) + '\n',
        encoding='utf-8',
    )

    if args.bundle_out:
        if not args.bundle_manifest_out:
            raise SystemExit('--bundle-out requires --bundle-manifest-out')

        bundle_path = Path(args.bundle_out)
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        bundle_data = b''.join(bundle_bytes_parts)
        bundle_path.write_bytes(bundle_data)
        bundle_sha = hashlib.sha256(bundle_data).hexdigest()

        bundle_manifest = {
            'schema': 'stunir.ir_bundle_manifest.v1',
            'bundle': str(bundle_path.as_posix()),
            'bundle_format': 'concat_raw',
            'bundle_sha256': bundle_sha,
            'epoch': epoch,
            'dcbor': {
                'float_policy': float_policy.value,
            },
            'entries': sorted(bundle_entries, key=lambda x: x['file']),
        }

        bm_path = Path(args.bundle_manifest_out)
        bm_path.parent.mkdir(parents=True, exist_ok=True)
        bm_path.write_text(
            json.dumps(bundle_manifest, sort_keys=True, separators=(',', ':'), ensure_ascii=False) + '\n',
            encoding='utf-8',
        )

    print(f"Wrote IR files to {out_root} and manifest {man_path}")
    if args.bundle_out:
        print(f"Wrote IR bundle {args.bundle_out} and bundle manifest {args.bundle_manifest_out}")


if __name__ == '__main__':
    main()
