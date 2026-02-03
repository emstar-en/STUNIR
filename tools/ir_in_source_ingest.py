#!/usr/bin/env python3
"""Tier A ingest: IR-in-source.

Extract a canonical IR bundle byte payload from source files using STUNIR marker
records, without parsing the host language.

Outputs:
- --out-bundle: exact bytes of the resolved IR bundle
- --out-json: canonical JSON metadata (stunir-json-c14n-v1 style)
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

HEX_RE = re.compile(r"^[0-9a-f]{64}$")


def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(2)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canon_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')


def safe_relpath(p: str) -> Path:
    if not isinstance(p, str) or not p:
        die("path/uri must be non-empty string")
    pp = Path(p)
    if pp.is_absolute():
        die(f"absolute paths forbidden in uri: {p}")
    norm = Path(os.path.normpath(p))
    norm_s = str(norm).replace('\\\\', '/')
    if norm_s == '..' or norm_s.startswith('../') or '/..' in norm_s:
        die(f"path traversal forbidden in uri: {p}")
    return norm


def strip_comment_wrappers(line: str, comment_styles: List[str]) -> str:
    s = line.lstrip()
    mpos = s.find('STUNIR_IR_')
    if mpos != -1:
        return s[mpos:]

    for tok in comment_styles:
        t = tok.strip()
        if t in ('//', '#', '%', '--') and s.startswith(t):
            return s[len(t):].lstrip()

    if s.startswith('/*'):
        s2 = s[2:]
        if s2.endswith('*/'):
            s2 = s2[:-2]
        return s2.strip()

    if s.startswith('{-'):
        s2 = s[2:]
        if s2.endswith('-}'):
            s2 = s2[:-2]
        return s2.strip()

    return s


def parse_markers_matrix(path: Path) -> Dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding='utf-8'))
    except Exception as e:
        die(f"failed to read markers matrix JSON {path}: {e}")
    if not isinstance(obj, dict):
        die("markers matrix must be a JSON object")
    return obj


def extract_record_from_lines(lines: List[str]) -> Dict[str, Any]:
    sha_lines: List[str] = []
    ref_uri: Optional[str] = None

    in_b64 = False
    saw_begin = False
    saw_end = False
    b64_lines: List[str] = []

    for raw in lines:
        line = raw.rstrip('\n')

        if line.startswith('STUNIR_IR_REF uri='):
            if ref_uri is not None:
                die('multiple STUNIR_IR_REF records found in a single file')
            ref_uri = line[len('STUNIR_IR_REF uri='):].strip()
            if not ref_uri:
                die('empty uri in STUNIR_IR_REF')
            continue

        if line.startswith('STUNIR_IR_SHA256 '):
            sha = line[len('STUNIR_IR_SHA256 '):].strip().lower()
            if not HEX_RE.match(sha):
                die(f"invalid STUNIR_IR_SHA256 (expected 64 hex chars): {sha!r}")
            sha_lines.append(sha)
            continue

        if line == 'STUNIR_IR_B64URL_BEGIN':
            if in_b64 or saw_begin:
                die('multiple STUNIR_IR_B64URL_BEGIN markers found')
            in_b64 = True
            saw_begin = True
            b64_lines = []
            continue

        if line == 'STUNIR_IR_B64URL_END':
            if not in_b64:
                die('STUNIR_IR_B64URL_END without STUNIR_IR_B64URL_BEGIN')
            in_b64 = False
            saw_end = True
            continue

        if in_b64:
            b64_lines.append(line)

    out: Dict[str, Any] = {}

    # Embedded record present iff BEGIN/END present
    if saw_begin or saw_end:
        if not (saw_begin and saw_end):
            die('embedded record incomplete: missing BEGIN or END')
        if not sha_lines:
            die('embedded record requires STUNIR_IR_SHA256 before B64URL block')
        embedded_sha = sha_lines[0]
        payload = ''.join(b64_lines)
        payload = re.sub(r"\s+", "", payload)
        if '=' in payload:
            die("embedded base64url must be unpadded ('=' forbidden)")
        try:
            pad = '=' * ((4 - (len(payload) % 4)) % 4)
            decoded = base64.urlsafe_b64decode(payload + pad)
        except Exception as e:
            die(f"failed to decode embedded base64url payload: {e}")
        got = sha256_bytes(decoded)
        if got != embedded_sha:
            die(f"embedded payload sha256 mismatch: expected {embedded_sha}, got {got}")
        out['embedded'] = {'sha256': embedded_sha, 'bytes': decoded}

    # Reference record present iff REF present
    if ref_uri is not None:
        if not sha_lines:
            die('reference record requires STUNIR_IR_SHA256 ...')
        ref_sha = sha_lines[0]
        out['reference'] = {'uri': ref_uri, 'sha256': ref_sha}

    # If both embedded and reference present, they must agree
    if 'embedded' in out and 'reference' in out:
        if out['embedded']['sha256'] != out['reference']['sha256']:
            die('embedded and reference records disagree (sha256 differs)')

    return out


def ingest_from_source_file(path: Path, markers_matrix: Dict[str, Any]) -> Dict[str, Any]:
    comment_styles: List[str] = []
    try:
        for _lang, ld in (markers_matrix.get('languages') or {}).items():
            for cs in (ld.get('comment_styles') or []):
                if isinstance(cs, str) and cs not in comment_styles:
                    comment_styles.append(cs)
    except Exception:
        comment_styles = ['//', '#', '%', '--', '/* */', '{- -}']

    try:
        raw_lines = path.read_text(encoding='utf-8', errors='replace').splitlines(True)
    except Exception as e:
        die(f"failed to read source file {path}: {e}")

    marker_lines: List[str] = []
    for ln in raw_lines:
        stripped = strip_comment_wrappers(ln, comment_styles).rstrip('\n')
        if 'STUNIR_IR_' in stripped:
            pos = stripped.find('STUNIR_IR_')
            marker_lines.append(stripped[pos:])

    recs = extract_record_from_lines(marker_lines)
    return {
        'source_path': path.as_posix(),
        'source_sha256': sha256_file(path),
        'recs': recs,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo-root', default='.')
    ap.add_argument('--markers-matrix', required=True)
    ap.add_argument('--sources', default='')
    ap.add_argument('--ir-bundle', default='')
    ap.add_argument('--ir-bundle-sha256', default='')
    ap.add_argument('--out-json', required=True)
    ap.add_argument('--out-bundle', required=True)
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()

    mm_path = Path(args.markers_matrix)
    if not mm_path.is_absolute():
        mm_path = (repo_root / mm_path).resolve()
    if not mm_path.exists():
        die(f"markers matrix not found: {mm_path}")
    markers_matrix = parse_markers_matrix(mm_path)
    mm_sha = sha256_file(mm_path)

    sources: List[str] = []
    if args.sources:
        sources = [s.strip() for s in args.sources.split(',') if s.strip()]

    direct_ir = args.ir_bundle.strip()
    if not sources and not direct_ir:
        die('must provide --sources or --ir-bundle')

    selected_mode: Optional[str] = None
    resolved_bytes: Optional[bytes] = None
    resolved_sha: Optional[str] = None
    resolved_ref_uri: Optional[str] = None

    source_meta: List[Dict[str, Any]] = []

    # Direct ir-bundle input
    if direct_ir:
        ir_path = Path(direct_ir)
        if not ir_path.is_absolute():
            ir_path = (repo_root / ir_path).resolve()
        if not ir_path.exists():
            die(f"--ir-bundle not found: {ir_path}")
        b = ir_path.read_bytes()
        got = sha256_bytes(b)
        exp = args.ir_bundle_sha256.strip().lower()
        if exp:
            if not HEX_RE.match(exp):
                die('--ir-bundle-sha256 must be 64 hex chars')
            if got != exp:
                die(f"direct ir bundle sha256 mismatch: expected {exp}, got {got}")
        selected_mode = 'direct'
        resolved_bytes = b
        resolved_sha = got

    # Source markers
    if sources:
        for sp in sources:
            p = Path(sp)
            if not p.is_absolute():
                p = (repo_root / p).resolve()
            if not p.exists():
                die(f"source file not found: {sp}")
            rep = ingest_from_source_file(p, markers_matrix)
            recs = rep['recs']

            # Resolve bytes for this source (embedded preferred)
            mode = None
            b = None
            if 'embedded' in recs:
                mode = 'embedded'
                b = recs['embedded']['bytes']
            elif 'reference' in recs:
                mode = 'reference'
                uri = recs['reference']['uri']
                uri_p = safe_relpath(uri)
                ref_path = (repo_root / uri_p).resolve()
                if not ref_path.exists():
                    die(f"referenced IR bundle not found: {uri} (resolved {ref_path})")
                ref_bytes = ref_path.read_bytes()
                got = sha256_bytes(ref_bytes)
                exp = recs['reference']['sha256'].lower()
                if got != exp:
                    die(f"reference IR bundle sha256 mismatch for {uri}: expected {exp}, got {got}")
                b = ref_bytes
                resolved_ref_uri = uri
            else:
                die(f"no STUNIR_IR_* record found in source file: {rep['source_path']}")

            bsha = sha256_bytes(b)
            if resolved_sha is None:
                resolved_bytes = b
                resolved_sha = bsha
                selected_mode = mode
            else:
                if resolved_sha != bsha:
                    die('conflicting IR bundle payloads across sources (sha256 differs)')
                if mode == 'embedded' and selected_mode != 'embedded':
                    selected_mode = 'embedded'

            source_meta.append({
                'path': os.path.relpath(Path(rep['source_path']), repo_root).replace('\\\\','/'),
                'sha256': rep['source_sha256'],
                'has_embedded': ('embedded' in recs),
                'has_reference': ('reference' in recs),
                'reference_uri': recs.get('reference', {}).get('uri') if 'reference' in recs else None,
                'reference_sha256': recs.get('reference', {}).get('sha256') if 'reference' in recs else None,
                'embedded_sha256': recs.get('embedded', {}).get('sha256') if 'embedded' in recs else None,
            })

    if resolved_bytes is None or resolved_sha is None or selected_mode is None:
        die('failed to resolve IR bundle bytes')

    out_bundle = Path(args.out_bundle)
    if not out_bundle.is_absolute():
        out_bundle = (repo_root / out_bundle).resolve()
    out_bundle.parent.mkdir(parents=True, exist_ok=True)
    out_bundle.write_bytes(resolved_bytes)

    out_json = Path(args.out_json)
    if not out_json.is_absolute():
        out_json = (repo_root / out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)

    meta: Dict[str, Any] = {
        'schema': 'stunir.ir_in_source_ingest.v0',
        'markers_matrix': {
            'path': os.path.relpath(mm_path, repo_root).replace('\\\\','/'),
            'sha256': mm_sha,
            'doc_type': markers_matrix.get('doc_type'),
        },
        'inputs': {
            'sources': source_meta,
            'direct_ir_bundle': {
                'path': args.ir_bundle if args.ir_bundle else None,
                'expected_sha256': args.ir_bundle_sha256.lower() if args.ir_bundle_sha256 else None,
            },
        },
        'result': {
            'selected_mode': selected_mode,
            'bundle': {
                'path': os.path.relpath(out_bundle, repo_root).replace('\\\\','/'),
                'sha256': resolved_sha,
                'length': len(resolved_bytes),
            },
            'reference_uri': resolved_ref_uri,
        },
    }

    out_json.write_bytes(canon_bytes(meta) + b"\n")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
