#!/usr/bin/env python3
"""Generate minimal deterministic Common Lisp outputs from an IR manifest."""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
from typing import Any

def sha256_bytes(b: bytes) -> str:
    """Compute a SHA-256 digest for bytes."""
    return hashlib.sha256(b).hexdigest()

def canonical_json(obj: Any) -> str:
    """Serialize an object to canonical JSON with a trailing newline."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + '\n'

def write_text(p: Path, s: str) -> None:
    """Write text to a file path, creating parent directories as needed."""
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding='utf-8', newline='\n')

def main() -> None:
    """Emit Common Lisp scaffolding and payload from an IR manifest."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True)
    ap.add_argument('--ir-manifest', required=True)
    ap.add_argument('--out-root', required=True)
    args = ap.parse_args()
    out_root = Path(args.out_root)
    ir_manifest = json.loads(Path(args.ir_manifest).read_text(encoding='utf-8'))
    ir_sha = sha256_bytes(canonical_json(ir_manifest).encode('utf-8'))
    payload = {'kind':'stunir.sample_output','variant':args.variant,'ir_manifest_sha256':ir_sha,'ir_manifest':ir_manifest}
    write_text(out_root/'payload.json', canonical_json(payload))
    write_text(out_root/'package.lisp', '(defpackage :stunir.generated (:use :cl) (:export :main))\n')
    runtime = ''.join([
        '(in-package :stunir.generated)\n\n',
        '(defun slurp-file (path)\n',
        '  (with-open-file (in path :direction :input :external-format :utf-8)\n',
        '    (let ((out (make-string-output-stream)))\n',
        '      (loop for line = (read-line in nil nil) while line do\n',
        '            (write-string line out)\n',
        '            (write-char #\\Newline out))\n',
        '      (get-output-stream-string out))))\n',
        '(defun main ()\n',
        '  (write-string (slurp-file "payload.json"))\n',
        '  (write-char #\\Newline))\n',
    ])
    write_text(out_root/'runtime.lisp', runtime)
    program = ''.join(['(load "package.lisp")\n','(load "runtime.lisp")\n','(stunir.generated:main)\n'])
    write_text(out_root/'program.lisp', program)
    write_text(out_root/'README.md', 'Common Lisp output (minimal sample).\n')
if __name__ == '__main__':
    main()
