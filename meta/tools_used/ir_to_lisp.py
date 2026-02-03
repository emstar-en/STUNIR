#!/usr/bin/env python3
# STUNIR: minimal deterministic Common Lisp codegen (portable + sbcl variant)
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()
def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + '\n'
def write_text(p: Path, s: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding='utf-8', newline='\n')
def main():
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
