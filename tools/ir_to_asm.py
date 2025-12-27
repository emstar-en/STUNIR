#!/usr/bin/env python3
# STUNIR: minimal deterministic Assembly codegen
from __future__ import annotations
import argparse, json
from pathlib import Path

def _w(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8", newline="\n")

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--ir-manifest", required=True)
    ap.add_argument("--out-root", required=True)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Generate Main Code
    code = """section .data
    msg db "Hello from Deterministic Assembly!", 0xA
    len equ $ - msg

section .text
    global _start

_start:
    mov edx, len
    mov ecx, msg
    mov ebx, 1
    mov eax, 4
    int 0x80

    mov ebx, 0
    mov eax, 1
    int 0x80
"""
    _w(out_root / "program.asm", code)

    _w(out_root / "README.md", "Assembly output (minimal backend).\n")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
