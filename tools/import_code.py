#!/usr/bin/env python3
# STUNIR: Import Code Tool
# Scans inputs/ and creates spec/imported/ specs.
from __future__ import annotations
import argparse, json, os
from pathlib import Path

def _w(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8", newline="\n")

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", required=True)
    ap.add_argument("--out-root", required=True)
    args = ap.parse_args()

    in_root = Path(args.input_root)
    out_root = Path(args.out_root)

    if not in_root.exists():
        print(f"Input root {in_root} does not exist. Skipping.")
        return 0

    print(f"Scanning {in_root}...")

    # Map extensions to languages
    ext_map = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".go": "go", ".rs": "rust", ".c": "c", ".cpp": "cpp",
        ".java": "java", ".rb": "ruby", ".php": "php",
        ".sh": "bash"
    }

    count = 0
    for root, dirs, files in os.walk(in_root):
        for f in files:
            p = Path(root) / f
            ext = p.suffix.lower()
            if ext in ext_map:
                lang = ext_map[ext]
                rel_path = p.relative_to(in_root)
                safe_name = str(rel_path).replace("/", "_").replace("\\", "_").replace(".", "_")

                content = p.read_text(encoding="utf-8", errors="replace")

                spec = {
                    "kind": "stunir.spec.v1",
                    "meta": {"origin": str(rel_path), "lang": lang},
                    "modules": [{"name": p.stem, "code": content, "lang": lang}]
                }

                out_path = out_root / f"{safe_name}.json"
                _w(out_path, json.dumps(spec, indent=2))
                count += 1

    print(f"Imported {count} files to {out_root}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
