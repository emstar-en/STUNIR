#!/usr/bin/env python3
import argparse
import json
import hashlib
import sys
from pathlib import Path

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--out-spec", required=True)
    args = parser.parse_args()

    root = Path(args.input_root)
    modules = []
    
    if root.exists():
        # Deterministic walk: Sort by path to ensure stable order
        for path in sorted(root.rglob("*")):
            if path.is_file():
                # Calculate relative path (e.g., "subdir/file.txt")
                rel_path = path.relative_to(root).as_posix()
                
                # Hash the content
                file_hash = sha256_file(path)
                
                modules.append({
                    "path": rel_path,
                    "sha256": file_hash
                })

    # Construct the Spec Object
    spec = {
        "kind": "spec",
        "modules": modules
    }

    # Output Canonical JSON
    with open(args.out_spec, "w") as f:
        json.dump(spec, f, separators=(',', ':'), sort_keys=True)
        f.write('\n') # Trailing newline for shell compatibility

if __name__ == "__main__":
    main()
