#!/usr/bin/env python3
import argparse, json, os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-root', default='inputs')
    parser.add_argument('--out-root', default='spec/imported')
    args = parser.parse_args()

    in_root = Path(args.input_root)
    out_root = Path(args.out_root)

    if not in_root.exists():
        # No inputs, nothing to do
        return

    # Ensure output directory exists
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {in_root} for source code...")

    # Walk through inputs
    count = 0
    for p in in_root.rglob('*'):
        if p.is_file() and not p.name.startswith('.'):
            try:
                # Try reading as text (UTF-8)
                # Future: Add binary support (base64) if needed
                content = p.read_text(encoding='utf-8')

                # Create STUNIR Blob Spec
                spec = {
                    "kind": "stunir.blob",
                    "metadata": {
                        "original_filename": p.name,
                        "relative_path": p.relative_to(in_root).as_posix(),
                        "extension": p.suffix,
                        "ingest_strategy": "literal"
                    },
                    "content": content
                }

                # Generate output path: spec/imported/<rel_path>.json
                # We replace / with _ to flatten, or keep structure?
                # Keeping structure is better for organization, but flattening avoids deep nesting issues.
                # Let's mirror the structure inside spec/imported/

                rel_path = p.relative_to(in_root)
                out_path = out_root / rel_path.parent / (p.name + ".json")
                out_path.parent.mkdir(parents=True, exist_ok=True)

                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(spec, f, indent=2, sort_keys=True)
                    f.write('\n')

                print(f"  [+] Ingested: {rel_path} -> {out_path}")
                count += 1

            except UnicodeDecodeError:
                print(f"  [!] Skipping binary file: {p}")
            except Exception as e:
                print(f"  [!] Error processing {p}: {e}")

    if count == 0:
        print("No valid text files found in inputs/.")
    else:
        print(f"Ingestion complete. {count} files wrapped as specs.")

if __name__ == '__main__':
    main()
