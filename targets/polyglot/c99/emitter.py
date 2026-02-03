#!/usr/bin/env python3
"""STUNIR C99 Target Emitter - Emit ISO C99 code from IR.

This tool is part of the targets → polyglot → c99 pipeline stage.
It converts STUNIR IR to C99-compliant code.

Usage:
    emitter.py <ir.json> --output=<dir>
    emitter.py --help
"""

import sys
import json
from pathlib import Path

# Add parent directory to path for c_base import
sys.path.insert(0, str(Path(__file__).parent.parent))
from c_base import CEmitterBase, canonical_json


class C99Emitter(CEmitterBase):
    """C99 (ISO C99) code emitter."""
    
    DIALECT = 'c99'


def parse_args(argv):
    """Parse command line arguments."""
    args = {'output': None, 'input': None}
    for arg in argv[1:]:
        if arg.startswith('--output='):
            args['output'] = arg.split('=', 1)[1]
        elif arg == '--help':
            print(__doc__)
            sys.exit(0)
        elif not arg.startswith('--'):
            args['input'] = arg
    return args


def main():
    args = parse_args(sys.argv)
    if not args['input']:
        print(f"Usage: {sys.argv[0]} <ir.json> --output=<dir>", file=sys.stderr)
        sys.exit(1)
    
    out_dir = args['output'] or 'c99_output'
    
    try:
        with open(args['input'], 'r') as f:
            ir_data = json.load(f)
        
        emitter = C99Emitter(ir_data, out_dir)
        emitter.emit()
        
        manifest = emitter.emit_manifest()
        manifest_path = Path(out_dir) / 'manifest.json'
        manifest_path.write_text(canonical_json(manifest), encoding='utf-8')
        
        print(f"C99 code emitted to {out_dir}/", file=sys.stderr)
        print(f"Files: {len(emitter.generated_files)}", file=sys.stderr)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
