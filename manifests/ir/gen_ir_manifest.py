#!/usr/bin/env python3
"""STUNIR IR Manifest Generator.

Part of manifests → ir pipeline stage (Issue #1015).
Generates deterministic manifests for IR artifacts in asm/ir/.

Usage:
    gen_ir_manifest.py [--ir-dir=<dir>] [--output=<file>]
"""

import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base import BaseManifestGenerator, scan_directory, canonical_json


class IRManifestGenerator(BaseManifestGenerator):
    """Generator for IR manifests."""
    
    def __init__(self):
        super().__init__('ir')
    
    def _collect_entries(self, ir_dir: str = 'asm/ir', **kwargs):
        """Collect IR artifacts."""
        entries = scan_directory(ir_dir, extensions=['.dcbor', '.json'])
        
        # Add IR-specific metadata
        for entry in entries:
            entry['artifact_type'] = 'ir'
            if entry['name'].endswith('.dcbor'):
                entry['format'] = 'dcbor'
            else:
                entry['format'] = 'json'
        
        return entries


def main():
    ir_dir = 'asm/ir'
    output = 'receipts/ir_manifest.json'
    
    for arg in sys.argv[1:]:
        if arg.startswith('--ir-dir='):
            ir_dir = arg.split('=', 1)[1]
        elif arg.startswith('--output='):
            output = arg.split('=', 1)[1]
    
    generator = IRManifestGenerator()
    manifest = generator.generate(ir_dir=ir_dir)
    
    generator.write(manifest, output)
    
    print(f"IR manifest written to {output}", file=sys.stderr)
    print(f"Entries: {manifest['entry_count']}", file=sys.stderr)
    print(f"Hash: {manifest['manifest_hash']}", file=sys.stderr)
    print(canonical_json(manifest))


if __name__ == "__main__":
    main()
