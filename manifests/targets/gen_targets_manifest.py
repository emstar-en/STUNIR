#!/usr/bin/env python3
"""STUNIR Targets Manifest Generator.

Part of manifests → targets pipeline stage (Issue #1043).
Generates deterministic manifests for generated target code.

Usage:
    gen_targets_manifest.py [--targets-dir=<dir>] [--output=<file>]
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base import BaseManifestGenerator, scan_directory, canonical_json


class TargetsManifestGenerator(BaseManifestGenerator):
    """Generator for targets manifests."""
    
    TARGET_EXTENSIONS = [
        '.py', '.rs', '.c', '.h', '.asm', '.s',
        '.wasm', '.wat', '.js', '.ts', '.go',
        '.json', '.toml', '.yaml'
    ]
    
    def __init__(self):
        super().__init__('targets')
    
    def _collect_entries(self, targets_dir='targets', **kwargs):
        """Collect target artifacts."""
        entries = scan_directory(targets_dir, extensions=self.TARGET_EXTENSIONS)
        
        for entry in entries:
            entry['artifact_type'] = 'target'
            
            # Determine target type from path
            path_parts = entry.get('path', '').split(os.sep)
            if len(path_parts) >= 2:
                entry['target_category'] = path_parts[0]  # e.g., 'polyglot', 'assembly'
                entry['target_type'] = path_parts[1] if len(path_parts) > 1 else 'unknown'
            
            # Determine format from extension
            ext = entry['name'].split('.')[-1]
            entry['format'] = ext
        
        return entries


def main():
    targets_dir = 'targets'
    output = 'receipts/targets_manifest.json'
    
    for arg in sys.argv[1:]:
        if arg.startswith('--targets-dir='):
            targets_dir = arg.split('=', 1)[1]
        elif arg.startswith('--output='):
            output = arg.split('=', 1)[1]
    
    generator = TargetsManifestGenerator()
    manifest = generator.generate(targets_dir=targets_dir)
    
    generator.write(manifest, output)
    
    print(f"Targets manifest written to {output}", file=sys.stderr)
    print(f"Entries: {manifest['entry_count']}", file=sys.stderr)
    print(f"Hash: {manifest['manifest_hash']}", file=sys.stderr)
    print(canonical_json(manifest))


if __name__ == "__main__":
    main()
