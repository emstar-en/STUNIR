#!/usr/bin/env python3
"""STUNIR Runtime Manifest Generator.

Part of manifests → runtime pipeline stage (Issue #1074).
Generates deterministic manifests for runtime configuration.

Usage:
    gen_runtime_manifest.py [--output=<file>]
"""

import sys
import os
import platform
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base import BaseManifestGenerator, canonical_json


class RuntimeManifestGenerator(BaseManifestGenerator):
    """Generator for runtime manifests."""
    
    REQUIRED_TOOLS = ['python3', 'ghc', 'cargo', 'gcc', 'sha256sum']
    
    def __init__(self):
        super().__init__('runtime')
    
    def _collect_entries(self, **kwargs):
        """Collect runtime environment information."""
        entries = []
        
        # System information
        entries.append({
            'entry_type': 'system',
            'name': 'platform',
            'value': platform.system(),
            'artifact_type': 'runtime_config'
        })
        entries.append({
            'entry_type': 'system',
            'name': 'architecture',
            'value': platform.machine(),
            'artifact_type': 'runtime_config'
        })
        entries.append({
            'entry_type': 'system',
            'name': 'python_version',
            'value': platform.python_version(),
            'artifact_type': 'runtime_config'
        })
        
        # Tool availability
        for tool in self.REQUIRED_TOOLS:
            tool_path = shutil.which(tool)
            entries.append({
                'entry_type': 'tool',
                'name': tool,
                'available': tool_path is not None,
                'path': tool_path or 'not_found',
                'artifact_type': 'runtime_tool'
            })
        
        return entries


def main():
    output = 'receipts/runtime_manifest.json'
    
    for arg in sys.argv[1:]:
        if arg.startswith('--output='):
            output = arg.split('=', 1)[1]
    
    generator = RuntimeManifestGenerator()
    manifest = generator.generate()
    
    generator.write(manifest, output)
    
    print(f"Runtime manifest written to {output}", file=sys.stderr)
    print(f"Entries: {manifest['entry_count']}", file=sys.stderr)
    print(f"Hash: {manifest['manifest_hash']}", file=sys.stderr)
    print(canonical_json(manifest))


if __name__ == "__main__":
    main()
