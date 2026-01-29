#!/usr/bin/env python3
"""STUNIR Contracts Manifest Generator.

Part of manifests → contracts pipeline stage (Issue #1042).
Generates deterministic manifests for validated contracts.

Usage:
    gen_contracts_manifest.py [--contracts-dir=<dir>] [--output=<file>]
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base import BaseManifestGenerator, scan_directory, canonical_json


class ContractsManifestGenerator(BaseManifestGenerator):
    """Generator for contracts manifests."""
    
    def __init__(self):
        super().__init__('contracts')
    
    def _collect_entries(self, contracts_dir='contracts', **kwargs):
        """Collect contract files."""
        entries = scan_directory(contracts_dir, extensions=['.json', '.yaml', '.toml'])
        
        for entry in entries:
            entry['artifact_type'] = 'contract'
            
            # Try to extract contract metadata
            filepath = os.path.join(contracts_dir, entry.get('path', ''))
            if filepath.endswith('.json'):
                try:
                    with open(filepath, 'r') as f:
                        contract_data = json.load(f)
                    entry['contract_version'] = contract_data.get('version', 'unknown')
                    entry['contract_schema'] = contract_data.get('schema', 'unknown')
                except (json.JSONDecodeError, UnicodeDecodeError, IOError, OSError) as e:
                    # Log the specific error for debugging while handling gracefully
                    import logging
                    logging.debug(f"Could not parse contract {filepath}: {type(e).__name__}: {e}")
                    entry['contract_version'] = 'unknown'
                    entry['contract_schema'] = 'unknown'
            
            entry['format'] = entry['name'].split('.')[-1]
        
        return entries


def main():
    contracts_dir = 'contracts'
    output = 'receipts/contracts_manifest.json'
    
    for arg in sys.argv[1:]:
        if arg.startswith('--contracts-dir='):
            contracts_dir = arg.split('=', 1)[1]
        elif arg.startswith('--output='):
            output = arg.split('=', 1)[1]
    
    generator = ContractsManifestGenerator()
    manifest = generator.generate(contracts_dir=contracts_dir)
    
    generator.write(manifest, output)
    
    print(f"Contracts manifest written to {output}", file=sys.stderr)
    print(f"Entries: {manifest['entry_count']}", file=sys.stderr)
    print(f"Hash: {manifest['manifest_hash']}", file=sys.stderr)
    print(canonical_json(manifest))


if __name__ == "__main__":
    main()
