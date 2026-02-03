#!/usr/bin/env python3
"""STUNIR Contracts Manifest Verifier.

Part of manifests → contracts pipeline stage (Issue #1042).
Verifies contracts manifests against actual contract files.

Usage:
    verify_contracts_manifest.py <manifest.json> [--contracts-dir=<dir>]
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base import BaseManifestVerifier, compute_file_hash


class ContractsManifestVerifier(BaseManifestVerifier):
    """Verifier for contracts manifests."""
    
    def __init__(self):
        super().__init__('contracts')
    
    def _verify_entries(self, manifest, contracts_dir='contracts', **kwargs):
        errors = []
        warnings = []
        stats = {'verified': 0, 'missing': 0, 'hash_mismatch': 0}
        
        for entry in manifest.get('entries', []):
            filepath = os.path.join(contracts_dir, entry.get('path', ''))
            
            if not os.path.exists(filepath):
                errors.append(f"Contract not found: {entry.get('path')}")
                stats['missing'] += 1
                continue
            
            actual_hash = compute_file_hash(filepath)
            expected_hash = entry.get('hash', '')
            
            if expected_hash and actual_hash != expected_hash:
                errors.append(f"Hash mismatch for {entry.get('path')}")
                stats['hash_mismatch'] += 1
            else:
                stats['verified'] += 1
        
        return errors, warnings, stats


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <manifest.json> [--contracts-dir=<dir>]", file=sys.stderr)
        sys.exit(1)
    
    manifest_path = sys.argv[1]
    contracts_dir = 'contracts'
    
    for arg in sys.argv[2:]:
        if arg.startswith('--contracts-dir='):
            contracts_dir = arg.split('=', 1)[1]
    
    verifier = ContractsManifestVerifier()
    is_valid, errors, warnings, stats = verifier.verify(manifest_path, contracts_dir=contracts_dir)
    
    print(f"Manifest: {manifest_path}")
    print(f"Verified: {stats.get('verified', 0)}")
    print(f"Missing: {stats.get('missing', 0)}")
    print(f"Valid: {is_valid}")
    
    if errors:
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)
    
    print("✓ Contracts manifest verification passed")


if __name__ == "__main__":
    main()
