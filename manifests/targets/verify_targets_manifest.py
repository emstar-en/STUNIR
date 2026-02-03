#!/usr/bin/env python3
"""STUNIR Targets Manifest Verifier.

Part of manifests → targets pipeline stage (Issue #1043).
Verifies targets manifests against actual target files.

Usage:
    verify_targets_manifest.py <manifest.json> [--targets-dir=<dir>]
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base import BaseManifestVerifier, compute_file_hash


class TargetsManifestVerifier(BaseManifestVerifier):
    """Verifier for targets manifests."""
    
    def __init__(self):
        super().__init__('targets')
    
    def _verify_entries(self, manifest, targets_dir='targets', **kwargs):
        errors = []
        warnings = []
        stats = {'verified': 0, 'missing': 0, 'hash_mismatch': 0}
        
        for entry in manifest.get('entries', []):
            filepath = os.path.join(targets_dir, entry.get('path', ''))
            
            if not os.path.exists(filepath):
                errors.append(f"Target not found: {entry.get('path')}")
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
        print(f"Usage: {sys.argv[0]} <manifest.json> [--targets-dir=<dir>]", file=sys.stderr)
        sys.exit(1)
    
    manifest_path = sys.argv[1]
    targets_dir = 'targets'
    
    for arg in sys.argv[2:]:
        if arg.startswith('--targets-dir='):
            targets_dir = arg.split('=', 1)[1]
    
    verifier = TargetsManifestVerifier()
    is_valid, errors, warnings, stats = verifier.verify(manifest_path, targets_dir=targets_dir)
    
    print(f"Manifest: {manifest_path}")
    print(f"Verified: {stats.get('verified', 0)}")
    print(f"Missing: {stats.get('missing', 0)}")
    print(f"Valid: {is_valid}")
    
    if errors:
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)
    
    print("✓ Targets manifest verification passed")


if __name__ == "__main__":
    main()
