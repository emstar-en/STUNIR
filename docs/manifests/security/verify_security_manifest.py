#!/usr/bin/env python3
"""STUNIR Security Manifest Verifier.

Part of manifests → security pipeline stage (Issue #1144).
Verifies security manifests for completeness.

Usage:
    verify_security_manifest.py <manifest.json>
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base import BaseManifestVerifier


class SecurityManifestVerifier(BaseManifestVerifier):
    """Verifier for security manifests."""
    
    CRITICAL_CHECKS = ['deterministic_output', 'hash_verification', 'canonical_json', 'input_validation']
    
    def __init__(self):
        super().__init__('security')
    
    def _verify_entries(self, manifest, **kwargs):
        errors = []
        warnings = []
        stats = {'verified': 0, 'missing': 0, 'failed': 0}
        
        check_names = [e.get('check_name') for e in manifest.get('entries', [])]
        
        for critical in self.CRITICAL_CHECKS:
            if critical not in check_names:
                errors.append(f"Critical security check missing: {critical}")
                stats['missing'] += 1
            else:
                stats['verified'] += 1
        
        # Verify all entries have required fields
        for entry in manifest.get('entries', []):
            if entry.get('status') != 'attested':
                if entry.get('critical', False):
                    errors.append(f"Critical check not attested: {entry.get('check_name')}")
                    stats['failed'] += 1
                else:
                    warnings.append(f"Optional check not attested: {entry.get('check_name')}")
        
        return errors, warnings, stats


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <manifest.json>", file=sys.stderr)
        sys.exit(1)
    
    manifest_path = sys.argv[1]
    
    verifier = SecurityManifestVerifier()
    is_valid, errors, warnings, stats = verifier.verify(manifest_path)
    
    print(f"Manifest: {manifest_path}")
    print(f"Checks verified: {stats.get('verified', 0)}")
    print(f"Checks missing: {stats.get('missing', 0)}")
    print(f"Checks failed: {stats.get('failed', 0)}")
    print(f"Valid: {is_valid}")
    
    if warnings:
        for w in warnings:
            print(f"  ⚠ {w}")
    
    if errors:
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)
    
    print("✓ Security manifest verification passed")


if __name__ == "__main__":
    main()
