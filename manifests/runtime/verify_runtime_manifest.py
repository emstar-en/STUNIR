#!/usr/bin/env python3
"""STUNIR Runtime Manifest Verifier.

Part of manifests → runtime pipeline stage (Issue #1074).
Verifies runtime manifests against current environment.

Usage:
    verify_runtime_manifest.py <manifest.json>
"""

import sys
import os
import platform
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base import BaseManifestVerifier


class RuntimeManifestVerifier(BaseManifestVerifier):
    """Verifier for runtime manifests."""
    
    CRITICAL_TOOLS = ['python3', 'sha256sum']
    
    def __init__(self):
        super().__init__('runtime')
    
    def _verify_entries(self, manifest, **kwargs):
        errors = []
        warnings = []
        stats = {'verified': 0, 'missing': 0, 'mismatch': 0}
        
        for entry in manifest.get('entries', []):
            entry_type = entry.get('entry_type')
            
            if entry_type == 'system':
                # Verify system matches
                name = entry.get('name')
                expected = entry.get('value')
                
                if name == 'platform':
                    actual = platform.system()
                elif name == 'architecture':
                    actual = platform.machine()
                elif name == 'python_version':
                    actual = platform.python_version()
                else:
                    actual = None
                
                if actual and actual != expected:
                    warnings.append(f"System mismatch for {name}: expected={expected}, actual={actual}")
                    stats['mismatch'] += 1
                else:
                    stats['verified'] += 1
            
            elif entry_type == 'tool':
                name = entry.get('name')
                was_available = entry.get('available')
                is_available = shutil.which(name) is not None
                
                if was_available and not is_available:
                    if name in self.CRITICAL_TOOLS:
                        errors.append(f"Critical tool no longer available: {name}")
                    else:
                        warnings.append(f"Tool no longer available: {name}")
                    stats['missing'] += 1
                else:
                    stats['verified'] += 1
        
        return errors, warnings, stats


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <manifest.json>", file=sys.stderr)
        sys.exit(1)
    
    manifest_path = sys.argv[1]
    
    verifier = RuntimeManifestVerifier()
    is_valid, errors, warnings, stats = verifier.verify(manifest_path)
    
    print(f"Manifest: {manifest_path}")
    print(f"Verified: {stats.get('verified', 0)}")
    print(f"Missing tools: {stats.get('missing', 0)}")
    print(f"Mismatches: {stats.get('mismatch', 0)}")
    print(f"Valid: {is_valid}")
    
    if warnings:
        for w in warnings:
            print(f"  ⚠ {w}")
    
    if errors:
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)
    
    print("✓ Runtime manifest verification passed")


if __name__ == "__main__":
    main()
