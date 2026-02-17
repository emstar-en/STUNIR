#!/usr/bin/env python3
"""STUNIR IR Manifest Verifier.

Part of manifests → ir pipeline stage (Issue #1015).
Verifies IR manifests against actual artifacts.

Usage:
    verify_ir_manifest.py <manifest.json> [--ir-dir=<dir>]
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Tuple, Any
from base import BaseManifestVerifier, compute_file_hash


class IRManifestVerifier(BaseManifestVerifier):
    """Verifier for IR manifests."""

    def __init__(self) -> None:
        super().__init__('ir')

    def _verify_entries(self, manifest: Dict[str, Any], ir_dir: str = 'asm/ir', **kwargs: Any) -> Tuple[List[str], List[str], Dict[str, int]]:
        errors: List[str] = []
        warnings: List[str] = []
        stats: Dict[str, int] = {'verified': 0, 'missing': 0, 'hash_mismatch': 0}

        for entry in manifest.get('entries', []):
            filepath = os.path.join(ir_dir, entry.get('path', ''))

            if not os.path.exists(filepath):
                errors.append(f"IR artifact not found: {entry.get('path')}")
                stats['missing'] += 1
                continue

            actual_hash = compute_file_hash(filepath)
            expected_hash = entry.get('hash', '')

            if expected_hash and actual_hash != expected_hash:
                errors.append(f"Hash mismatch for {entry.get('path')}: expected={expected_hash}, actual={actual_hash}")
                stats['hash_mismatch'] += 1
            else:
                stats['verified'] += 1

        return errors, warnings, stats


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <manifest.json> [--ir-dir=<dir>]", file=sys.stderr)
        sys.exit(1)

    manifest_path = sys.argv[1]
    ir_dir = 'asm/ir'

    for arg in sys.argv[2:]:
        if arg.startswith('--ir-dir='):
            ir_dir = arg.split('=', 1)[1]
    
    verifier = IRManifestVerifier()
    is_valid, errors, warnings, stats = verifier.verify(manifest_path, ir_dir=ir_dir)
    
    print(f"Manifest: {manifest_path}")
    print(f"Verified: {stats.get('verified', 0)}")
    print(f"Missing: {stats.get('missing', 0)}")
    print(f"Hash mismatch: {stats.get('hash_mismatch', 0)}")
    print(f"Valid: {is_valid}")
    
    if errors:
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)
    
    print("✓ IR manifest verification passed")


if __name__ == "__main__":
    main()
