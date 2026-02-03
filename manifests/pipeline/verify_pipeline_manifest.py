#!/usr/bin/env python3
"""STUNIR Pipeline Manifest Verifier.

Part of manifests → pipeline pipeline stage (Issue #1073).
Verifies pipeline manifests for consistency.

Usage:
    verify_pipeline_manifest.py <manifest.json>
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base import BaseManifestVerifier


class PipelineManifestVerifier(BaseManifestVerifier):
    """Verifier for pipeline manifests."""
    
    REQUIRED_STAGES = ['spec_parse', 'ir_emit', 'ir_canonicalize', 'target_emit', 'manifest_gen', 'verify']
    
    def __init__(self):
        super().__init__('pipeline')
    
    def _verify_entries(self, manifest, **kwargs):
        errors = []
        warnings = []
        stats = {'verified': 0, 'missing': 0}
        
        stage_names = [e.get('stage_name') for e in manifest.get('entries', [])]
        
        for required in self.REQUIRED_STAGES:
            if required in stage_names:
                stats['verified'] += 1
            else:
                warnings.append(f"Missing recommended stage: {required}")
                stats['missing'] += 1
        
        # Verify stage ordering
        entries = manifest.get('entries', [])
        orders = [e.get('stage_order', 0) for e in entries]
        if orders != sorted(orders):
            errors.append("Pipeline stages are not in correct order")
        
        return errors, warnings, stats


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <manifest.json>", file=sys.stderr)
        sys.exit(1)
    
    manifest_path = sys.argv[1]
    
    verifier = PipelineManifestVerifier()
    is_valid, errors, warnings, stats = verifier.verify(manifest_path)
    
    print(f"Manifest: {manifest_path}")
    print(f"Stages verified: {stats.get('verified', 0)}")
    print(f"Stages missing: {stats.get('missing', 0)}")
    print(f"Valid: {is_valid}")
    
    if warnings:
        for w in warnings:
            print(f"  ⚠ {w}")
    
    if errors:
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)
    
    print("✓ Pipeline manifest verification passed")


if __name__ == "__main__":
    main()
