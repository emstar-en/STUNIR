#!/usr/bin/env python3
"""STUNIR Performance Manifest Verifier.

Part of manifests → performance pipeline stage (Issue #1145).
Verifies performance manifests for completeness.

Usage:
    verify_performance_manifest.py <manifest.json>
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base import BaseManifestVerifier


class PerformanceManifestVerifier(BaseManifestVerifier):
    """Verifier for performance manifests."""
    
    REQUIRED_BENCHMARKS = ['spec_parse_time', 'ir_emit_time', 'total_pipeline_time']
    
    def __init__(self):
        super().__init__('performance')
    
    def _verify_entries(self, manifest, **kwargs):
        errors = []
        warnings = []
        stats = {'verified': 0, 'missing': 0}
        
        benchmark_names = [e.get('benchmark_name') for e in manifest.get('entries', [])]
        
        for required in self.REQUIRED_BENCHMARKS:
            if required not in benchmark_names:
                errors.append(f"Required benchmark missing: {required}")
                stats['missing'] += 1
            else:
                stats['verified'] += 1
        
        # Verify all entries have valid units
        valid_units = ['ms', 's', 'MB', 'KB', 'bytes', 'count']
        for entry in manifest.get('entries', []):
            unit = entry.get('unit', '')
            if unit and unit not in valid_units:
                warnings.append(f"Unknown unit '{unit}' for benchmark {entry.get('benchmark_name')}")
        
        return errors, warnings, stats


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <manifest.json>", file=sys.stderr)
        sys.exit(1)
    
    manifest_path = sys.argv[1]
    
    verifier = PerformanceManifestVerifier()
    is_valid, errors, warnings, stats = verifier.verify(manifest_path)
    
    print(f"Manifest: {manifest_path}")
    print(f"Benchmarks verified: {stats.get('verified', 0)}")
    print(f"Benchmarks missing: {stats.get('missing', 0)}")
    print(f"Valid: {is_valid}")
    
    if warnings:
        for w in warnings:
            print(f"  ⚠ {w}")
    
    if errors:
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)
    
    print("✓ Performance manifest verification passed")


if __name__ == "__main__":
    main()
