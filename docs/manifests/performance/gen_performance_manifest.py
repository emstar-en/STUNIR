#!/usr/bin/env python3
"""STUNIR Performance Manifest Generator.

Part of manifests → performance pipeline stage (Issue #1145).
Generates deterministic manifests for performance benchmarks.

Usage:
    gen_performance_manifest.py [--output=<file>]
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base import BaseManifestGenerator, canonical_json


class PerformanceManifestGenerator(BaseManifestGenerator):
    """Generator for performance manifests."""
    
    BENCHMARK_DEFINITIONS = [
        {'name': 'spec_parse_time', 'unit': 'ms', 'description': 'Time to parse STUNIR spec'},
        {'name': 'ir_emit_time', 'unit': 'ms', 'description': 'Time to emit IR'},
        {'name': 'target_emit_time', 'unit': 'ms', 'description': 'Time to emit target code'},
        {'name': 'manifest_gen_time', 'unit': 'ms', 'description': 'Time to generate manifests'},
        {'name': 'verification_time', 'unit': 'ms', 'description': 'Time to verify artifacts'},
        {'name': 'total_pipeline_time', 'unit': 'ms', 'description': 'Total pipeline execution time'},
        {'name': 'memory_peak', 'unit': 'MB', 'description': 'Peak memory usage'},
        {'name': 'output_size', 'unit': 'bytes', 'description': 'Total output size'},
    ]
    
    def __init__(self):
        super().__init__('performance')
    
    def _collect_entries(self, **kwargs):
        """Collect performance benchmark definitions."""
        entries = []
        
        for benchmark in self.BENCHMARK_DEFINITIONS:
            entry = {
                'benchmark_name': benchmark['name'],
                'unit': benchmark['unit'],
                'description': benchmark['description'],
                'value': None,  # To be filled by actual benchmark runs
                'artifact_type': 'performance_benchmark'
            }
            entries.append(entry)
        
        return entries


def main():
    output = 'receipts/performance_manifest.json'
    
    for arg in sys.argv[1:]:
        if arg.startswith('--output='):
            output = arg.split('=', 1)[1]
    
    generator = PerformanceManifestGenerator()
    manifest = generator.generate()
    
    generator.write(manifest, output)
    
    print(f"Performance manifest written to {output}", file=sys.stderr)
    print(f"Benchmarks: {manifest['entry_count']}", file=sys.stderr)
    print(f"Hash: {manifest['manifest_hash']}", file=sys.stderr)
    print(canonical_json(manifest))


if __name__ == "__main__":
    main()
