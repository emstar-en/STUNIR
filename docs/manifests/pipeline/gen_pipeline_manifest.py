#!/usr/bin/env python3
"""STUNIR Pipeline Manifest Generator.

Part of manifests → pipeline pipeline stage (Issue #1073).
Generates deterministic manifests for pipeline execution state.

Usage:
    gen_pipeline_manifest.py [--pipeline-config=<file>] [--output=<file>]
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base import BaseManifestGenerator, canonical_json


class PipelineManifestGenerator(BaseManifestGenerator):
    """Generator for pipeline manifests."""
    
    PIPELINE_STAGES = [
        {'name': 'spec_parse', 'order': 1, 'description': 'Parse STUNIR spec'},
        {'name': 'ir_emit', 'order': 2, 'description': 'Emit IR artifacts'},
        {'name': 'ir_canonicalize', 'order': 3, 'description': 'Canonicalize IR to dCBOR'},
        {'name': 'target_emit', 'order': 4, 'description': 'Emit target code'},
        {'name': 'manifest_gen', 'order': 5, 'description': 'Generate manifests'},
        {'name': 'verify', 'order': 6, 'description': 'Verify artifacts'},
    ]
    
    def __init__(self):
        super().__init__('pipeline')
    
    def _collect_entries(self, **kwargs):
        """Collect pipeline stage definitions."""
        entries = []
        
        for stage in self.PIPELINE_STAGES:
            entry = {
                'stage_name': stage['name'],
                'stage_order': stage['order'],
                'description': stage['description'],
                'status': 'defined',
                'artifact_type': 'pipeline_stage'
            }
            entries.append(entry)
        
        return entries


def main():
    output = 'receipts/pipeline_manifest.json'
    
    for arg in sys.argv[1:]:
        if arg.startswith('--output='):
            output = arg.split('=', 1)[1]
    
    generator = PipelineManifestGenerator()
    manifest = generator.generate()
    
    generator.write(manifest, output)
    
    print(f"Pipeline manifest written to {output}", file=sys.stderr)
    print(f"Stages: {manifest['entry_count']}", file=sys.stderr)
    print(f"Hash: {manifest['manifest_hash']}", file=sys.stderr)
    print(canonical_json(manifest))


if __name__ == "__main__":
    main()
