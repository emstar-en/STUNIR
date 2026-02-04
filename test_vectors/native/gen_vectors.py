#!/usr/bin/env python3
"""STUNIR Native Test Vector Generator.

Generates deterministic test vectors for native tool integration.
Part of Issue #1034: Complete test_vectors → native pipeline stage.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base import (
    BaseTestVectorGenerator,
    canonical_json,
    compute_sha256,
    DEFAULT_EPOCH
)
from typing import Dict, List, Any


class NativeTestVectorGenerator(BaseTestVectorGenerator):
    """Generator for native tool test vectors."""

    def _generate_vectors(self) -> List[Dict[str, Any]]:
        """Generate native tool test vectors."""
        vectors: List[Dict[str, Any]] = []
        
        # Test 1: Haskell native tool manifest generation
        vectors.append(self.create_vector(
            index=1,
            name="Haskell Manifest Generation",
            description="Verify stunir-native gen-ir-manifest produces valid output",
            test_input={
                "tool": "stunir-native",
                "command": "gen-ir-manifest",
                "ir_dir": "asm/ir",
                "files": [
                    {"name": "module_a.dcbor", "size": 256},
                    {"name": "module_b.dcbor", "size": 512}
                ]
            },
            expected_output={
                "manifest_generated": True,
                "entry_count": 2,
                "deterministic": True,
                "schema": "stunir.manifest.ir.v1"
            },
            tags=["unit", "haskell", "manifest", "native"]
        ))
        
        # Test 2: Haskell provenance generation
        vectors.append(self.create_vector(
            index=2,
            name="Haskell Provenance Generation",
            description="Verify stunir-native gen-provenance produces valid C header",
            test_input={
                "tool": "stunir-native",
                "command": "gen-provenance",
                "spec_hash": "abc123def456",
                "modules": ["core", "util"]
            },
            expected_output={
                "header_generated": True,
                "has_epoch_macro": True,
                "has_spec_hash_macro": True,
                "has_module_list": True
            },
            tags=["unit", "haskell", "provenance", "native"]
        ))
        
        # Test 3: Rust native tool compilation
        vectors.append(self.create_vector(
            index=3,
            name="Rust Tool Compilation",
            description="Verify Rust native tool compiles successfully",
            test_input={
                "tool": "stunir-rust",
                "command": "build",
                "profile": "release",
                "target": "x86_64-unknown-linux-gnu"
            },
            expected_output={
                "compile_success": True,
                "binary_produced": True,
                "deterministic_hash": True
            },
            tags=["integration", "rust", "build", "native"]
        ))
        
        # Test 4: Native tool version check
        vectors.append(self.create_vector(
            index=4,
            name="Native Tool Version Check",
            description="Verify native tools report correct version",
            test_input={
                "tools": ["stunir-native", "stunir-rust"],
                "command": "--version"
            },
            expected_output={
                "all_respond": True,
                "version_format_valid": True,
                "schema_version_present": True
            },
            tags=["unit", "version", "native"]
        ))
        
        # Test 5: Native tool determinism check
        vectors.append(self.create_vector(
            index=5,
            name="Native Tool Determinism",
            description="Verify native tools produce deterministic output",
            test_input={
                "tool": "stunir-native",
                "command": "gen-ir-manifest",
                "runs": 2,
                "ir_dir": "asm/ir"
            },
            expected_output={
                "outputs_identical": True,
                "hashes_match": True,
                "deterministic": True
            },
            tags=["determinism", "native", "critical"]
        ))
        
        return vectors


def main() -> int:
    """Generate native test vectors."""
    import argparse
    parser = argparse.ArgumentParser(description='Generate native test vectors')
    parser.add_argument('--output', '-o', default=None, help='Output directory')
    args = parser.parse_args()
    
    generator = NativeTestVectorGenerator('native', args.output)
    count, manifest_hash = generator.write_vectors()
    
    print(f"Generated {count} native test vectors", file=sys.stderr)
    print(f"Manifest hash: {manifest_hash}", file=sys.stderr)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
