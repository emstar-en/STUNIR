#!/usr/bin/env python3
"""STUNIR Polyglot Test Vector Generator.

Generates deterministic test vectors for cross-language target generation.
Part of Issue #1035: Complete test_vectors → polyglot pipeline stage.
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
from typing import Dict, List


class PolyglotTestVectorGenerator(BaseTestVectorGenerator):
    """Generator for polyglot target test vectors."""
    
    def _generate_vectors(self) -> List[Dict]:
        """Generate polyglot test vectors."""
        vectors = []
        
        # Test 1: Rust target generation
        vectors.append(self.create_vector(
            index=1,
            name="Rust Target Generation",
            description="Verify IR-to-Rust target code generation",
            test_input={
                "ir": {
                    "schema": "stunir.ir.v1",
                    "module": "test_module",
                    "functions": [
                        {"name": "add", "params": ["i32", "i32"], "returns": "i32"}
                    ]
                },
                "target": "rust"
            },
            expected_output={
                "target_generated": True,
                "has_cargo_toml": True,
                "has_lib_rs": True,
                "type_mapping_valid": True
            },
            tags=["unit", "rust", "polyglot"]
        ))
        
        # Test 2: C89 target generation
        vectors.append(self.create_vector(
            index=2,
            name="C89 Target Generation",
            description="Verify IR-to-C89 target code generation",
            test_input={
                "ir": {
                    "schema": "stunir.ir.v1",
                    "module": "test_module",
                    "functions": [
                        {"name": "multiply", "params": ["f64", "f64"], "returns": "f64"}
                    ]
                },
                "target": "c89"
            },
            expected_output={
                "target_generated": True,
                "ansi_compliant": True,
                "has_header": True,
                "has_source": True
            },
            tags=["unit", "c89", "polyglot"]
        ))
        
        # Test 3: C99 target generation
        vectors.append(self.create_vector(
            index=3,
            name="C99 Target Generation",
            description="Verify IR-to-C99 target code generation",
            test_input={
                "ir": {
                    "schema": "stunir.ir.v1",
                    "module": "test_module",
                    "functions": [
                        {"name": "compute", "params": ["i64"], "returns": "bool"}
                    ]
                },
                "target": "c99"
            },
            expected_output={
                "target_generated": True,
                "c99_features_used": True,
                "has_stdbool": True,
                "has_stdint": True
            },
            tags=["unit", "c99", "polyglot"]
        ))
        
        # Test 4: Cross-language type mapping
        vectors.append(self.create_vector(
            index=4,
            name="Cross-Language Type Mapping",
            description="Verify IR types map correctly to all targets",
            test_input={
                "ir_types": ["i32", "i64", "f32", "f64", "bool", "string"],
                "targets": ["rust", "c89", "c99"]
            },
            expected_output={
                "all_types_mapped": True,
                "rust_mappings": {"i32": "i32", "i64": "i64", "f32": "f32", "f64": "f64", "bool": "bool", "string": "String"},
                "c89_mappings": {"i32": "int", "i64": "long", "f32": "float", "f64": "double", "bool": "int", "string": "char*"},
                "c99_mappings": {"i32": "int32_t", "i64": "int64_t", "f32": "float", "f64": "double", "bool": "bool", "string": "char*"}
            },
            tags=["unit", "types", "polyglot"]
        ))
        
        # Test 5: Polyglot determinism
        vectors.append(self.create_vector(
            index=5,
            name="Polyglot Determinism",
            description="Verify all targets produce deterministic output",
            test_input={
                "ir": {
                    "schema": "stunir.ir.v1",
                    "module": "determinism_test",
                    "functions": [{"name": "test", "params": [], "returns": "void"}]
                },
                "targets": ["rust", "c89", "c99"],
                "runs": 2
            },
            expected_output={
                "all_deterministic": True,
                "rust_hash_stable": True,
                "c89_hash_stable": True,
                "c99_hash_stable": True
            },
            tags=["determinism", "polyglot", "critical"]
        ))
        
        return vectors


def main():
    """Generate polyglot test vectors."""
    import argparse
    parser = argparse.ArgumentParser(description='Generate polyglot test vectors')
    parser.add_argument('--output', '-o', default=None, help='Output directory')
    args = parser.parse_args()
    
    generator = PolyglotTestVectorGenerator('polyglot', args.output)
    count, manifest_hash = generator.write_vectors()
    
    print(f"Generated {count} polyglot test vectors", file=sys.stderr)
    print(f"Manifest hash: {manifest_hash}", file=sys.stderr)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
