#!/usr/bin/env python3
"""STUNIR Edge Cases Test Vector Generator.

Generates deterministic test vectors for boundary conditions and edge cases.
Part of Issue #1065: Complete test_vectors → edge_cases pipeline stage.
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


class EdgeCasesTestVectorGenerator(BaseTestVectorGenerator):
    """Generator for edge case test vectors."""
    
    def _generate_vectors(self) -> List[Dict]:
        """Generate edge case test vectors."""
        vectors = []
        
        # Test 1: Empty input handling
        vectors.append(self.create_vector(
            index=1,
            name="Empty Spec Input",
            description="Verify graceful handling of empty spec file",
            test_input={
                "spec": {},
                "operation": "parse"
            },
            expected_output={
                "parsed": False,
                "error": "Empty or invalid spec",
                "graceful_failure": True
            },
            tags=["edge_case", "empty", "error_handling"]
        ))
        
        # Test 2: None/null value handling
        vectors.append(self.create_vector(
            index=2,
            name="Null Value Handling",
            description="Verify handling of null values in spec",
            test_input={
                "spec": {
                    "schema": "stunir.spec.v1",
                    "id": None,
                    "profile": "profile3"
                },
                "operation": "validate"
            },
            expected_output={
                "valid": False,
                "null_fields": ["id"],
                "error": "Required field 'id' is null"
            },
            tags=["edge_case", "null", "validation"]
        ))
        
        # Test 3: Maximum field length
        vectors.append(self.create_vector(
            index=3,
            name="Maximum Field Length",
            description="Verify handling of maximum length strings",
            test_input={
                "spec": {
                    "schema": "stunir.spec.v1",
                    "id": "a" * 1000,  # Very long ID
                    "profile": "profile3"
                },
                "operation": "validate"
            },
            expected_output={
                "valid": False,
                "error": "Field 'id' exceeds maximum length (256)",
                "max_length_enforced": True
            },
            tags=["edge_case", "length", "boundary"]
        ))
        
        # Test 4: Unicode handling
        vectors.append(self.create_vector(
            index=4,
            name="Unicode Character Handling",
            description="Verify proper handling of Unicode characters",
            test_input={
                "spec": {
                    "schema": "stunir.spec.v1",
                    "id": "spec_中文_日本語_한국語_001",
                    "profile": "profile3",
                    "description": "Élève café naïve résumé"
                },
                "operation": "serialize"
            },
            expected_output={
                "serialized": True,
                "unicode_preserved": True,
                "encoding": "utf-8"
            },
            tags=["edge_case", "unicode", "i18n"]
        ))
        
        # Test 5: Circular reference detection
        vectors.append(self.create_vector(
            index=5,
            name="Circular Reference Detection",
            description="Verify detection of circular module references",
            test_input={
                "modules": [
                    {"name": "A", "imports": ["B"]},
                    {"name": "B", "imports": ["C"]},
                    {"name": "C", "imports": ["A"]}  # Circular
                ],
                "operation": "analyze_dependencies"
            },
            expected_output={
                "circular_detected": True,
                "cycle": ["A", "B", "C", "A"],
                "error": "Circular dependency detected"
            },
            tags=["edge_case", "circular", "dependency"]
        ))
        
        # Test 6: Extremely nested structure
        vectors.append(self.create_vector(
            index=6,
            name="Deep Nesting Handling",
            description="Verify handling of deeply nested structures",
            test_input={
                "spec": {
                    "level1": {
                        "level2": {
                            "level3": {
                                "level4": {
                                    "level5": {
                                        "value": "deep"
                                    }
                                }
                            }
                        }
                    }
                },
                "operation": "flatten",
                "max_depth": 10
            },
            expected_output={
                "flattened": True,
                "depth_detected": 5,
                "within_limit": True
            },
            tags=["edge_case", "nesting", "structure"]
        ))
        
        # Test 7: Special character handling in keys
        vectors.append(self.create_vector(
            index=7,
            name="Special Characters in Keys",
            description="Verify handling of special characters in JSON keys",
            test_input={
                "spec": {
                    "schema": "stunir.spec.v1",
                    "id": "spec-with-dashes",
                    "profile": "profile3",
                    "meta": {
                        "key.with.dots": "value1",
                        "key:with:colons": "value2",
                        "key/with/slashes": "value3"
                    }
                },
                "operation": "serialize"
            },
            expected_output={
                "serialized": True,
                "keys_preserved": True,
                "canonical_output": True
            },
            tags=["edge_case", "special_chars", "serialization"]
        ))
        
        return vectors


def main():
    """Generate edge cases test vectors."""
    import argparse
    parser = argparse.ArgumentParser(description='Generate edge cases test vectors')
    parser.add_argument('--output', '-o', default=None, help='Output directory')
    args = parser.parse_args()
    
    generator = EdgeCasesTestVectorGenerator('edge_cases', args.output)
    count, manifest_hash = generator.write_vectors()
    
    print(f"Generated {count} edge cases test vectors", file=sys.stderr)
    print(f"Manifest hash: {manifest_hash}", file=sys.stderr)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
