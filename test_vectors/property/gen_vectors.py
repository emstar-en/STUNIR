#!/usr/bin/env python3
"""STUNIR Property Test Vector Generator.

Generates deterministic test vectors for property-based testing and invariants.
Part of Issue #1135: Complete test_vectors → property pipeline stage.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base import (
    BaseTestVectorGenerator,
    canonical_json,
    compute_sha256,
    DEFAULT_EPOCH,
    seeded_rng
)
from typing import Dict, List


class PropertyTestVectorGenerator(BaseTestVectorGenerator):
    """Generator for property-based test vectors."""
    
    def _generate_vectors(self) -> List[Dict]:
        """Generate property-based test vectors."""
        vectors = []
        rng = seeded_rng(42)  # Deterministic seed
        
        # Test 1: Idempotence property
        vectors.append(self.create_vector(
            index=1,
            name="Canonicalization Idempotence",
            description="Verify canonicalize(canonicalize(x)) == canonicalize(x)",
            test_input={
                "property": "idempotence",
                "operation": "canonicalize",
                "data": {"z": 3, "a": 1, "m": 2},  # Unsorted
                "iterations": 3
            },
            expected_output={
                "property_holds": True,
                "all_outputs_equal": True,
                "canonical_form": '{"a":1,"m":2,"z":3}'
            },
            tags=["property", "idempotence", "canonicalization"]
        ))
        
        # Test 2: Commutativity property (hash independence of order)
        vectors.append(self.create_vector(
            index=2,
            name="Hash Order Independence",
            description="Verify hash of sorted data equals hash of unsorted then sorted",
            test_input={
                "property": "commutativity",
                "operation": "canonical_hash",
                "data_variants": [
                    {"a": 1, "b": 2, "c": 3},
                    {"c": 3, "a": 1, "b": 2},
                    {"b": 2, "c": 3, "a": 1}
                ]
            },
            expected_output={
                "property_holds": True,
                "all_hashes_equal": True,
                "deterministic": True
            },
            tags=["property", "commutativity", "hash"]
        ))
        
        # Test 3: Invertibility property
        vectors.append(self.create_vector(
            index=3,
            name="Serialize-Deserialize Roundtrip",
            description="Verify deserialize(serialize(x)) == x",
            test_input={
                "property": "invertibility",
                "operations": ["serialize", "deserialize"],
                "data": {
                    "schema": "stunir.spec.v1",
                    "id": "roundtrip_test",
                    "profile": "profile3",
                    "values": [1, 2.5, True, "text", None]
                }
            },
            expected_output={
                "property_holds": True,
                "roundtrip_successful": True,
                "data_preserved": True
            },
            tags=["property", "invertibility", "roundtrip"]
        ))
        
        # Test 4: Monotonicity property
        vectors.append(self.create_vector(
            index=4,
            name="Epoch Monotonicity",
            description="Verify epochs always increase for sequential operations",
            test_input={
                "property": "monotonicity",
                "field": "epoch",
                "sequence": [
                    {"op": "create", "epoch": DEFAULT_EPOCH},
                    {"op": "update", "epoch": DEFAULT_EPOCH + 1},
                    {"op": "finalize", "epoch": DEFAULT_EPOCH + 2}
                ]
            },
            expected_output={
                "property_holds": True,
                "strictly_increasing": True,
                "no_regression": True
            },
            tags=["property", "monotonicity", "temporal"]
        ))
        
        # Test 5: Transitivity property
        vectors.append(self.create_vector(
            index=5,
            name="Dependency Transitivity",
            description="Verify if A depends on B, and B depends on C, then A indirectly depends on C",
            test_input={
                "property": "transitivity",
                "relation": "depends_on",
                "graph": {
                    "A": ["B"],
                    "B": ["C"],
                    "C": []
                }
            },
            expected_output={
                "property_holds": True,
                "transitive_closure": {
                    "A": ["B", "C"],
                    "B": ["C"],
                    "C": []
                }
            },
            tags=["property", "transitivity", "dependency"]
        ))
        
        # Test 6: Determinism property (critical for STUNIR)
        vectors.append(self.create_vector(
            index=6,
            name="Pipeline Determinism",
            description="Verify full pipeline produces identical output for identical input",
            test_input={
                "property": "determinism",
                "pipeline_stages": ["parse", "transform", "emit", "manifest"],
                "runs": 5,
                "seed": 42,
                "spec": {
                    "schema": "stunir.spec.v1",
                    "id": "determinism_spec",
                    "profile": "profile3"
                }
            },
            expected_output={
                "property_holds": True,
                "all_outputs_identical": True,
                "hash_stable_across_runs": True,
                "no_timestamp_variance": True
            },
            tags=["property", "determinism", "critical", "pipeline"]
        ))
        
        # Test 7: Associativity property
        vectors.append(self.create_vector(
            index=7,
            name="Manifest Merge Associativity",
            description="Verify merge(merge(A, B), C) == merge(A, merge(B, C))",
            test_input={
                "property": "associativity",
                "operation": "manifest_merge",
                "manifests": [
                    {"entries": [{"id": "a", "hash": "h1"}]},
                    {"entries": [{"id": "b", "hash": "h2"}]},
                    {"entries": [{"id": "c", "hash": "h3"}]}
                ]
            },
            expected_output={
                "property_holds": True,
                "left_assoc_hash": True,
                "right_assoc_hash": True,
                "hashes_equal": True
            },
            tags=["property", "associativity", "merge"]
        ))
        
        return vectors


def main():
    """Generate property test vectors."""
    import argparse
    parser = argparse.ArgumentParser(description='Generate property test vectors')
    parser.add_argument('--output', '-o', default=None, help='Output directory')
    args = parser.parse_args()
    
    generator = PropertyTestVectorGenerator('property', args.output)
    count, manifest_hash = generator.write_vectors()
    
    print(f"Generated {count} property test vectors", file=sys.stderr)
    print(f"Manifest hash: {manifest_hash}", file=sys.stderr)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
