#!/usr/bin/env python3
"""STUNIR Contracts Test Vector Generator.

Generates deterministic test vectors for contracts validation.
Part of Issue #1011: Complete test_vectors → contracts pipeline stage.
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


class ContractsTestVectorGenerator(BaseTestVectorGenerator):
    """Generator for contracts test vectors."""
    
    def _generate_vectors(self) -> List[Dict]:
        """Generate contracts test vectors."""
        vectors = []
        
        # Test 1: Profile 2 contract validation
        vectors.append(self.create_vector(
            index=1,
            name="Profile 2 Contract Structure",
            description="Verify Profile 2 contract schema compliance",
            test_input={
                "contract": {
                    "schema": "stunir.contract.profile2.v1",
                    "id": "contract_p2_001",
                    "profile": "profile2",
                    "stages": ["build"],
                    "epoch": DEFAULT_EPOCH
                }
            },
            expected_output={
                "valid": True,
                "profile_valid": True,
                "schema_compliant": True
            },
            tags=["unit", "profile2", "contracts"]
        ))
        
        # Test 2: Profile 3 contract validation
        vectors.append(self.create_vector(
            index=2,
            name="Profile 3 Contract Structure",
            description="Verify Profile 3 contract with test stage",
            test_input={
                "contract": {
                    "schema": "stunir.contract.profile3.v1",
                    "id": "contract_p3_001",
                    "profile": "profile3",
                    "stages": ["build", "test"],
                    "epoch": DEFAULT_EPOCH
                }
            },
            expected_output={
                "valid": True,
                "profile_valid": True,
                "has_test_stage": True
            },
            tags=["unit", "profile3", "contracts"]
        ))
        
        # Test 3: Profile 4 contract validation
        vectors.append(self.create_vector(
            index=3,
            name="Profile 4 Contract Structure",
            description="Verify Profile 4 contract with attestation",
            test_input={
                "contract": {
                    "schema": "stunir.contract.profile4.v1",
                    "id": "contract_p4_001",
                    "profile": "profile4",
                    "stages": ["build", "test", "attest"],
                    "attestation": {
                        "type": "deterministic",
                        "verifier": "internal"
                    },
                    "epoch": DEFAULT_EPOCH
                }
            },
            expected_output={
                "valid": True,
                "profile_valid": True,
                "has_attestation": True
            },
            tags=["unit", "profile4", "contracts", "attestation"]
        ))
        
        # Test 4: Invalid profile contract
        vectors.append(self.create_vector(
            index=4,
            name="Invalid Profile Contract",
            description="Detect invalid profile specification",
            test_input={
                "contract": {
                    "schema": "stunir.contract.v1",
                    "id": "contract_invalid_001",
                    "profile": "profile99",
                    "stages": ["build"],
                    "epoch": DEFAULT_EPOCH
                }
            },
            expected_output={
                "valid": False,
                "profile_valid": False,
                "error": "Unknown profile: profile99"
            },
            tags=["unit", "validation", "contracts", "negative"]
        ))
        
        # Test 5: Contract stage ordering
        vectors.append(self.create_vector(
            index=5,
            name="Contract Stage Ordering",
            description="Verify contract stages are properly ordered",
            test_input={
                "contract": {
                    "schema": "stunir.contract.profile3.v1",
                    "id": "contract_order_001",
                    "profile": "profile3",
                    "stages": ["test", "build"],  # Wrong order
                    "epoch": DEFAULT_EPOCH
                }
            },
            expected_output={
                "valid": False,
                "stage_order_valid": False,
                "error": "Stages must be ordered: build before test"
            },
            tags=["unit", "ordering", "contracts"]
        ))
        
        return vectors


def main():
    """Generate contracts test vectors."""
    import argparse
    parser = argparse.ArgumentParser(description='Generate contracts test vectors')
    parser.add_argument('--output', '-o', default=None, help='Output directory')
    args = parser.parse_args()
    
    generator = ContractsTestVectorGenerator('contracts', args.output)
    count, manifest_hash = generator.write_vectors()
    
    print(f"Generated {count} contracts test vectors", file=sys.stderr)
    print(f"Manifest hash: {manifest_hash}", file=sys.stderr)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
