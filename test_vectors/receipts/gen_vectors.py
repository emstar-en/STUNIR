#!/usr/bin/env python3
"""STUNIR Receipts Test Vector Generator.

Generates deterministic test vectors for receipts validation.
Part of Issue #1036: Complete test_vectors → receipts pipeline stage.
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


class ReceiptsTestVectorGenerator(BaseTestVectorGenerator):
    """Generator for receipts test vectors."""
    
    def _generate_vectors(self) -> List[Dict]:
        """Generate receipts test vectors."""
        vectors = []
        
        # Test 1: Basic receipt validation
        vectors.append(self.create_vector(
            index=1,
            name="Basic Receipt Structure",
            description="Verify basic receipt JSON structure is valid",
            test_input={
                "receipt": {
                    "schema": "stunir.receipt.v1",
                    "id": "receipt_001",
                    "artifact_hash": "abc123def456",
                    "epoch": DEFAULT_EPOCH
                }
            },
            expected_output={
                "valid": True,
                "schema_compliant": True,
                "has_required_fields": True
            },
            tags=["unit", "schema", "receipts"]
        ))
        
        # Test 2: Receipt with missing required field
        vectors.append(self.create_vector(
            index=2,
            name="Receipt Missing Hash",
            description="Verify detection of missing artifact_hash field",
            test_input={
                "receipt": {
                    "schema": "stunir.receipt.v1",
                    "id": "receipt_002",
                    "epoch": DEFAULT_EPOCH
                }
            },
            expected_output={
                "valid": False,
                "schema_compliant": False,
                "missing_fields": ["artifact_hash"]
            },
            tags=["unit", "validation", "receipts"]
        ))
        
        # Test 3: Receipt hash verification
        vectors.append(self.create_vector(
            index=3,
            name="Receipt Hash Verification",
            description="Verify receipt hash matches artifact content",
            test_input={
                "artifact_content": "Hello, STUNIR!",
                "receipt": {
                    "schema": "stunir.receipt.v1",
                    "id": "receipt_003",
                    "artifact_hash": compute_sha256("Hello, STUNIR!"),
                    "epoch": DEFAULT_EPOCH
                }
            },
            expected_output={
                "valid": True,
                "hash_verified": True,
                "hash_match": True
            },
            tags=["unit", "hash", "integrity", "receipts"]
        ))
        
        # Test 4: Receipt manifest generation
        vectors.append(self.create_vector(
            index=4,
            name="Receipt Manifest Generation",
            description="Verify receipts manifest is properly generated",
            test_input={
                "receipts": [
                    {"id": "r1", "hash": "hash1"},
                    {"id": "r2", "hash": "hash2"}
                ]
            },
            expected_output={
                "manifest_generated": True,
                "entry_count": 2,
                "deterministic": True
            },
            tags=["unit", "manifest", "receipts"]
        ))
        
        # Test 5: Receipt timestamp validation
        vectors.append(self.create_vector(
            index=5,
            name="Receipt Epoch Validation",
            description="Verify receipt epoch is valid Unix timestamp",
            test_input={
                "receipt": {
                    "schema": "stunir.receipt.v1",
                    "id": "receipt_005",
                    "artifact_hash": "abc123",
                    "epoch": DEFAULT_EPOCH
                }
            },
            expected_output={
                "valid": True,
                "epoch_valid": True,
                "epoch_range_valid": True
            },
            tags=["unit", "timestamp", "receipts"]
        ))
        
        return vectors


def main():
    """Generate receipts test vectors."""
    import argparse
    parser = argparse.ArgumentParser(description='Generate receipts test vectors')
    parser.add_argument('--output', '-o', default=None, help='Output directory')
    args = parser.parse_args()
    
    generator = ReceiptsTestVectorGenerator('receipts', args.output)
    count, manifest_hash = generator.write_vectors()
    
    print(f"Generated {count} receipts test vectors", file=sys.stderr)
    print(f"Manifest hash: {manifest_hash}", file=sys.stderr)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
