#!/usr/bin/env python3
"""STUNIR Test Vectors Package.

Provides test vector generation and validation for STUNIR pipeline.
Part of Phase 5: Test Vectors pipeline stages.

Issues Addressed:
- test_vectors/contracts/1011: Complete test_vectors → contracts pipeline stage
- test_vectors/native/1034: Complete test_vectors → native pipeline stage
- test_vectors/polyglot/1035: Complete test_vectors → polyglot pipeline stage
- test_vectors/receipts/1036: Complete test_vectors → receipts pipeline stage
- test_vectors/edge_cases/1065: Complete test_vectors → edge_cases pipeline stage
- test_vectors/property/1135: Complete test_vectors → property pipeline stage
"""

from .base import (
    canonical_json,
    canonical_json_pretty,
    compute_sha256,
    compute_file_hash,
    seeded_rng,
    generate_test_id,
    validate_schema,
    BaseTestVectorGenerator,
    BaseTestVectorValidator,
    DEFAULT_EPOCH,
    TEST_SEED
)

__all__ = [
    'canonical_json',
    'canonical_json_pretty',
    'compute_sha256',
    'compute_file_hash',
    'seeded_rng',
    'generate_test_id',
    'validate_schema',
    'BaseTestVectorGenerator',
    'BaseTestVectorValidator',
    'DEFAULT_EPOCH',
    'TEST_SEED'
]

__version__ = '1.0.0'
