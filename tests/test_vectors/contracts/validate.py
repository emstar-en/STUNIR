#!/usr/bin/env python3
"""STUNIR Contracts Test Vector Validator.

Validates contracts test vectors for integrity and correctness.
Part of Issue #1011: Complete test_vectors → contracts pipeline stage.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base import BaseTestVectorValidator, validate_schema
from typing import Dict, List, Tuple


class ContractsTestVectorValidator(BaseTestVectorValidator):
    """Validator for contracts test vectors."""
    
    REQUIRED_FIELDS = ['schema', 'id', 'name', 'input', 'expected_output']
    VALID_PROFILES = ['profile2', 'profile3', 'profile4']
    
    def _validate_vector(self, vector: Dict) -> Tuple[bool, str]:
        """Validate a contracts test vector."""
        # Check required fields
        is_valid, missing = validate_schema(vector, self.REQUIRED_FIELDS)
        if not is_valid:
            return False, f"Missing required fields: {missing}"
        
        # Check schema prefix
        schema = vector.get('schema', '')
        if not schema.startswith('stunir.test_vector.contracts.'):
            return False, f"Invalid schema: {schema}"
        
        # Check input has contract data
        test_input = vector.get('input', {})
        if 'contract' not in test_input:
            return False, "Input must contain 'contract' field"
        
        return True, "Validation passed"


def main():
    """Validate contracts test vectors."""
    import argparse
    parser = argparse.ArgumentParser(description='Validate contracts test vectors')
    parser.add_argument('--dir', '-d', default=None, help='Test vectors directory')
    args = parser.parse_args()
    
    validator = ContractsTestVectorValidator('contracts', args.dir)
    passed, failed, results = validator.validate_all()
    
    print(f"Contracts Test Vectors Validation", file=sys.stderr)
    print(f"  Passed: {passed}", file=sys.stderr)
    print(f"  Failed: {failed}", file=sys.stderr)
    
    for result in results:
        status = '✓' if result['valid'] else '✗'
        print(f"  {status} {result['id']}: {result['validation']}", file=sys.stderr)
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
