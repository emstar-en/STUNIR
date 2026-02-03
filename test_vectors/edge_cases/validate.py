#!/usr/bin/env python3
"""STUNIR Edge Cases Test Vector Validator.

Validates edge case test vectors for integrity and correctness.
Part of Issue #1065: Complete test_vectors → edge_cases pipeline stage.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base import BaseTestVectorValidator, validate_schema
from typing import Dict, List, Tuple


class EdgeCasesTestVectorValidator(BaseTestVectorValidator):
    """Validator for edge case test vectors."""
    
    REQUIRED_FIELDS = ['schema', 'id', 'name', 'input', 'expected_output']
    
    def _validate_vector(self, vector: Dict) -> Tuple[bool, str]:
        """Validate an edge case test vector."""
        # Check required fields
        is_valid, missing = validate_schema(vector, self.REQUIRED_FIELDS)
        if not is_valid:
            return False, f"Missing required fields: {missing}"
        
        # Check schema prefix
        schema = vector.get('schema', '')
        if not schema.startswith('stunir.test_vector.edge_cases.'):
            return False, f"Invalid schema: {schema}"
        
        # Edge cases should have operation specified
        test_input = vector.get('input', {})
        if 'operation' not in test_input:
            return False, "Edge case input should specify 'operation'"
        
        # Edge cases should have tags including 'edge_case'
        tags = vector.get('tags', [])
        if 'edge_case' not in tags:
            return False, "Edge case vector should have 'edge_case' tag"
        
        return True, "Validation passed"


def main():
    """Validate edge cases test vectors."""
    import argparse
    parser = argparse.ArgumentParser(description='Validate edge cases test vectors')
    parser.add_argument('--dir', '-d', default=None, help='Test vectors directory')
    args = parser.parse_args()
    
    validator = EdgeCasesTestVectorValidator('edge_cases', args.dir)
    passed, failed, results = validator.validate_all()
    
    print(f"Edge Cases Test Vectors Validation", file=sys.stderr)
    print(f"  Passed: {passed}", file=sys.stderr)
    print(f"  Failed: {failed}", file=sys.stderr)
    
    for result in results:
        status = '✓' if result['valid'] else '✗'
        print(f"  {status} {result['id']}: {result['validation']}", file=sys.stderr)
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
