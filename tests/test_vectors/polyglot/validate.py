#!/usr/bin/env python3
"""STUNIR Polyglot Test Vector Validator.

Validates polyglot target test vectors for integrity and correctness.
Part of Issue #1035: Complete test_vectors → polyglot pipeline stage.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base import BaseTestVectorValidator, validate_schema
from typing import Dict, List, Tuple


class PolyglotTestVectorValidator(BaseTestVectorValidator):
    """Validator for polyglot test vectors."""
    
    REQUIRED_FIELDS = ['schema', 'id', 'name', 'input', 'expected_output']
    VALID_TARGETS = ['rust', 'c89', 'c99']
    
    def _validate_vector(self, vector: Dict) -> Tuple[bool, str]:
        """Validate a polyglot test vector."""
        # Check required fields
        is_valid, missing = validate_schema(vector, self.REQUIRED_FIELDS)
        if not is_valid:
            return False, f"Missing required fields: {missing}"
        
        # Check schema prefix
        schema = vector.get('schema', '')
        if not schema.startswith('stunir.test_vector.polyglot.'):
            return False, f"Invalid schema: {schema}"
        
        # Check input has target or IR specification
        test_input = vector.get('input', {})
        if 'target' not in test_input and 'targets' not in test_input and 'ir_types' not in test_input:
            return False, "Input must contain 'target', 'targets', or 'ir_types' field"
        
        return True, "Validation passed"


def main():
    """Validate polyglot test vectors."""
    import argparse
    parser = argparse.ArgumentParser(description='Validate polyglot test vectors')
    parser.add_argument('--dir', '-d', default=None, help='Test vectors directory')
    args = parser.parse_args()
    
    validator = PolyglotTestVectorValidator('polyglot', args.dir)
    passed, failed, results = validator.validate_all()
    
    print(f"Polyglot Test Vectors Validation", file=sys.stderr)
    print(f"  Passed: {passed}", file=sys.stderr)
    print(f"  Failed: {failed}", file=sys.stderr)
    
    for result in results:
        status = '✓' if result['valid'] else '✗'
        print(f"  {status} {result['id']}: {result['validation']}", file=sys.stderr)
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
