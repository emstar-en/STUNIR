#!/usr/bin/env python3
"""STUNIR Native Test Vector Validator.

Validates native tool test vectors for integrity and correctness.
Part of Issue #1034: Complete test_vectors → native pipeline stage.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base import BaseTestVectorValidator, validate_schema
from typing import Dict, List, Tuple


class NativeTestVectorValidator(BaseTestVectorValidator):
    """Validator for native tool test vectors."""
    
    REQUIRED_FIELDS = ['schema', 'id', 'name', 'input', 'expected_output']
    VALID_TOOLS = ['stunir-native', 'stunir-rust']
    
    def _validate_vector(self, vector: Dict) -> Tuple[bool, str]:
        """Validate a native test vector."""
        # Check required fields
        is_valid, missing = validate_schema(vector, self.REQUIRED_FIELDS)
        if not is_valid:
            return False, f"Missing required fields: {missing}"
        
        # Check schema prefix
        schema = vector.get('schema', '')
        if not schema.startswith('stunir.test_vector.native.'):
            return False, f"Invalid schema: {schema}"
        
        # Check input has tool specification
        test_input = vector.get('input', {})
        if 'tool' not in test_input and 'tools' not in test_input:
            return False, "Input must contain 'tool' or 'tools' field"
        
        return True, "Validation passed"


def main():
    """Validate native test vectors."""
    import argparse
    parser = argparse.ArgumentParser(description='Validate native test vectors')
    parser.add_argument('--dir', '-d', default=None, help='Test vectors directory')
    args = parser.parse_args()
    
    validator = NativeTestVectorValidator('native', args.dir)
    passed, failed, results = validator.validate_all()
    
    print(f"Native Test Vectors Validation", file=sys.stderr)
    print(f"  Passed: {passed}", file=sys.stderr)
    print(f"  Failed: {failed}", file=sys.stderr)
    
    for result in results:
        status = '✓' if result['valid'] else '✗'
        print(f"  {status} {result['id']}: {result['validation']}", file=sys.stderr)
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
