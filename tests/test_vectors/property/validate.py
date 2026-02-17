#!/usr/bin/env python3
"""STUNIR Property Test Vector Validator.

Validates property-based test vectors for integrity and correctness.
Part of Issue #1135: Complete test_vectors → property pipeline stage.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base import BaseTestVectorValidator, validate_schema
from typing import Dict, List, Tuple


class PropertyTestVectorValidator(BaseTestVectorValidator):
    """Validator for property-based test vectors."""
    
    REQUIRED_FIELDS = ['schema', 'id', 'name', 'input', 'expected_output']
    VALID_PROPERTIES = [
        'idempotence', 'commutativity', 'invertibility',
        'monotonicity', 'transitivity', 'determinism', 'associativity'
    ]
    
    def _validate_vector(self, vector: Dict) -> Tuple[bool, str]:
        """Validate a property test vector."""
        # Check required fields
        is_valid, missing = validate_schema(vector, self.REQUIRED_FIELDS)
        if not is_valid:
            return False, f"Missing required fields: {missing}"
        
        # Check schema prefix
        schema = vector.get('schema', '')
        if not schema.startswith('stunir.test_vector.property.'):
            return False, f"Invalid schema: {schema}"
        
        # Property tests should have property specified
        test_input = vector.get('input', {})
        if 'property' not in test_input:
            return False, "Property test input should specify 'property'"
        
        # Check property is valid
        prop = test_input.get('property')
        if prop not in self.VALID_PROPERTIES:
            return False, f"Unknown property: {prop}"
        
        # Property tests should have tags including 'property'
        tags = vector.get('tags', [])
        if 'property' not in tags:
            return False, "Property test vector should have 'property' tag"
        
        # Expected output should indicate property_holds
        expected = vector.get('expected_output', {})
        if 'property_holds' not in expected:
            return False, "Expected output should have 'property_holds' field"
        
        return True, "Validation passed"


def main():
    """Validate property test vectors."""
    import argparse
    parser = argparse.ArgumentParser(description='Validate property test vectors')
    parser.add_argument('--dir', '-d', default=None, help='Test vectors directory')
    args = parser.parse_args()
    
    validator = PropertyTestVectorValidator('property', args.dir)
    passed, failed, results = validator.validate_all()
    
    print(f"Property Test Vectors Validation", file=sys.stderr)
    print(f"  Passed: {passed}", file=sys.stderr)
    print(f"  Failed: {failed}", file=sys.stderr)
    
    for result in results:
        status = '✓' if result['valid'] else '✗'
        print(f"  {status} {result['id']}: {result['validation']}", file=sys.stderr)
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
