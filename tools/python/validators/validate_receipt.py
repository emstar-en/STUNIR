#!/usr/bin/env python3
"""STUNIR Receipt Validator - Validates receipt files.

This tool is part of the tools → validators pipeline stage.
It validates receipt files against expected schema.

Usage:
    validate_receipt.py <receipt.json> [--strict]
"""

import json
import sys
import hashlib
from typing import Any, Dict, List, Tuple

# Receipt Schema definitions
RECEIPT_SCHEMAS = {
    "stunir.receipt.build.v1": {
        "required": ["schema", "receipt_target", "receipt_status"],
        "optional": ["receipt_build_epoch", "receipt_epoch_json", "receipt_inputs", "receipt_tool", "receipt_argv"]
    },
    "stunir.receipt.v1": {
        "required": ["schema", "target"],
        "optional": ["status", "epoch", "inputs", "tool", "hash"]
    }
}

def canonical_json(data: Any) -> str:
    """Generate canonical JSON output."""
    return json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=False)

def compute_sha256(data: Any) -> str:
    """Compute SHA-256 hash."""
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.sha256(data).hexdigest()

def validate_receipt(receipt_data: Dict[str, Any], strict: bool = False) -> Tuple[bool, List[str], List[str], Dict[str, Any]]:
    """Validate receipt data against schema.

    Returns:
        tuple: (is_valid, errors, warnings, metadata)
    """
    errors = []
    warnings = []
    metadata = {}
    
    # Check schema field
    schema = receipt_data.get('schema', '')
    if not schema:
        errors.append("Missing 'schema' field")
        return False, errors, warnings, metadata
    
    metadata['schema'] = schema
    
    # Get schema definition
    schema_def = RECEIPT_SCHEMAS.get(schema)
    if not schema_def:
        if strict:
            errors.append(f"Unknown schema: {schema}")
        else:
            warnings.append(f"Unknown schema: {schema}")
    
    # Validate required fields
    if schema_def:
        for field in schema_def['required']:
            if field not in receipt_data:
                errors.append(f"Missing required field: {field}")
        
        # Check optional fields in strict mode
        if strict:
            for field in schema_def['optional']:
                if field not in receipt_data:
                    warnings.append(f"Missing optional field: {field}")
    
    # Extract target info
    target = receipt_data.get('receipt_target') or receipt_data.get('target', '')
    if target:
        metadata['target'] = target
    
    # Extract status
    status = receipt_data.get('receipt_status') or receipt_data.get('status', '')
    if status:
        metadata['status'] = status
        if status not in ['success', 'failure', 'pending', 'skipped']:
            warnings.append(f"Unusual status value: {status}")
    
    # Validate tool info if present
    tool = receipt_data.get('receipt_tool') or receipt_data.get('tool', {})
    if tool:
        if isinstance(tool, dict):
            if 'name' not in tool:
                warnings.append("Tool missing 'name'")
            metadata['tool_name'] = tool.get('name', 'unknown')
        else:
            errors.append("'tool' must be an object")
    
    # Compute content hash
    content_hash = compute_sha256(canonical_json(receipt_data))
    metadata['content_hash'] = content_hash

    is_valid = len(errors) == 0
    return is_valid, errors, warnings, metadata

def main() -> None:
    """Validate a receipt JSON file via CLI and print results."""
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <receipt.json> [--strict]", file=sys.stderr)
        print("\nSTUNIR Receipt Validator - Validates receipt files.", file=sys.stderr)
        sys.exit(1)
    
    input_path = sys.argv[1]
    strict = '--strict' in sys.argv
    
    try:
        # Read receipt file
        with open(input_path, 'r') as f:
            receipt_data = json.load(f)
        
        # Validate
        is_valid, errors, warnings, metadata = validate_receipt(receipt_data, strict)
        
        # Output results
        print(f"File: {input_path}")
        print(f"Schema: {metadata.get('schema', 'unknown')}")
        print(f"Target: {metadata.get('target', 'unknown')}")
        print(f"Status: {metadata.get('status', 'unknown')}")
        print(f"Tool: {metadata.get('tool_name', 'unknown')}")
        print(f"Valid: {is_valid}")
        
        if warnings:
            print("\nWarnings:")
            for w in warnings:
                print(f"  ⚠ {w}")
        
        if errors:
            print("\nErrors:")
            for e in errors:
                print(f"  ✗ {e}")
            sys.exit(1)
        
        print("\n✓ Receipt validation passed")
        
    except json.JSONDecodeError as e:
        print(f"JSON Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
