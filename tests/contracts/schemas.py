#!/usr/bin/env python3
"""
STUNIR JSON Schema Definitions
==============================

Defines JSON schemas for contract validation.
"""

import json
from typing import Any, Dict, List, Tuple

# ============================================================================
# IR Schema
# ============================================================================

IR_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "STUNIR IR",
    "type": "object",
    "required": ["module", "functions"],
    "properties": {
        "ir_schema": {"type": "string", "pattern": "^stunir\\.ir\\.v\\d+$"},
        "module": {"type": "string", "minLength": 1},
        "ir_version": {"type": "string"},
        "ir_epoch": {"type": "integer", "minimum": 0},
        "ir_spec_hash": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
        "functions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name"],
                "properties": {
                    "name": {"type": "string"},
                    "params": {"type": "array"},
                    "return_type": {"type": "string"},
                    "body": {"type": "array"}
                }
            }
        },
        "types": {"type": "array"},
        "imports": {"type": "array"},
        "exports": {"type": "array", "items": {"type": "string"}}
    }
}

# ============================================================================
# Manifest Schema
# ============================================================================

MANIFEST_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "STUNIR Manifest",
    "type": "object",
    "required": ["manifest_schema", "entries"],
    "properties": {
        "manifest_schema": {"type": "string", "pattern": "^stunir\\.manifest\\.[a-z]+\\.v\\d+$"},
        "manifest_epoch": {"type": "integer", "minimum": 0},
        "manifest_hash": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
        "entries": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "hash"],
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "path": {"type": "string"},
                    "hash": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
                    "size": {"type": "integer", "minimum": 0},
                    "artifact_type": {"type": "string"},
                    "format": {"type": "string"}
                }
            }
        }
    }
}

# ============================================================================
# Receipt Schema
# ============================================================================

RECEIPT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "STUNIR Receipt",
    "type": "object",
    "required": ["receipt_schema", "module", "hash"],
    "properties": {
        "receipt_schema": {"type": "string", "pattern": "^stunir\\.receipt\\.v\\d+$"},
        "module": {"type": "string", "minLength": 1},
        "hash": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
        "timestamp": {"type": "integer", "minimum": 0},
        "artifacts": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "hash"],
                "properties": {
                    "name": {"type": "string"},
                    "hash": {"type": "string", "pattern": "^[a-f0-9]{64}$"}
                }
            }
        },
        "metadata": {"type": "object"}
    }
}

# ============================================================================
# Validation Functions
# ============================================================================

def validate_against_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate data against JSON schema without external dependencies."""
    errors = []
    
    def check_type(value, expected_type, path):
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None)
        }
        expected = type_map.get(expected_type)
        if expected and not isinstance(value, expected):
            errors.append(f"{path}: expected {expected_type}, got {type(value).__name__}")
            return False
        return True
    
    def validate_obj(obj, sch, path=""):
        # Check type
        obj_type = sch.get("type")
        if obj_type:
            if not check_type(obj, obj_type, path):
                return
        
        # Check required
        if obj_type == "object" and "required" in sch:
            for req in sch["required"]:
                if req not in obj:
                    errors.append(f"{path}: missing required field '{req}'")
        
        # Check properties
        if obj_type == "object" and "properties" in sch and isinstance(obj, dict):
            for key, prop_schema in sch["properties"].items():
                if key in obj:
                    validate_obj(obj[key], prop_schema, f"{path}.{key}")
        
        # Check array items
        if obj_type == "array" and "items" in sch and isinstance(obj, list):
            for i, item in enumerate(obj):
                validate_obj(item, sch["items"], f"{path}[{i}]")
        
        # Check minimum
        if "minimum" in sch and isinstance(obj, (int, float)):
            if obj < sch["minimum"]:
                errors.append(f"{path}: value {obj} below minimum {sch['minimum']}")
        
        # Check minLength
        if "minLength" in sch and isinstance(obj, str):
            if len(obj) < sch["minLength"]:
                errors.append(f"{path}: string length {len(obj)} below minimum {sch['minLength']}")
        
        # Check pattern
        if "pattern" in sch and isinstance(obj, str):
            import re
            if not re.match(sch["pattern"], obj):
                errors.append(f"{path}: string '{obj[:50]}' doesn't match pattern '{sch['pattern']}'")
    
    validate_obj(data, schema)
    return len(errors) == 0, errors


def validate_ir(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate IR data against schema."""
    return validate_against_schema(data, IR_SCHEMA)


def validate_manifest(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate manifest data against schema."""
    return validate_against_schema(data, MANIFEST_SCHEMA)


def validate_receipt(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate receipt data against schema."""
    return validate_against_schema(data, RECEIPT_SCHEMA)
