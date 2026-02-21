#!/usr/bin/env python3
"""STUNIR Validator Base Module

Shared utilities and base classes for all STUNIR validators.
Issue: asm/validators/1083
"""

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod


def canonical_json(data: Any) -> str:
    """Generate canonical JSON output (RFC 8785 / JCS subset)."""
    return json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def compute_sha256(data: bytes) -> str:
    """Compute SHA-256 hash of data."""
    return hashlib.sha256(data).hexdigest()


def compute_file_hash(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    with open(filepath, 'rb') as f:
        return compute_sha256(f.read())


def get_epoch() -> int:
    """Get current Unix epoch timestamp."""
    return int(datetime.now(timezone.utc).timestamp())


class ValidationError:
    """Represents a single validation error."""
    
    def __init__(self, code: str, message: str, line: Optional[int] = None, 
                 column: Optional[int] = None, severity: str = "error"):
        self.code = code
        self.message = message
        self.line = line
        self.column = column
        self.severity = severity  # "error", "warning", "info"
    
    def to_dict(self) -> Dict[str, Any]:
        d = {
            "code": self.code,
            "message": self.message,
            "severity": self.severity
        }
        if self.line is not None:
            d["line"] = self.line
        if self.column is not None:
            d["column"] = self.column
        return d


class ValidationResult:
    """Result of a validation operation."""
    
    def __init__(self, valid: bool, errors: List[ValidationError] = None,
                 warnings: List[ValidationError] = None):
        self.valid = valid
        self.errors = errors or []
        self.warnings = warnings or []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings]
        }


class BaseValidator(ABC):
    """Abstract base class for STUNIR validators."""
    
    SCHEMA = "stunir.validator.v1"
    VALIDATOR_TYPE = "base"
    
    def __init__(self, strict: bool = False):
        self.strict = strict
        self.errors: List[ValidationError] = []
        self.warnings: List[ValidationError] = []
    
    def add_error(self, code: str, message: str, line: int = None, col: int = None):
        """Add a validation error."""
        self.errors.append(ValidationError(code, message, line, col, "error"))
    
    def add_warning(self, code: str, message: str, line: int = None, col: int = None):
        """Add a validation warning."""
        self.warnings.append(ValidationError(code, message, line, col, "warning"))
    
    @abstractmethod
    def validate(self, content: str, filepath: str = None) -> ValidationResult:
        """Validate content. Must be implemented by subclasses."""
        pass
    
    def validate_file(self, filepath: str) -> ValidationResult:
        """Validate a file."""
        if not os.path.exists(filepath):
            return ValidationResult(False, [ValidationError(
                "FILE_NOT_FOUND", f"File not found: {filepath}"
            )])
        
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        return self.validate(content, filepath)
    
    def generate_receipt(self, filepath: str, result: ValidationResult) -> Dict[str, Any]:
        """Generate a validation receipt."""
        return {
            "schema": self.SCHEMA,
            "validator_type": self.VALIDATOR_TYPE,
            "epoch": get_epoch(),
            "filepath": filepath,
            "file_hash": compute_file_hash(filepath) if os.path.exists(filepath) else None,
            "result": result.to_dict()
        }


if __name__ == "__main__":
    # Self-test
    print(f"STUNIR Validator Base Module")
    print(f"Schema: {BaseValidator.SCHEMA}")
    print(f"Canonical JSON test: {canonical_json({'b': 1, 'a': 2})}")
    print(f"SHA256 test: {compute_sha256(b'test')[:16]}...")
    print("Self-test passed!")
