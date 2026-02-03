#!/usr/bin/env python3
"""STUNIR Error Handling Example

This example demonstrates error handling patterns in STUNIR:
- Custom exception classes
- Validation with detailed errors
- Recovery strategies
- Error logging and reporting

Usage:
    python error_handling.py
"""

import json
import hashlib
import sys
import traceback
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

# =============================================================================
# Logging Configuration
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('stunir')

# =============================================================================
# Custom Exceptions
# =============================================================================

class STUNIRError(Exception):
    """Base exception for all STUNIR errors."""
    
    def __init__(self, message: str, code: str = "STUNIR_ERROR", 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}
        self.timestamp = datetime.now(timezone.utc).isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": self.code,
            "message": str(self),
            "details": self.details,
            "timestamp": self.timestamp
        }

class ValidationError(STUNIRError):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, field: str, value: Any = None):
        super().__init__(
            message, 
            code="VALIDATION_ERROR",
            details={"field": field, "value": str(value)[:100]}
        )
        self.field = field
        self.value = value

class ParseError(STUNIRError):
    """Raised when parsing fails."""
    
    def __init__(self, message: str, line: Optional[int] = None, 
                 column: Optional[int] = None):
        details = {}
        if line is not None:
            details["line"] = line
        if column is not None:
            details["column"] = column
        super().__init__(message, code="PARSE_ERROR", details=details)

class IRGenerationError(STUNIRError):
    """Raised when IR generation fails."""
    
    def __init__(self, message: str, module: str, function: Optional[str] = None):
        super().__init__(
            message,
            code="IR_GENERATION_ERROR",
            details={"module": module, "function": function}
        )

class HashMismatchError(STUNIRError):
    """Raised when hash verification fails."""
    
    def __init__(self, expected: str, actual: str, artifact: str):
        super().__init__(
            f"Hash mismatch for {artifact}",
            code="HASH_MISMATCH",
            details={
                "expected": expected,
                "actual": actual,
                "artifact": artifact
            }
        )

class DeterminismError(STUNIRError):
    """Raised when determinism verification fails."""
    
    def __init__(self, message: str, hashes: List[str]):
        super().__init__(
            message,
            code="DETERMINISM_ERROR",
            details={"hashes": hashes}
        )

# =============================================================================
# Error Severity
# =============================================================================

class ErrorSeverity(Enum):
    """Severity levels for errors."""
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

# =============================================================================
# Error Collection
# =============================================================================

@dataclass
class ErrorReport:
    """Container for collected errors."""
    errors: List[STUNIRError] = field(default_factory=list)
    warnings: List[STUNIRError] = field(default_factory=list)
    
    def add_error(self, error: STUNIRError):
        self.errors.append(error)
        logger.error(f"{error.code}: {error}")
    
    def add_warning(self, error: STUNIRError):
        self.warnings.append(error)
        logger.warning(f"{error.code}: {error}")
    
    def has_errors(self) -> bool:
        return len(self.errors) > 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
            "error_count": len(self.errors),
            "warning_count": len(self.warnings)
        }

# =============================================================================
# Validation Utilities
# =============================================================================

class SpecValidator:
    """Validates STUNIR spec files."""
    
    REQUIRED_FIELDS = ["name", "version", "functions"]
    VALID_TYPES = ["i32", "i64", "f32", "f64", "bool", "str", "void"]
    
    def __init__(self):
        self.report = ErrorReport()
    
    def validate(self, spec: Dict[str, Any]) -> Tuple[bool, ErrorReport]:
        """Validate a spec and return (is_valid, report)."""
        logger.info("Starting spec validation...")
        
        # Check required fields
        for field in self.REQUIRED_FIELDS:
            if field not in spec:
                self.report.add_error(
                    ValidationError(f"Missing required field: {field}", field)
                )
        
        # Validate name
        if "name" in spec:
            name = spec["name"]
            if not isinstance(name, str):
                self.report.add_error(
                    ValidationError("Name must be a string", "name", name)
                )
            elif not name.isidentifier():
                self.report.add_warning(
                    ValidationError("Name should be a valid identifier", "name", name)
                )
        
        # Validate version
        if "version" in spec:
            version = spec["version"]
            if not self._validate_semver(version):
                self.report.add_warning(
                    ValidationError("Version should follow semver format", "version", version)
                )
        
        # Validate functions
        if "functions" in spec:
            self._validate_functions(spec["functions"])
        
        # Validate exports
        if "exports" in spec:
            self._validate_exports(spec)
        
        is_valid = not self.report.has_errors()
        logger.info(f"Validation {'passed' if is_valid else 'failed'}")
        
        return is_valid, self.report
    
    def _validate_semver(self, version: str) -> bool:
        """Check if version follows semantic versioning."""
        import re
        pattern = r'^\d+\.\d+\.\d+(-[a-zA-Z0-9]+)?$'
        return bool(re.match(pattern, str(version)))
    
    def _validate_functions(self, functions: List[Any]):
        """Validate function definitions."""
        if not isinstance(functions, list):
            self.report.add_error(
                ValidationError("Functions must be a list", "functions", type(functions))
            )
            return
        
        seen_names = set()
        for i, func in enumerate(functions):
            if not isinstance(func, dict):
                self.report.add_error(
                    ValidationError(f"Function {i} must be an object", f"functions[{i}]")
                )
                continue
            
            # Check function name
            if "name" not in func:
                self.report.add_error(
                    ValidationError(f"Function {i} missing name", f"functions[{i}].name")
                )
            else:
                name = func["name"]
                if name in seen_names:
                    self.report.add_error(
                        ValidationError(f"Duplicate function name: {name}", f"functions[{i}].name", name)
                    )
                seen_names.add(name)
            
            # Validate parameters
            if "params" in func:
                self._validate_params(func["params"], i)
            
            # Validate return type
            if "returns" in func:
                ret_type = func["returns"]
                if ret_type not in self.VALID_TYPES:
                    self.report.add_warning(
                        ValidationError(f"Unknown return type", f"functions[{i}].returns", ret_type)
                    )
    
    def _validate_params(self, params: List[Any], func_index: int):
        """Validate function parameters."""
        if not isinstance(params, list):
            self.report.add_error(
                ValidationError("Params must be a list", f"functions[{func_index}].params")
            )
            return
        
        for j, param in enumerate(params):
            if not isinstance(param, dict):
                self.report.add_error(
                    ValidationError(
                        f"Parameter must be an object",
                        f"functions[{func_index}].params[{j}]"
                    )
                )
                continue
            
            if "name" not in param:
                self.report.add_error(
                    ValidationError(
                        "Parameter missing name",
                        f"functions[{func_index}].params[{j}].name"
                    )
                )
            
            if "type" not in param:
                self.report.add_error(
                    ValidationError(
                        "Parameter missing type",
                        f"functions[{func_index}].params[{j}].type"
                    )
                )
            elif param["type"] not in self.VALID_TYPES:
                self.report.add_warning(
                    ValidationError(
                        f"Unknown parameter type",
                        f"functions[{func_index}].params[{j}].type",
                        param["type"]
                    )
                )
    
    def _validate_exports(self, spec: Dict[str, Any]):
        """Validate that exports reference existing functions."""
        exports = spec.get("exports", [])
        func_names = {f.get("name") for f in spec.get("functions", [])}
        
        for export in exports:
            if export not in func_names:
                self.report.add_error(
                    ValidationError(
                        f"Export references non-existent function: {export}",
                        "exports",
                        export
                    )
                )

# =============================================================================
# Recovery Strategies
# =============================================================================

class RecoveryStrategy:
    """Base class for error recovery strategies."""
    
    def can_recover(self, error: STUNIRError) -> bool:
        """Check if this strategy can handle the error."""
        return False
    
    def recover(self, error: STUNIRError, context: Dict[str, Any]) -> Any:
        """Attempt to recover from the error."""
        raise NotImplementedError

class DefaultValueRecovery(RecoveryStrategy):
    """Recover by providing default values."""
    
    DEFAULTS = {
        "name": "unnamed_module",
        "version": "0.0.0",
        "functions": [],
        "exports": []
    }
    
    def can_recover(self, error: STUNIRError) -> bool:
        if isinstance(error, ValidationError):
            return error.field in self.DEFAULTS
        return False
    
    def recover(self, error: STUNIRError, context: Dict[str, Any]) -> Any:
        if isinstance(error, ValidationError):
            default = self.DEFAULTS.get(error.field)
            logger.warning(f"Using default value for {error.field}: {default}")
            return default
        return None

class RetryRecovery(RecoveryStrategy):
    """Recover by retrying the operation."""
    
    def __init__(self, max_retries: int = 3, delay: float = 0.1):
        self.max_retries = max_retries
        self.delay = delay
    
    def can_recover(self, error: STUNIRError) -> bool:
        # Can retry transient errors
        return error.code in ["IO_ERROR", "NETWORK_ERROR"]
    
    def recover(self, error: STUNIRError, context: Dict[str, Any]) -> Any:
        import time
        
        operation = context.get("operation")
        if not operation:
            return None
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Retry attempt {attempt + 1}/{self.max_retries}")
                return operation()
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.delay * (attempt + 1))
        
        return None

# =============================================================================
# Error Handler
# =============================================================================

class ErrorHandler:
    """Centralized error handling."""
    
    def __init__(self):
        self.strategies: List[RecoveryStrategy] = [
            DefaultValueRecovery(),
            RetryRecovery()
        ]
        self.report = ErrorReport()
    
    def handle(self, error: STUNIRError, context: Optional[Dict[str, Any]] = None) -> Any:
        """Handle an error, attempting recovery if possible."""
        context = context or {}
        
        # Log the error
        self.report.add_error(error)
        
        # Try recovery strategies
        for strategy in self.strategies:
            if strategy.can_recover(error):
                try:
                    result = strategy.recover(error, context)
                    if result is not None:
                        logger.info(f"Recovered from {error.code}")
                        return result
                except Exception as e:
                    logger.warning(f"Recovery failed: {e}")
        
        # No recovery possible
        return None

# =============================================================================
# Demo Functions
# =============================================================================

def demonstrate_validation_errors():
    """Demonstrate validation error handling."""
    print("\n📋 Demonstrating Validation Errors")
    print("-" * 40)
    
    # Invalid spec with multiple issues
    invalid_spec = {
        "name": "123invalid",  # Invalid identifier
        # Missing version
        "functions": [
            {"name": "add"},  # Missing params and returns
            {"name": "add"},  # Duplicate name
            {"params": [{"name": "x"}]},  # Missing function name
        ],
        "exports": ["nonexistent"]  # References missing function
    }
    
    validator = SpecValidator()
    is_valid, report = validator.validate(invalid_spec)
    
    print(f"\nValidation result: {'✅ Valid' if is_valid else '❌ Invalid'}")
    print(f"Errors: {len(report.errors)}")
    print(f"Warnings: {len(report.warnings)}")
    
    if report.errors:
        print("\nErrors found:")
        for error in report.errors:
            print(f"  - [{error.code}] {error}")
    
    if report.warnings:
        print("\nWarnings:")
        for warning in report.warnings:
            print(f"  - [{warning.code}] {warning}")

def demonstrate_exception_hierarchy():
    """Demonstrate exception hierarchy."""
    print("\n📋 Demonstrating Exception Hierarchy")
    print("-" * 40)
    
    errors = [
        ValidationError("Missing required field", "name"),
        ParseError("Unexpected token", line=42, column=15),
        IRGenerationError("Cannot generate IR", "test_module", "broken_func"),
        HashMismatchError("abc123", "def456", "test.dcbor"),
        DeterminismError("Non-deterministic output", ["hash1", "hash2", "hash3"])
    ]
    
    for error in errors:
        print(f"\n{error.__class__.__name__}:")
        print(f"  Code: {error.code}")
        print(f"  Message: {error}")
        print(f"  Details: {error.details}")
        
        # All inherit from STUNIRError
        assert isinstance(error, STUNIRError)

def demonstrate_recovery():
    """Demonstrate error recovery."""
    print("\n📋 Demonstrating Error Recovery")
    print("-" * 40)
    
    handler = ErrorHandler()
    
    # Missing name - can be recovered with default
    error = ValidationError("Missing required field: name", "name")
    result = handler.handle(error)
    print(f"\nRecovered 'name' field: {result}")
    
    # Unknown error - cannot recover
    error = STUNIRError("Unknown error", code="UNKNOWN")
    result = handler.handle(error)
    print(f"Cannot recover from unknown error: {result}")

# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point for error handling example."""
    print("="*60)
    print("STUNIR Error Handling Example")
    print("="*60)
    
    # Run demonstrations
    demonstrate_exception_hierarchy()
    demonstrate_validation_errors()
    demonstrate_recovery()
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print("""
✅ Demonstrated:
   - Custom exception hierarchy
   - Detailed validation with field-level errors
   - Error collection and reporting
   - Recovery strategies for common errors
   - Structured error serialization (to_dict)

Best Practices:
   1. Use specific exception types for different error categories
   2. Include detailed context in error messages
   3. Collect multiple errors instead of failing on first
   4. Implement recovery strategies where possible
   5. Log errors with appropriate severity levels
""")
    
    print("\n✅ Error handling example completed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
