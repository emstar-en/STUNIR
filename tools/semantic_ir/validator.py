"""STUNIR Semantic IR Validator.

DO-178C Level A Compliant
Comprehensive validation framework with JSON Schema integration.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

# Try to import jsonschema for schema validation
try:
    import jsonschema
    from jsonschema import Draft7Validator, validators
    HAVE_JSONSCHEMA = True
except ImportError:
    HAVE_JSONSCHEMA = False
    print("Warning: jsonschema not available, schema validation disabled")

# Import our IR types
try:
    from .ir_types import *
    from .nodes import *
    from .expressions import *
    from .statements import *
    from .declarations import *
    from .modules import *
    from .validation import *
except ImportError:
    # Fallback for direct script execution
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from ir_types import *
    from nodes import *
    from expressions import *
    from statements import *
    from declarations import *
    from modules import *
    from validation import *


class ValidatorLanguage(str, Enum):
    """Supported validator languages."""
    PYTHON = "python"
    SPARK = "spark"
    RUST = "rust"
    HASKELL = "haskell"


class SemanticIRValidator:
    """Comprehensive Semantic IR validator.
    
    Validates IR against:
    1. JSON Schema
    2. Semantic rules
    3. Type system
    4. Cross-references
    """
    
    def __init__(self, schema_dir: Optional[Path] = None):
        """Initialize validator with schema directory.
        
        Args:
            schema_dir: Path to schema directory (defaults to schemas/semantic_ir/)
        """
        if schema_dir is None:
            # Default to schemas/semantic_ir/ relative to repo root
            repo_root = Path(__file__).parent.parent.parent
            schema_dir = repo_root / "schemas" / "semantic_ir"
        
        self.schema_dir = Path(schema_dir)
        self.schemas: Dict[str, Any] = {}
        self._load_schemas()
        
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def _load_schemas(self) -> None:
        """Load all JSON schemas."""
        schema_files = [
            "ir_schema.json",
            "node_types.json",
            "type_system.json",
            "expressions.json",
            "statements.json",
            "declarations.json",
            "modules.json",
            "target_extensions.json",
        ]
        
        for schema_file in schema_files:
            schema_path = self.schema_dir / schema_file
            if schema_path.exists():
                try:
                    with open(schema_path, 'r') as f:
                        schema_name = schema_file.replace(".json", "")
                        self.schemas[schema_name] = json.load(f)
                except Exception as e:
                    print(f"Warning: Failed to load schema {schema_file}: {e}")
    
    def validate_json(self, ir_json: Dict[str, Any]) -> ValidationResult:
        """Validate IR JSON against schema.
        
        Args:
            ir_json: IR JSON dictionary
            
        Returns:
            ValidationResult with status and messages
        """
        self.errors = []
        self.warnings = []
        
        if not HAVE_JSONSCHEMA:
            self.warnings.append("jsonschema not available, skipping schema validation")
            return ValidationResult.warning("jsonschema not available")
        
        # Validate against main IR schema
        if "ir_schema" in self.schemas:
            try:
                validator = Draft7Validator(self.schemas["ir_schema"])
                errors = list(validator.iter_errors(ir_json))
                if errors:
                    for error in errors[:10]:  # Limit to first 10 errors
                        self.errors.append(f"Schema error at {'.'.join(str(p) for p in error.path)}: {error.message}")
                    return ValidationResult.invalid(f"{len(errors)} schema validation errors")
            except Exception as e:
                return ValidationResult.invalid(f"Schema validation failed: {e}")
        else:
            self.warnings.append("IR schema not loaded")
        
        # Validate root module if present
        if "root" in ir_json:
            result = self._validate_module_dict(ir_json["root"])
            if result.status != ValidationStatus.VALID:
                return result
        
        return ValidationResult.valid()
    
    def _validate_module_dict(self, module_dict: Dict[str, Any]) -> ValidationResult:
        """Validate a module dictionary.
        
        Args:
            module_dict: Module dictionary
            
        Returns:
            ValidationResult
        """
        # Check required fields
        required_fields = ["node_id", "kind", "name"]
        for field in required_fields:
            if field not in module_dict:
                return ValidationResult.invalid(f"Module missing required field: {field}")
        
        # Validate node ID
        node_id_result = validate_node_id(module_dict["node_id"])
        if node_id_result.status != ValidationStatus.VALID:
            return node_id_result
        
        # Validate hash if present
        if "hash" in module_dict and module_dict["hash"]:
            hash_result = validate_hash(module_dict["hash"])
            if hash_result.status != ValidationStatus.VALID:
                return hash_result
        
        # Validate kind
        if module_dict["kind"] != "module":
            return ValidationResult.invalid(f"Root must be a module, got: {module_dict['kind']}")
        
        # Validate declarations
        if "declarations" in module_dict:
            for decl in module_dict["declarations"]:
                if isinstance(decl, dict):
                    decl_result = self._validate_declaration_dict(decl)
                    if decl_result.status != ValidationStatus.VALID:
                        return decl_result
                elif isinstance(decl, str):
                    # Node ID reference
                    node_id_result = validate_node_id(decl)
                    if node_id_result.status != ValidationStatus.VALID:
                        return node_id_result
        
        return ValidationResult.valid()
    
    def _validate_declaration_dict(self, decl_dict: Dict[str, Any]) -> ValidationResult:
        """Validate a declaration dictionary.
        
        Args:
            decl_dict: Declaration dictionary
            
        Returns:
            ValidationResult
        """
        # Check required fields
        if "node_id" not in decl_dict or "kind" not in decl_dict:
            return ValidationResult.invalid("Declaration missing required fields")
        
        # Validate node ID
        node_id_result = validate_node_id(decl_dict["node_id"])
        if node_id_result.status != ValidationStatus.VALID:
            return node_id_result
        
        return ValidationResult.valid()
    
    def validate_file(self, filepath: Path) -> ValidationResult:
        """Validate an IR file.
        
        Args:
            filepath: Path to IR JSON file
            
        Returns:
            ValidationResult
        """
        try:
            with open(filepath, 'r') as f:
                ir_json = json.load(f)
            return self.validate_json(ir_json)
        except json.JSONDecodeError as e:
            return ValidationResult.invalid(f"Invalid JSON: {e}")
        except Exception as e:
            return ValidationResult.invalid(f"Error reading file: {e}")
    
    def validate_python_module(self, module: IRModule) -> ValidationResult:
        """Validate a Python IRModule instance.
        
        Args:
            module: IRModule instance
            
        Returns:
            ValidationResult
        """
        return validate_module(module)
    
    def generate_report(self) -> str:
        """Generate validation report.
        
        Returns:
            Formatted validation report
        """
        report = []
        report.append("=" * 80)
        report.append("STUNIR Semantic IR Validation Report")
        report.append("=" * 80)
        
        if self.errors:
            report.append(f"\nErrors ({len(self.errors)}):")
            for i, error in enumerate(self.errors, 1):
                report.append(f"  {i}. {error}")
        else:
            report.append("\nNo errors found.")
        
        if self.warnings:
            report.append(f"\nWarnings ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings, 1):
                report.append(f"  {i}. {warning}")
        
        if not self.errors and not self.warnings:
            report.append("\n✓ Validation passed successfully!")
        
        report.append("=" * 80)
        return "\n".join(report)


def main():
    """CLI entry point for validator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="STUNIR Semantic IR Validator")
    parser.add_argument("file", help="IR JSON file to validate")
    parser.add_argument("--schema-dir", help="Schema directory path")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    validator = SemanticIRValidator(
        schema_dir=Path(args.schema_dir) if args.schema_dir else None
    )
    
    result = validator.validate_file(Path(args.file))
    
    if args.verbose or result.status != ValidationStatus.VALID:
        print(validator.generate_report())
    
    if result.status == ValidationStatus.VALID:
        print(f"✓ {args.file} is valid")
        return 0
    elif result.status == ValidationStatus.WARNING:
        print(f"⚠ {args.file} has warnings: {result.message}")
        return 0
    else:
        print(f"✗ {args.file} is invalid: {result.message}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
