"""Lisp family category parser."""

from typing import Dict
from .base import CategoryParser, ValidationResult, CategoryAST, AnalysisResult, CategoryIR


class LispParser(CategoryParser):
    """Parser for Lisp family specifications.
    
    Supports all 8 Lisp dialects:
    - Common Lisp
    - Scheme (R5RS, R6RS, R7RS)
    - Clojure
    - Racket
    - Emacs Lisp
    - Guile
    - Hy (Python interop)
    - Janet
    """

    SUPPORTED_DIALECTS = [
        "common-lisp", "scheme", "clojure", "racket",
        "emacs-lisp", "guile", "hy", "janet"
    ]

    def __init__(self):
        super().__init__("lisp")

    def validate_spec(self, spec: Dict) -> ValidationResult:
        """Validate Lisp specification."""
        errors = []
        warnings = []
        
        # Check for dialect
        dialect = spec.get("metadata", {}).get("dialect", "")
        if not dialect:
            errors.append("Missing dialect in metadata")
        elif dialect not in self.SUPPORTED_DIALECTS:
            errors.append(f"Unsupported Lisp dialect: {dialect}")
        
        # Check for S-expressions
        if "forms" not in spec and "functions" not in spec:
            warnings.append("No forms or functions defined")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    def build_category_ast(self, spec: Dict) -> CategoryAST:
        """Build Lisp-specific AST."""
        nodes = []
        
        # Parse forms (S-expressions)
        if "forms" in spec:
            for form in spec["forms"]:
                nodes.append({"type": "form", "data": form})
        
        # Parse macros
        if "macros" in spec:
            for macro in spec["macros"]:
                nodes.append({"type": "macro", "data": macro})
        
        # Parse packages (for Common Lisp)
        if "packages" in spec:
            for package in spec["packages"]:
                nodes.append({"type": "package", "data": package})
        
        return CategoryAST(
            category=self.category,
            nodes=nodes,
            metadata=spec.get("metadata", {})
        )

    def analyze_category_semantics(self, ast: CategoryAST) -> AnalysisResult:
        """Analyze Lisp semantics."""
        annotations = {
            "dialect": ast.metadata.get("dialect", "unknown"),
            "has_macros": any(node["type"] == "macro" for node in ast.nodes),
        }
        return AnalysisResult(success=True, annotations=annotations)

    def generate_category_ir(self, ast: CategoryAST) -> CategoryIR:
        """Generate Lisp IR."""
        data = {
            "dialect": ast.metadata.get("dialect", "unknown"),
            "form_count": len([n for n in ast.nodes if n["type"] == "form"]),
            "macro_count": len([n for n in ast.nodes if n["type"] == "macro"]),
        }
        return CategoryIR(category=self.category, data=data)
