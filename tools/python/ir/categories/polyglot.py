"""Polyglot category parser."""

from typing import Dict
from .base import CategoryParser, ValidationResult, CategoryAST, AnalysisResult, CategoryIR


class PolyglotParser(CategoryParser):
    """Parser for polyglot specifications.
    
    Supports: C89, C99, C11, C17, C23, Rust, Zig, etc.
    """

    def __init__(self):
        super().__init__("polyglot")

    def validate_spec(self, spec: Dict) -> ValidationResult:
        """Validate polyglot specification."""
        errors = []
        warnings = []
        
        # Check for target languages
        if "target_languages" not in spec.get("metadata", {}):
            warnings.append("Missing target_languages in metadata")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    def build_category_ast(self, spec: Dict) -> CategoryAST:
        """Build polyglot-specific AST."""
        nodes = [{"type": "polyglot", "data": spec}]
        return CategoryAST(
            category=self.category,
            nodes=nodes,
            metadata=spec.get("metadata", {})
        )

    def analyze_category_semantics(self, ast: CategoryAST) -> AnalysisResult:
        """Analyze polyglot semantics."""
        annotations = {"languages": ast.metadata.get("target_languages", [])}
        return AnalysisResult(success=True, annotations=annotations)

    def generate_category_ir(self, ast: CategoryAST) -> CategoryIR:
        """Generate polyglot IR."""
        data = {"target_languages": ast.metadata.get("target_languages", [])}
        return CategoryIR(category=self.category, data=data)
