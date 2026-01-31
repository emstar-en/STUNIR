"""Asp category parser."""

from typing import Dict
from .base import CategoryParser, ValidationResult, CategoryAST, AnalysisResult, CategoryIR


class ASPParser(CategoryParser):
    """Parser for asp specifications."""

    def __init__(self):
        super().__init__("asp")

    def validate_spec(self, spec: Dict) -> ValidationResult:
        """Validate asp specification."""
        errors = []
        warnings = []
        return ValidationResult(is_valid=True, errors=errors, warnings=warnings)

    def build_category_ast(self, spec: Dict) -> CategoryAST:
        """Build asp-specific AST."""
        nodes = [{"type": "asp", "data": spec}]
        return CategoryAST(
            category=self.category,
            nodes=nodes,
            metadata=spec.get("metadata", {})
        )

    def analyze_category_semantics(self, ast: CategoryAST) -> AnalysisResult:
        """Analyze asp semantics."""
        annotations = {"category": self.category}
        return AnalysisResult(success=True, annotations=annotations)

    def generate_category_ir(self, ast: CategoryAST) -> CategoryIR:
        """Generate asp IR."""
        data = {"category": self.category}
        return CategoryIR(category=self.category, data=data)
