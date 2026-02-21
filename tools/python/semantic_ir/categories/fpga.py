"""Fpga category parser."""

from typing import Dict
from .base import CategoryParser, ValidationResult, CategoryAST, AnalysisResult, CategoryIR


class FPGAParser(CategoryParser):
    """Parser for fpga specifications."""

    def __init__(self):
        super().__init__("fpga")

    def validate_spec(self, spec: Dict) -> ValidationResult:
        """Validate fpga specification."""
        errors = []
        warnings = []
        return ValidationResult(is_valid=True, errors=errors, warnings=warnings)

    def build_category_ast(self, spec: Dict) -> CategoryAST:
        """Build fpga-specific AST."""
        nodes = [{"type": "fpga", "data": spec}]
        return CategoryAST(
            category=self.category,
            nodes=nodes,
            metadata=spec.get("metadata", {})
        )

    def analyze_category_semantics(self, ast: CategoryAST) -> AnalysisResult:
        """Analyze fpga semantics."""
        annotations = {"category": self.category}
        return AnalysisResult(success=True, annotations=annotations)

    def generate_category_ir(self, ast: CategoryAST) -> CategoryIR:
        """Generate fpga IR."""
        data = {"category": self.category}
        return CategoryIR(category=self.category, data=data)
