"""Scientific category parser."""

from typing import Dict
from .base import CategoryParser, ValidationResult, CategoryAST, AnalysisResult, CategoryIR


class ScientificParser(CategoryParser):
    """Parser for scientific specifications."""

    def __init__(self):
        super().__init__("scientific")

    def validate_spec(self, spec: Dict) -> ValidationResult:
        """Validate scientific specification."""
        errors = []
        warnings = []
        return ValidationResult(is_valid=True, errors=errors, warnings=warnings)

    def build_category_ast(self, spec: Dict) -> CategoryAST:
        """Build scientific-specific AST."""
        nodes = [{"type": "scientific", "data": spec}]
        return CategoryAST(
            category=self.category,
            nodes=nodes,
            metadata=spec.get("metadata", {})
        )

    def analyze_category_semantics(self, ast: CategoryAST) -> AnalysisResult:
        """Analyze scientific semantics."""
        annotations = {"category": self.category}
        return AnalysisResult(success=True, annotations=annotations)

    def generate_category_ir(self, ast: CategoryAST) -> CategoryIR:
        """Generate scientific IR."""
        data = {"category": self.category}
        return CategoryIR(category=self.category, data=data)
