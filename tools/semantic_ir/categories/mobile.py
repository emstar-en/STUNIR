"""Mobile category parser."""

from typing import Dict
from .base import CategoryParser, ValidationResult, CategoryAST, AnalysisResult, CategoryIR


class MobileParser(CategoryParser):
    """Parser for mobile specifications."""

    def __init__(self):
        super().__init__("mobile")

    def validate_spec(self, spec: Dict) -> ValidationResult:
        """Validate mobile specification."""
        errors = []
        warnings = []
        return ValidationResult(is_valid=True, errors=errors, warnings=warnings)

    def build_category_ast(self, spec: Dict) -> CategoryAST:
        """Build mobile-specific AST."""
        nodes = [{"type": "mobile", "data": spec}]
        return CategoryAST(
            category=self.category,
            nodes=nodes,
            metadata=spec.get("metadata", {})
        )

    def analyze_category_semantics(self, ast: CategoryAST) -> AnalysisResult:
        """Analyze mobile semantics."""
        annotations = {"category": self.category}
        return AnalysisResult(success=True, annotations=annotations)

    def generate_category_ir(self, ast: CategoryAST) -> CategoryIR:
        """Generate mobile IR."""
        data = {"category": self.category}
        return CategoryIR(category=self.category, data=data)
