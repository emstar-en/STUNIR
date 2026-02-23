"""Bytecode category parser."""

from typing import Dict
from .base import CategoryParser, ValidationResult, CategoryAST, AnalysisResult, CategoryIR


class BytecodeParser(CategoryParser):
    """Parser for bytecode specifications."""

    def __init__(self):
        super().__init__("bytecode")

    def validate_spec(self, spec: Dict) -> ValidationResult:
        """Validate bytecode specification."""
        errors = []
        warnings = []
        return ValidationResult(is_valid=True, errors=errors, warnings=warnings)

    def build_category_ast(self, spec: Dict) -> CategoryAST:
        """Build bytecode-specific AST."""
        nodes = [{"type": "bytecode", "data": spec}]
        return CategoryAST(
            category=self.category,
            nodes=nodes,
            metadata=spec.get("metadata", {})
        )

    def analyze_category_semantics(self, ast: CategoryAST) -> AnalysisResult:
        """Analyze bytecode semantics."""
        annotations = {"category": self.category}
        return AnalysisResult(success=True, annotations=annotations)

    def generate_category_ir(self, ast: CategoryAST) -> CategoryIR:
        """Generate bytecode IR."""
        data = {"category": self.category}
        return CategoryIR(category=self.category, data=data)
