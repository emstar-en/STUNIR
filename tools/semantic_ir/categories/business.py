"""Business language category parser."""

from typing import Dict
from .base import CategoryParser, ValidationResult, CategoryAST, AnalysisResult, CategoryIR


class BusinessParser(CategoryParser):
    """Parser for business language specifications.
    
    Supports: COBOL, BASIC, RPG, etc.
    """

    def __init__(self):
        super().__init__("business")

    def validate_spec(self, spec: Dict) -> ValidationResult:
        return ValidationResult(is_valid=True)

    def build_category_ast(self, spec: Dict) -> CategoryAST:
        nodes = [{"type": "business", "data": spec}]
        return CategoryAST(category=self.category, nodes=nodes, metadata=spec.get("metadata", {}))

    def analyze_category_semantics(self, ast: CategoryAST) -> AnalysisResult:
        return AnalysisResult(success=True, annotations={})

    def generate_category_ir(self, ast: CategoryAST) -> CategoryIR:
        return CategoryIR(category=self.category, data={})
