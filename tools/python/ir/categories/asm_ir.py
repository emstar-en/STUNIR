"""AsmIr category parser."""

from typing import Dict
from .base import CategoryParser, ValidationResult, CategoryAST, AnalysisResult, CategoryIR


class ASMIRParser(CategoryParser):
    """Parser for asm_ir specifications."""

    def __init__(self):
        super().__init__("asm_ir")

    def validate_spec(self, spec: Dict) -> ValidationResult:
        """Validate asm_ir specification."""
        errors = []
        warnings = []
        return ValidationResult(is_valid=True, errors=errors, warnings=warnings)

    def build_category_ast(self, spec: Dict) -> CategoryAST:
        """Build asm_ir-specific AST."""
        nodes = [{"type": "asm_ir", "data": spec}]
        return CategoryAST(
            category=self.category,
            nodes=nodes,
            metadata=spec.get("metadata", {})
        )

    def analyze_category_semantics(self, ast: CategoryAST) -> AnalysisResult:
        """Analyze asm_ir semantics."""
        annotations = {"category": self.category}
        return AnalysisResult(success=True, annotations=annotations)

    def generate_category_ir(self, ast: CategoryAST) -> CategoryIR:
        """Generate asm_ir IR."""
        data = {"category": self.category}
        return CategoryIR(category=self.category, data=data)
