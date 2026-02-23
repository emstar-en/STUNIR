"""Assembly language category parser."""

from typing import Dict
from .base import CategoryParser, ValidationResult, CategoryAST, AnalysisResult, CategoryIR


class AssemblyParser(CategoryParser):
    """Parser for assembly language specifications.
    
    Supports: x86, x86-64, ARM, MIPS, RISC-V, etc.
    """

    def __init__(self):
        super().__init__("assembly")

    def validate_spec(self, spec: Dict) -> ValidationResult:
        """Validate assembly specification."""
        errors = []
        warnings = []
        
        # Check for instruction set
        if "instruction_set" not in spec.get("metadata", {}):
            errors.append("Missing instruction_set in metadata")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    def build_category_ast(self, spec: Dict) -> CategoryAST:
        """Build assembly-specific AST."""
        nodes = []
        
        if "instructions" in spec:
            nodes.append({"type": "instructions", "data": spec["instructions"]})
        
        if "registers" in spec:
            nodes.append({"type": "registers", "data": spec["registers"]})
        
        return CategoryAST(
            category=self.category,
            nodes=nodes,
            metadata=spec.get("metadata", {})
        )

    def analyze_category_semantics(self, ast: CategoryAST) -> AnalysisResult:
        """Analyze assembly semantics."""
        annotations = {"instruction_set": ast.metadata.get("instruction_set", "unknown")}
        return AnalysisResult(success=True, annotations=annotations)

    def generate_category_ir(self, ast: CategoryAST) -> CategoryIR:
        """Generate assembly IR."""
        data = {"instruction_set": ast.metadata.get("instruction_set", "unknown")}
        return CategoryIR(category=self.category, data=data)
