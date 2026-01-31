"""Parser category parser (for parser generators)."""

from typing import Dict
from .base import CategoryParser, ValidationResult, CategoryAST, AnalysisResult, CategoryIR


class ParserParser(CategoryParser):
    """Parser for parser generator specifications.
    
    Supports: Yacc, Bison, ANTLR, custom parsers
    """

    def __init__(self):
        super().__init__("parser")

    def validate_spec(self, spec: Dict) -> ValidationResult:
        """Validate parser specification."""
        errors = []
        warnings = []
        
        if "grammar" not in spec and "rules" not in spec:
            warnings.append("No grammar or rules defined in parser specification")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    def build_category_ast(self, spec: Dict) -> CategoryAST:
        """Build parser-specific AST."""
        nodes = []
        
        if "grammar" in spec:
            nodes.append({"type": "grammar", "data": spec["grammar"]})
        
        if "rules" in spec:
            nodes.append({"type": "rules", "data": spec["rules"]})
        
        if "productions" in spec:
            nodes.append({"type": "productions", "data": spec["productions"]})
        
        return CategoryAST(
            category=self.category,
            nodes=nodes,
            metadata=spec.get("metadata", {})
        )

    def analyze_category_semantics(self, ast: CategoryAST) -> AnalysisResult:
        """Analyze parser semantics."""
        annotations = {"category": self.category}
        return AnalysisResult(success=True, annotations=annotations)

    def generate_category_ir(self, ast: CategoryAST) -> CategoryIR:
        """Generate parser IR."""
        data = {"category": self.category}
        return CategoryIR(category=self.category, data=data)
