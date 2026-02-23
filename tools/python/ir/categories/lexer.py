"""Lexer category parser."""

from typing import Dict
from .base import CategoryParser, ValidationResult, CategoryAST, AnalysisResult, CategoryIR


class LexerParser(CategoryParser):
    """Parser for lexer specifications.
    
    Supports: Flex, Lex, custom lexers
    """

    def __init__(self):
        super().__init__("lexer")

    def validate_spec(self, spec: Dict) -> ValidationResult:
        """Validate lexer specification."""
        errors = []
        warnings = []
        
        if "tokens" not in spec:
            warnings.append("No tokens defined in lexer specification")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    def build_category_ast(self, spec: Dict) -> CategoryAST:
        """Build lexer-specific AST."""
        nodes = []
        
        if "tokens" in spec:
            nodes.append({"type": "tokens", "data": spec["tokens"]})
        
        if "rules" in spec:
            nodes.append({"type": "rules", "data": spec["rules"]})
        
        return CategoryAST(
            category=self.category,
            nodes=nodes,
            metadata=spec.get("metadata", {})
        )

    def analyze_category_semantics(self, ast: CategoryAST) -> AnalysisResult:
        """Analyze lexer semantics."""
        annotations = {"category": self.category}
        return AnalysisResult(success=True, annotations=annotations)

    def generate_category_ir(self, ast: CategoryAST) -> CategoryIR:
        """Generate lexer IR."""
        data = {"category": self.category}
        return CategoryIR(category=self.category, data=data)
