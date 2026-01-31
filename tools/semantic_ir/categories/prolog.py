"""Prolog family category parser."""

from typing import Dict
from .base import CategoryParser, ValidationResult, CategoryAST, AnalysisResult, CategoryIR


class PrologParser(CategoryParser):
    """Parser for Prolog family specifications.
    
    Supports all 8 Prolog dialects:
    - SWI-Prolog
    - GNU Prolog
    - SICStus Prolog
    - YAP (Yet Another Prolog)
    - XSB
    - Ciao Prolog
    - B-Prolog
    - ECLiPSe
    """

    SUPPORTED_DIALECTS = [
        "swi-prolog", "gnu-prolog", "sicstus", "yap",
        "xsb", "ciao", "b-prolog", "eclipse"
    ]

    def __init__(self):
        super().__init__("prolog")

    def validate_spec(self, spec: Dict) -> ValidationResult:
        """Validate Prolog specification."""
        errors = []
        warnings = []
        
        # Check for dialect
        dialect = spec.get("metadata", {}).get("dialect", "")
        if not dialect:
            errors.append("Missing dialect in metadata")
        elif dialect not in self.SUPPORTED_DIALECTS:
            errors.append(f"Unsupported Prolog dialect: {dialect}")
        
        # Check for predicates
        if "predicates" not in spec and "facts" not in spec and "rules" not in spec:
            warnings.append("No predicates, facts, or rules defined")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    def build_category_ast(self, spec: Dict) -> CategoryAST:
        """Build Prolog-specific AST."""
        nodes = []
        
        # Parse facts
        if "facts" in spec:
            for fact in spec["facts"]:
                nodes.append({"type": "fact", "data": fact})
        
        # Parse rules
        if "rules" in spec:
            for rule in spec["rules"]:
                nodes.append({"type": "rule", "data": rule})
        
        # Parse predicates
        if "predicates" in spec:
            for predicate in spec["predicates"]:
                nodes.append({"type": "predicate", "data": predicate})
        
        # Parse queries
        if "queries" in spec:
            for query in spec["queries"]:
                nodes.append({"type": "query", "data": query})
        
        return CategoryAST(
            category=self.category,
            nodes=nodes,
            metadata=spec.get("metadata", {})
        )

    def analyze_category_semantics(self, ast: CategoryAST) -> AnalysisResult:
        """Analyze Prolog semantics."""
        annotations = {
            "dialect": ast.metadata.get("dialect", "unknown"),
            "fact_count": len([n for n in ast.nodes if n["type"] == "fact"]),
            "rule_count": len([n for n in ast.nodes if n["type"] == "rule"]),
        }
        return AnalysisResult(success=True, annotations=annotations)

    def generate_category_ir(self, ast: CategoryAST) -> CategoryIR:
        """Generate Prolog IR."""
        data = {
            "dialect": ast.metadata.get("dialect", "unknown"),
            "fact_count": len([n for n in ast.nodes if n["type"] == "fact"]),
            "rule_count": len([n for n in ast.nodes if n["type"] == "rule"]),
        }
        return CategoryIR(category=self.category, data=data)
