"""WebAssembly category parser."""

from typing import Dict
from .base import CategoryParser, ValidationResult, CategoryAST, AnalysisResult, CategoryIR


class WASMParser(CategoryParser):
    """Parser for WebAssembly specifications.
    
    Supports: WASM MVP, WASM with SIMD, WASM with threads, etc.
    """

    def __init__(self):
        super().__init__("wasm")

    def validate_spec(self, spec: Dict) -> ValidationResult:
        """Validate WASM specification."""
        errors = []
        warnings = []
        
        # Check for WASM version
        if "wasm_version" not in spec.get("metadata", {}):
            warnings.append("Missing wasm_version in metadata, defaulting to MVP")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    def build_category_ast(self, spec: Dict) -> CategoryAST:
        """Build WASM-specific AST."""
        nodes = [{"type": "wasm_module", "data": spec}]
        return CategoryAST(
            category=self.category,
            nodes=nodes,
            metadata=spec.get("metadata", {})
        )

    def analyze_category_semantics(self, ast: CategoryAST) -> AnalysisResult:
        """Analyze WASM semantics."""
        annotations = {"wasm_version": ast.metadata.get("wasm_version", "mvp")}
        return AnalysisResult(success=True, annotations=annotations)

    def generate_category_ir(self, ast: CategoryAST) -> CategoryIR:
        """Generate WASM IR."""
        data = {"wasm_version": ast.metadata.get("wasm_version", "mvp")}
        return CategoryIR(category=self.category, data=data)
