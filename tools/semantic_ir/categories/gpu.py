"""GPU category parser."""

from typing import Dict
from .base import CategoryParser, ValidationResult, CategoryAST, AnalysisResult, CategoryIR


class GPUParser(CategoryParser):
    """Parser for GPU kernel specifications.
    
    Supports: CUDA, ROCm, OpenCL, Metal, Vulkan, SYCL, etc.
    """

    def __init__(self):
        super().__init__("gpu")

    def validate_spec(self, spec: Dict) -> ValidationResult:
        """Validate GPU specification."""
        errors = []
        warnings = []
        
        # Check for GPU platform
        if "gpu_platform" not in spec.get("metadata", {}):
            errors.append("Missing gpu_platform in metadata")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    def build_category_ast(self, spec: Dict) -> CategoryAST:
        """Build GPU-specific AST."""
        nodes = []
        
        if "kernels" in spec:
            nodes.append({"type": "kernels", "data": spec["kernels"]})
        
        if "shared_memory" in spec:
            nodes.append({"type": "shared_memory", "data": spec["shared_memory"]})
        
        return CategoryAST(
            category=self.category,
            nodes=nodes,
            metadata=spec.get("metadata", {})
        )

    def analyze_category_semantics(self, ast: CategoryAST) -> AnalysisResult:
        """Analyze GPU semantics."""
        annotations = {"platform": ast.metadata.get("gpu_platform", "unknown")}
        return AnalysisResult(success=True, annotations=annotations)

    def generate_category_ir(self, ast: CategoryAST) -> CategoryIR:
        """Generate GPU IR."""
        data = {"gpu_platform": ast.metadata.get("gpu_platform", "unknown")}
        return CategoryIR(category=self.category, data=data)
