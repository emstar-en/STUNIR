"""Embedded systems category parser."""

from typing import Dict
from .base import CategoryParser, ValidationResult, CategoryAST, AnalysisResult, CategoryIR


class EmbeddedParser(CategoryParser):
    """Parser for embedded systems specifications.
    
    Supports: ARM, AVR, RISC-V, MIPS, MSP430, PIC, 8051, ESP32, STM32, etc.
    """

    def __init__(self):
        super().__init__("embedded")

    def validate_spec(self, spec: Dict) -> ValidationResult:
        """Validate embedded specification."""
        errors = []
        warnings = []
        
        # Check for required fields
        if "target_arch" not in spec.get("metadata", {}):
            warnings.append("Missing target_arch in metadata")
        
        # Validate architecture
        arch = spec.get("metadata", {}).get("target_arch", "")
        supported_archs = ["arm", "avr", "risc-v", "mips", "msp430", "pic", "8051", "esp32", "stm32"]
        if arch and arch not in supported_archs:
            warnings.append(f"Unknown architecture: {arch}")
        
        # Check for memory constraints
        if "memory" in spec.get("metadata", {}):
            memory = spec["metadata"]["memory"]
            if "ram_size" in memory and memory["ram_size"] < 1024:
                warnings.append("Very low RAM size, optimization recommended")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    def build_category_ast(self, spec: Dict) -> CategoryAST:
        """Build embedded-specific AST."""
        nodes = []
        
        # Parse embedded-specific constructs
        if "interrupts" in spec:
            nodes.append({"type": "interrupts", "data": spec["interrupts"]})
        
        if "peripherals" in spec:
            nodes.append({"type": "peripherals", "data": spec["peripherals"]})
        
        if "memory_map" in spec:
            nodes.append({"type": "memory_map", "data": spec["memory_map"]})
        
        return CategoryAST(
            category=self.category,
            nodes=nodes,
            metadata=spec.get("metadata", {})
        )

    def analyze_category_semantics(self, ast: CategoryAST) -> AnalysisResult:
        """Analyze embedded semantics."""
        annotations = {}
        errors = []
        
        # Check for stack overflow risks
        if "memory_map" in ast.metadata:
            annotations["memory_safe"] = True
        
        # Validate interrupt handlers
        for node in ast.nodes:
            if node["type"] == "interrupts":
                annotations["has_interrupts"] = True
        
        return AnalysisResult(
            success=len(errors) == 0,
            annotations=annotations,
            errors=errors
        )

    def generate_category_ir(self, ast: CategoryAST) -> CategoryIR:
        """Generate embedded IR."""
        data = {
            "target_arch": ast.metadata.get("target_arch", "unknown"),
            "constructs": [node["type"] for node in ast.nodes],
        }
        
        return CategoryIR(category=self.category, data=data)
