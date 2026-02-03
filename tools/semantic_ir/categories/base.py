"""Base class for category-specific parsers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any


@dataclass
class ValidationResult:
    """Result of category validation."""
    is_valid: bool
    errors: List[str] = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


@dataclass
class CategoryAST:
    """Category-specific AST."""
    category: str
    nodes: List[Any]
    metadata: Dict[str, Any]


@dataclass
class AnalysisResult:
    """Result of semantic analysis."""
    success: bool
    annotations: Dict[str, Any]
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


@dataclass
class CategoryIR:
    """Category-specific IR."""
    category: str
    data: Dict[str, Any]


class CategoryParser(ABC):
    """Base class for all category-specific parsers."""

    def __init__(self, category: str):
        self.category = category

    @abstractmethod
    def validate_spec(self, spec: Dict) -> ValidationResult:
        """Validate category-specific specification.
        
        Args:
            spec: Specification dictionary
            
        Returns:
            Validation result with errors/warnings
        """
        pass

    @abstractmethod
    def build_category_ast(self, spec: Dict) -> CategoryAST:
        """Build category-specific AST from specification.
        
        Args:
            spec: Specification dictionary
            
        Returns:
            Category-specific AST
        """
        pass

    @abstractmethod
    def analyze_category_semantics(self, ast: CategoryAST) -> AnalysisResult:
        """Perform category-specific semantic analysis.
        
        Args:
            ast: Category-specific AST
            
        Returns:
            Analysis result with annotations
        """
        pass

    @abstractmethod
    def generate_category_ir(self, ast: CategoryAST) -> CategoryIR:
        """Generate category-specific IR.
        
        Args:
            ast: Category-specific AST
            
        Returns:
            Category-specific IR
        """
        pass
