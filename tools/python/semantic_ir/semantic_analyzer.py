"""Semantic Analyzer for Semantic IR parser."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .types import (
    ParseError,
    ErrorType,
    ErrorSeverity,
    Type,
    Function,
    Expression,
    Statement,
)
from .ast_builder import AST, ASTNode


@dataclass
class AnnotatedAST:
    """AST with semantic annotations."""
    ast: AST
    type_annotations: Dict[int, Type] = field(default_factory=dict)
    control_flow_graph: Dict[str, List[str]] = field(default_factory=dict)
    data_flow: Dict[str, Set[str]] = field(default_factory=dict)
    warnings: List[ParseError] = field(default_factory=list)


@dataclass
class TypeCheckResult:
    """Result of type checking."""
    success: bool
    inferred_type: Optional[Type] = None
    errors: List[ParseError] = field(default_factory=list)


@dataclass
class CFGResult:
    """Control flow graph analysis result."""
    reachable_blocks: Set[str]
    unreachable_blocks: Set[str]
    missing_returns: List[str]


class SemanticAnalyzer:
    """Performs semantic analysis on AST."""

    def __init__(self, category: str):
        self.category = category
        self.errors: List[ParseError] = []
        self.warnings: List[ParseError] = []
        self.type_context: Dict[str, Type] = {}

    def analyze(self, ast: AST) -> AnnotatedAST:
        """Perform full semantic analysis on AST."""
        annotated = AnnotatedAST(ast=ast)
        
        # Build type context from type definitions
        for typedef in ast.types:
            self.type_context[typedef.name] = typedef.type
        
        # Analyze functions
        for func in ast.functions:
            self._analyze_function(func, annotated)
        
        # Check for undefined references
        self._check_undefined_references(ast)
        
        # Validate control flow
        for func in ast.functions:
            cfg_result = self.validate_control_flow(func)
            annotated.control_flow_graph[func.name] = list(cfg_result.reachable_blocks)
            
            # Add warnings for unreachable code
            for block in cfg_result.unreachable_blocks:
                self.warnings.append(ParseError(
                    location=func.location,
                    error_type=ErrorType.SEMANTIC,
                    message=f"Unreachable code in function '{func.name}'",
                    severity=ErrorSeverity.WARNING,
                ))
        
        annotated.warnings = self.warnings
        return annotated

    def check_types(self, node: ASTNode) -> TypeCheckResult:
        """Check types for a node."""
        # Simplified type checking
        return TypeCheckResult(success=True)

    def infer_type(self, expr: Expression) -> Type:
        """Infer type of expression."""
        if expr.type:
            return expr.type
        
        # Type inference based on expression kind
        if expr.kind == "literal":
            if isinstance(expr.value, int):
                return Type(name="i32", is_primitive=True)
            elif isinstance(expr.value, float):
                return Type(name="f64", is_primitive=True)
            elif isinstance(expr.value, str):
                return Type(name="string", is_primitive=True)
            elif isinstance(expr.value, bool):
                return Type(name="bool", is_primitive=True)
        
        elif expr.kind == "variable":
            # Look up variable in type context
            var_name = expr.value
            if var_name in self.type_context:
                return self.type_context[var_name]
        
        # Default to unknown type
        return Type(name="unknown", is_primitive=False)

    def validate_control_flow(self, func: Function) -> CFGResult:
        """Validate control flow in function."""
        reachable = set(["entry"])
        unreachable = set()
        missing_returns = []
        
        # Simplified control flow analysis
        # In a real implementation, this would build a proper CFG
        
        # Check if non-void function has return statement
        if func.return_type.name != "void":
            has_return = any(stmt.kind == "return" for stmt in func.body)
            if not has_return:
                missing_returns.append(func.name)
                self.errors.append(ParseError(
                    location=func.location,
                    error_type=ErrorType.SEMANTIC,
                    message=f"Function '{func.name}' must return a value",
                ))
        
        return CFGResult(
            reachable_blocks=reachable,
            unreachable_blocks=unreachable,
            missing_returns=missing_returns,
        )

    def _analyze_function(self, func: Function, annotated: AnnotatedAST):
        """Analyze a single function."""
        # Add parameters to type context
        for param in func.parameters:
            self.type_context[param.name] = param.type
        
        # Analyze function body
        for stmt in func.body:
            self._analyze_statement(stmt, annotated)
        
        # Remove parameters from type context
        for param in func.parameters:
            self.type_context.pop(param.name, None)

    def _analyze_statement(self, stmt: Statement, annotated: AnnotatedAST):
        """Analyze a single statement."""
        # Type check expressions in statement
        for expr in stmt.expressions:
            inferred_type = self.infer_type(expr)
            if expr.location:
                # Store type annotation by expression ID (using location as proxy)
                annotated.type_annotations[id(expr)] = inferred_type
        
        # Recursively analyze nested statements
        for nested_stmt in stmt.statements:
            self._analyze_statement(nested_stmt, annotated)

    def _check_undefined_references(self, ast: AST):
        """Check for undefined references."""
        # This would check that all referenced symbols are defined
        # Simplified for now
        pass
