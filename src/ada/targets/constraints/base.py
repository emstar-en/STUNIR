"""Base emitter class for constraint programming.

This module provides the base class for constraint emitters
(MiniZinc, CHR, etc.).
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import hashlib
import json

from ir.constraints import (
    ConstraintModel, Variable, ArrayVariable, Domain, Parameter,
    Constraint, RelationalConstraint, LogicalConstraint, GlobalConstraint,
    Objective, SearchAnnotation, ConstraintEmitterResult,
    Expression, VariableRef, Literal, ArrayAccess, BinaryOp, UnaryOp,
    FunctionCall, SetLiteral, Comprehension,
    ConstraintType, ObjectiveType, VariableType, DomainType
)


def canonical_json(obj: Any) -> str:
    """Generate canonical JSON string.
    
    Args:
        obj: Object to serialize
        
    Returns:
        Canonical JSON string with sorted keys
    """
    return json.dumps(obj, sort_keys=True, separators=(',', ':'))


def compute_sha256(data: bytes) -> str:
    """Compute SHA256 hash of data.
    
    Args:
        data: Bytes to hash
        
    Returns:
        Hex-encoded SHA256 hash
    """
    return hashlib.sha256(data).hexdigest()


class BaseConstraintEmitter(ABC):
    """Abstract base class for constraint emitters.
    
    Provides common functionality for emitting constraint
    languages like MiniZinc and CHR.
    """
    
    DIALECT: str = "base"
    VERSION: str = "1.0"
    FILE_EXTENSION: str = ".txt"
    
    def __init__(self, pretty_print: bool = True):
        """Initialize the emitter.
        
        Args:
            pretty_print: Whether to format output with indentation
        """
        self.pretty_print = pretty_print
        self._indent = 0
        self._output: List[str] = []
        self._warnings: List[str] = []
    
    @abstractmethod
    def emit(self, model: ConstraintModel) -> ConstraintEmitterResult:
        """Emit code from a ConstraintModel.
        
        Args:
            model: The constraint model to emit
            
        Returns:
            ConstraintEmitterResult with code and manifest
        """
        pass
    
    @abstractmethod
    def _emit_variable(self, var: Variable) -> None:
        """Emit a variable declaration.
        
        Args:
            var: Variable to emit
        """
        pass
    
    @abstractmethod
    def _emit_constraint(self, constraint: Constraint) -> None:
        """Emit a constraint.
        
        Args:
            constraint: Constraint to emit
        """
        pass
    
    @abstractmethod
    def _emit_objective(self, objective: Objective) -> None:
        """Emit the objective function.
        
        Args:
            objective: Objective to emit
        """
        pass
    
    def _line(self, text: str = "") -> None:
        """Add a line to output with current indentation.
        
        Args:
            text: Line text
        """
        if text:
            self._output.append("  " * self._indent + text)
        else:
            self._output.append("")
    
    def _indent_inc(self) -> None:
        """Increase indentation level."""
        self._indent += 1
    
    def _indent_dec(self) -> None:
        """Decrease indentation level."""
        self._indent = max(0, self._indent - 1)
    
    def _get_code(self) -> str:
        """Get accumulated code.
        
        Returns:
            Complete code string
        """
        return "\n".join(self._output)
    
    def _warn(self, message: str) -> None:
        """Add a warning.
        
        Args:
            message: Warning message
        """
        self._warnings.append(message)
    
    def _generate_manifest(self, model: ConstraintModel, code: str) -> Dict[str, Any]:
        """Generate build manifest.
        
        Args:
            model: The constraint model
            code: Generated code
            
        Returns:
            Manifest dictionary
        """
        code_bytes = code.encode('utf-8')
        return {
            "schema": f"stunir.manifest.constraints.{self.DIALECT}.v1",
            "generator": f"stunir.constraints.{self.DIALECT}_emitter",
            "version": self.VERSION,
            "model_name": model.name,
            "statistics": {
                "variables": len(model.variables),
                "arrays": len(model.arrays),
                "parameters": len(model.parameters),
                "constraints": len(model.constraints),
            },
            "output": {
                "hash": compute_sha256(code_bytes),
                "size": len(code_bytes),
                "format": self.DIALECT,
            },
            "warnings": self._warnings,
        }
    
    def _format_expression(self, expr: Expression) -> str:
        """Format an expression.
        
        Args:
            expr: Expression to format
            
        Returns:
            Formatted expression string
        """
        if isinstance(expr, VariableRef):
            return expr.name
        elif isinstance(expr, Literal):
            if isinstance(expr.value, bool):
                return 'true' if expr.value else 'false'
            return str(expr.value)
        elif isinstance(expr, ArrayAccess):
            indices = ", ".join(self._format_expression(i) for i in expr.indices)
            return f"{expr.array}[{indices}]"
        elif isinstance(expr, BinaryOp):
            left = self._format_expression(expr.left)
            right = self._format_expression(expr.right)
            if expr.op in ('min', 'max', 'div', 'mod'):
                return f"{expr.op}({left}, {right})"
            return f"({left} {expr.op} {right})"
        elif isinstance(expr, UnaryOp):
            operand = self._format_expression(expr.operand)
            if expr.op == '-':
                return f"-{operand}"
            return f"{expr.op}({operand})"
        elif isinstance(expr, FunctionCall):
            args = ", ".join(self._format_expression(a) for a in expr.args)
            return f"{expr.name}({args})"
        elif isinstance(expr, SetLiteral):
            elems = ", ".join(self._format_expression(e) for e in expr.elements)
            return "{" + elems + "}"
        elif isinstance(expr, Comprehension):
            expr_str = self._format_expression(expr.expr)
            gens = ", ".join(f"{v} in {lb}..{ub}" for v, lb, ub in expr.generators)
            if expr.condition:
                cond = self._format_expression(expr.condition)
                return f"[{expr_str} | {gens} where {cond}]"
            return f"[{expr_str} | {gens}]"
        else:
            return str(expr)
