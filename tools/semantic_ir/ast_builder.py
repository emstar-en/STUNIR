"""AST Builder for Semantic IR parser."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import json

from .types import (
    SourceLocation,
    ParseError,
    ErrorType,
    ErrorSeverity,
    Type,
    Parameter,
    Function,
    TypeDef,
    Constant,
    Import,
    Expression,
    Statement,
)


@dataclass
class ASTNode:
    """Base class for all AST nodes."""
    location: SourceLocation
    children: List['ASTNode'] = field(default_factory=list)


@dataclass
class AST:
    """Abstract Syntax Tree."""
    functions: List[Function] = field(default_factory=list)
    types: List[TypeDef] = field(default_factory=list)
    constants: List[Constant] = field(default_factory=list)
    imports: List[Import] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    symbol_table: Dict[str, ASTNode] = field(default_factory=dict)


class ASTBuilder:
    """Builds Abstract Syntax Tree from specification."""

    def __init__(self, category: str, filename: str = "<input>"):
        self.category = category
        self.filename = filename
        self.errors: List[ParseError] = []
        self.current_line = 1
        self.current_column = 1

    def build_ast(self, spec: Dict) -> AST:
        """Build AST from specification dictionary."""
        ast = AST()
        
        # Extract metadata
        ast.metadata = spec.get("metadata", {})
        
        # Build functions
        for func_spec in spec.get("functions", []):
            try:
                func = self.build_function(func_spec)
                ast.functions.append(func)
                ast.symbol_table[func.name] = func
            except Exception as e:
                self._add_error(
                    ErrorType.SYNTAX,
                    f"Failed to build function: {e}",
                    SourceLocation(self.filename, 0, 0)
                )
        
        # Build types
        for type_spec in spec.get("types", []):
            try:
                typedef = self.build_typedef(type_spec)
                ast.types.append(typedef)
                ast.symbol_table[typedef.name] = typedef
            except Exception as e:
                self._add_error(
                    ErrorType.SYNTAX,
                    f"Failed to build type: {e}",
                    SourceLocation(self.filename, 0, 0)
                )
        
        # Build constants
        for const_spec in spec.get("constants", []):
            try:
                const = self.build_constant(const_spec)
                ast.constants.append(const)
                ast.symbol_table[const.name] = const
            except Exception as e:
                self._add_error(
                    ErrorType.SYNTAX,
                    f"Failed to build constant: {e}",
                    SourceLocation(self.filename, 0, 0)
                )
        
        # Build imports
        for import_spec in spec.get("imports", []):
            try:
                imp = self.build_import(import_spec)
                ast.imports.append(imp)
            except Exception as e:
                self._add_error(
                    ErrorType.SYNTAX,
                    f"Failed to build import: {e}",
                    SourceLocation(self.filename, 0, 0)
                )
        
        return ast

    def build_function(self, func_spec: Dict) -> Function:
        """Build function node from specification."""
        location = self._make_location(func_spec.get("location", {}))
        
        # Build parameters
        parameters = []
        for param_spec in func_spec.get("parameters", []):
            param_type = self.build_type(param_spec.get("type", {}))
            param = Parameter(
                name=param_spec["name"],
                type=param_type,
                location=self._make_location(param_spec.get("location", {}))
            )
            parameters.append(param)
        
        # Build return type
        return_type = self.build_type(func_spec.get("return_type", {"name": "void"}))
        
        # Build body (simplified for now)
        body = []
        for stmt_spec in func_spec.get("body", []):
            stmt = self.build_statement(stmt_spec)
            body.append(stmt)
        
        return Function(
            name=func_spec["name"],
            parameters=parameters,
            return_type=return_type,
            body=body,
            location=location,
            is_inline=func_spec.get("is_inline", False),
            is_static=func_spec.get("is_static", False),
        )

    def build_type(self, type_spec: Dict) -> Type:
        """Build type from specification."""
        if isinstance(type_spec, str):
            # Simple type name
            return Type(name=type_spec, is_primitive=True)
        
        type_obj = Type(
            name=type_spec.get("name", "unknown"),
            is_primitive=type_spec.get("is_primitive", False),
            is_pointer=type_spec.get("is_pointer", False),
            is_array=type_spec.get("is_array", False),
            array_size=type_spec.get("array_size"),
        )
        
        # Build element type for arrays/pointers
        if "element_type" in type_spec:
            type_obj.element_type = self.build_type(type_spec["element_type"])
        
        # Build struct fields
        if "fields" in type_spec:
            for field_name, field_type_spec in type_spec["fields"].items():
                type_obj.fields[field_name] = self.build_type(field_type_spec)
        
        return type_obj

    def build_typedef(self, type_spec: Dict) -> TypeDef:
        """Build type definition from specification."""
        location = self._make_location(type_spec.get("location", {}))
        type_obj = self.build_type(type_spec.get("type", {}))
        
        return TypeDef(
            name=type_spec["name"],
            type=type_obj,
            location=location,
        )

    def build_constant(self, const_spec: Dict) -> Constant:
        """Build constant from specification."""
        location = self._make_location(const_spec.get("location", {}))
        type_obj = self.build_type(const_spec.get("type", {}))
        
        return Constant(
            name=const_spec["name"],
            type=type_obj,
            value=const_spec.get("value"),
            location=location,
        )

    def build_import(self, import_spec: Dict) -> Import:
        """Build import from specification."""
        location = self._make_location(import_spec.get("location", {}))
        
        return Import(
            module=import_spec["module"],
            items=import_spec.get("items", []),
            location=location,
        )

    def build_statement(self, stmt_spec: Dict) -> Statement:
        """Build statement from specification."""
        location = self._make_location(stmt_spec.get("location", {}))
        
        stmt = Statement(
            kind=stmt_spec.get("kind", "unknown"),
            location=location,
        )
        
        # Build expressions in statement
        for expr_spec in stmt_spec.get("expressions", []):
            expr = self.build_expression(expr_spec)
            stmt.expressions.append(expr)
        
        # Build nested statements
        for nested_spec in stmt_spec.get("statements", []):
            nested_stmt = self.build_statement(nested_spec)
            stmt.statements.append(nested_stmt)
        
        return stmt

    def build_expression(self, expr_spec: Dict) -> Expression:
        """Build expression from specification."""
        location = self._make_location(expr_spec.get("location", {}))
        
        expr = Expression(
            kind=expr_spec.get("kind", "unknown"),
            value=expr_spec.get("value"),
            location=location,
        )
        
        # Build type if provided
        if "type" in expr_spec:
            expr.type = self.build_type(expr_spec["type"])
        
        return expr

    def resolve_reference(self, ref: str) -> Optional[ASTNode]:
        """Resolve reference to symbol."""
        # This would be implemented to look up symbols in symbol table
        return None

    def _make_location(self, loc_spec: Dict) -> SourceLocation:
        """Create source location from specification."""
        return SourceLocation(
            file=loc_spec.get("file", self.filename),
            line=loc_spec.get("line", self.current_line),
            column=loc_spec.get("column", self.current_column),
            length=loc_spec.get("length", 1),
        )

    def _add_error(self, error_type: ErrorType, message: str, location: SourceLocation):
        """Add parsing error."""
        error = ParseError(
            location=location,
            error_type=error_type,
            message=message,
        )
        self.errors.append(error)
