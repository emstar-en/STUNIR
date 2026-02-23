"""STUNIR IR Visitor - Python Reference Implementation

Visitor pattern for traversing and processing IR structures.
Based on Ada SPARK visitor patterns.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, List

from .types import (
    IRModule,
    IRFunction,
    IRType,
    IRStatement,
    IRParameter,
    IRTypeField,
)


class IRVisitor(ABC):
    """Base visitor for IR traversal.
    
    Implements the Visitor pattern for walking the IR tree.
    Subclasses can override specific visit methods to customize behavior.
    """

    def visit_module(self, module: IRModule) -> Any:
        """Visit an IR module (entry point).
        
        Args:
            module: IR module to visit
            
        Returns:
            Result of visiting the module
        """
        result = self.enter_module(module)
        
        # Visit all types
        for ir_type in module.types:
            self.visit_type(ir_type)
        
        # Visit all functions
        for function in module.functions:
            self.visit_function(function)
        
        return self.exit_module(module, result)

    def visit_type(self, ir_type: IRType) -> Any:
        """Visit a type definition.
        
        Args:
            ir_type: Type to visit
            
        Returns:
            Result of visiting the type
        """
        result = self.enter_type(ir_type)
        
        # Visit all fields
        for field in ir_type.fields:
            self.visit_field(field)
        
        return self.exit_type(ir_type, result)

    def visit_field(self, field: IRTypeField) -> Any:
        """Visit a type field.
        
        Args:
            field: Field to visit
            
        Returns:
            Result of visiting the field
        """
        return self.process_field(field)

    def visit_function(self, function: IRFunction) -> Any:
        """Visit a function definition.
        
        Args:
            function: Function to visit
            
        Returns:
            Result of visiting the function
        """
        result = self.enter_function(function)
        
        # Visit all parameters
        for param in function.parameters:
            self.visit_parameter(param)
        
        # Visit all statements
        for stmt in function.statements:
            self.visit_statement(stmt)
        
        return self.exit_function(function, result)

    def visit_parameter(self, param: IRParameter) -> Any:
        """Visit a function parameter.
        
        Args:
            param: Parameter to visit
            
        Returns:
            Result of visiting the parameter
        """
        return self.process_parameter(param)

    def visit_statement(self, stmt: IRStatement) -> Any:
        """Visit a statement.
        
        Args:
            stmt: Statement to visit
            
        Returns:
            Result of visiting the statement
        """
        return self.process_statement(stmt)

    # Abstract methods to be implemented by subclasses

    @abstractmethod
    def enter_module(self, module: IRModule) -> Any:
        """Called when entering a module."""
        pass

    @abstractmethod
    def exit_module(self, module: IRModule, context: Any) -> Any:
        """Called when exiting a module."""
        pass

    @abstractmethod
    def enter_type(self, ir_type: IRType) -> Any:
        """Called when entering a type definition."""
        pass

    @abstractmethod
    def exit_type(self, ir_type: IRType, context: Any) -> Any:
        """Called when exiting a type definition."""
        pass

    @abstractmethod
    def process_field(self, field: IRTypeField) -> Any:
        """Process a type field."""
        pass

    @abstractmethod
    def enter_function(self, function: IRFunction) -> Any:
        """Called when entering a function."""
        pass

    @abstractmethod
    def exit_function(self, function: IRFunction, context: Any) -> Any:
        """Called when exiting a function."""
        pass

    @abstractmethod
    def process_parameter(self, param: IRParameter) -> Any:
        """Process a function parameter."""
        pass

    @abstractmethod
    def process_statement(self, stmt: IRStatement) -> Any:
        """Process a statement."""
        pass


class CodeGenVisitor(IRVisitor):
    """Code generation visitor base class.
    
    Extends IRVisitor to accumulate generated code as a string.
    """

    def __init__(self):
        """Initialize visitor with empty code buffer."""
        self.code: List[str] = []
        self.indent_level: int = 0

    def emit(self, line: str = ""):
        """Emit a line of code with current indentation.
        
        Args:
            line: Line of code to emit (without indentation)
        """
        if line:
            self.code.append("    " * self.indent_level + line)
        else:
            self.code.append("")

    def get_code(self) -> str:
        """Get accumulated code as string.
        
        Returns:
            Generated code
        """
        return "\n".join(self.code)

    def increase_indent(self):
        """Increase indentation level."""
        self.indent_level += 1

    def decrease_indent(self):
        """Decrease indentation level."""
        self.indent_level = max(0, self.indent_level - 1)

    # Default implementations (can be overridden)

    def enter_module(self, module: IRModule) -> Any:
        """Default module entry."""
        return None

    def exit_module(self, module: IRModule, context: Any) -> Any:
        """Default module exit."""
        return self.get_code()

    def enter_type(self, ir_type: IRType) -> Any:
        """Default type entry."""
        return None

    def exit_type(self, ir_type: IRType, context: Any) -> Any:
        """Default type exit."""
        return None

    def process_field(self, field: IRTypeField) -> Any:
        """Default field processing."""
        return None

    def enter_function(self, function: IRFunction) -> Any:
        """Default function entry."""
        return None

    def exit_function(self, function: IRFunction, context: Any) -> Any:
        """Default function exit."""
        return None

    def process_parameter(self, param: IRParameter) -> Any:
        """Default parameter processing."""
        return None

    def process_statement(self, stmt: IRStatement) -> Any:
        """Default statement processing."""
        return None
