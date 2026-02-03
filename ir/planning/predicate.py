"""Predicate and type definitions for Planning IR.

This module defines types, parameters, predicates, functions,
and atoms used in PDDL planning.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TypeDef:
    """Type definition in PDDL.
    
    Represents a type in the PDDL type hierarchy.
    
    Attributes:
        name: Type name (e.g., "location", "vehicle")
        parent: Parent type name (default: "object")
    
    Example:
        TypeDef("city", "location")  # city is a subtype of location
    """
    name: str
    parent: str = "object"
    
    def __post_init__(self):
        """Validate type definition."""
        if not self.name:
            raise ValueError("Type name cannot be empty")
        if not self.name[0].isalpha() and self.name[0] != '_':
            raise ValueError(f"Invalid type name: {self.name}")


@dataclass
class Parameter:
    """Action or predicate parameter.
    
    Represents a typed parameter in PDDL.
    
    Attributes:
        name: Parameter name (e.g., "?from", "?to")
        param_type: Type of parameter (e.g., "location")
    
    Example:
        Parameter("?loc", "location")
    """
    name: str
    param_type: str = "object"
    
    def __post_init__(self):
        """Validate parameter."""
        if not self.name:
            raise ValueError("Parameter name cannot be empty")
        # PDDL parameters typically start with ?
        if not self.name.startswith('?'):
            self.name = '?' + self.name


@dataclass
class Predicate:
    """Predicate definition.
    
    Represents a predicate schema in PDDL domain.
    
    Attributes:
        name: Predicate name (e.g., "at", "connected")
        parameters: List of typed parameters
    
    Example:
        Predicate("at", [Parameter("?obj", "object"), Parameter("?loc", "location")])
    """
    name: str
    parameters: List[Parameter] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate predicate."""
        if not self.name:
            raise ValueError("Predicate name cannot be empty")
    
    def arity(self) -> int:
        """Return the arity (number of parameters)."""
        return len(self.parameters)
    
    def get_signature(self) -> str:
        """Return a type signature string."""
        if not self.parameters:
            return f"{self.name}/0"
        types = ",".join(p.param_type for p in self.parameters)
        return f"{self.name}({types})"


@dataclass
class Function:
    """Numeric function definition.
    
    Represents a function in PDDL for numeric fluents.
    
    Attributes:
        name: Function name (e.g., "distance", "fuel-level")
        parameters: List of typed parameters
        return_type: Return type (number or object type)
    
    Example:
        Function("distance", [Parameter("?from", "location"), 
                              Parameter("?to", "location")], "number")
    """
    name: str
    parameters: List[Parameter] = field(default_factory=list)
    return_type: str = "number"
    
    def __post_init__(self):
        """Validate function."""
        if not self.name:
            raise ValueError("Function name cannot be empty")


@dataclass
class Atom:
    """Ground or lifted atom (predicate application).
    
    Represents an application of a predicate to arguments.
    
    Attributes:
        predicate: Predicate name
        arguments: List of arguments (variables or constants)
    
    Example:
        Atom("at", ["?robot", "loc1"])  # Lifted atom
        Atom("at", ["robot1", "loc1"])  # Ground atom
    """
    predicate: str
    arguments: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate atom."""
        if not self.predicate:
            raise ValueError("Predicate name cannot be empty")
    
    def is_ground(self) -> bool:
        """Check if atom is ground (no variables)."""
        return not any(arg.startswith('?') for arg in self.arguments)
    
    def arity(self) -> int:
        """Return the arity."""
        return len(self.arguments)


@dataclass
class FunctionApplication:
    """Function application (for numeric expressions).
    
    Attributes:
        function: Function name
        arguments: List of arguments
    
    Example:
        FunctionApplication("distance", ["loc1", "loc2"])
    """
    function: str
    arguments: List[str] = field(default_factory=list)
