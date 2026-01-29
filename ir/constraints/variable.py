"""Decision variable definitions for Constraint Programming.

This module defines decision variables and array variables
for constraint satisfaction problems.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, TYPE_CHECKING

from .constraint_ir import VariableType

if TYPE_CHECKING:
    from .domain import Domain


@dataclass
class IndexSet:
    """Index set for arrays.
    
    Attributes:
        ranges: List of (lower, upper) bounds for each dimension
    """
    ranges: List[Tuple[int, int]]  # [(lb1, ub1), (lb2, ub2), ...]
    
    @property
    def dimensions(self) -> int:
        """Get number of dimensions."""
        return len(self.ranges)
    
    @property
    def size(self) -> int:
        """Get total number of elements."""
        result = 1
        for lb, ub in self.ranges:
            result *= (ub - lb + 1)
        return result
    
    def to_minizinc(self) -> str:
        """Convert to MiniZinc index set syntax."""
        parts = [f"{lb}..{ub}" for lb, ub in self.ranges]
        return ", ".join(parts)
    
    def __str__(self) -> str:
        return self.to_minizinc()


@dataclass
class Variable:
    """Decision variable in a constraint model.
    
    Attributes:
        name: Variable identifier
        var_type: Type of variable (INT, FLOAT, BOOL, SET)
        domain: Domain of the variable
        annotations: Optional annotations (e.g., search hints)
    """
    name: str
    var_type: VariableType
    domain: 'Domain'
    annotations: List[str] = field(default_factory=list)
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        if isinstance(other, Variable):
            return self.name == other.name
        return False
    
    def __str__(self) -> str:
        return f"{self.name}: {self.var_type.name} in {self.domain}"


@dataclass
class ArrayVariable:
    """Array of decision variables.
    
    Attributes:
        name: Array identifier
        element_type: Type of elements
        index_set: Index set definition
        element_domain: Domain for each element
        annotations: Optional annotations
    """
    name: str
    element_type: VariableType
    index_set: IndexSet
    element_domain: 'Domain'
    annotations: List[str] = field(default_factory=list)
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        if isinstance(other, ArrayVariable):
            return self.name == other.name
        return False
    
    @property
    def size(self) -> int:
        """Get total number of elements."""
        return self.index_set.size
    
    def __str__(self) -> str:
        return f"{self.name}[{self.index_set}]: {self.element_type.name} in {self.element_domain}"


@dataclass
class Parameter:
    """Parameter (constant) in a constraint model.
    
    Unlike variables, parameters have fixed values.
    
    Attributes:
        name: Parameter identifier
        value: The parameter value
        param_type: Type of parameter
    """
    name: str
    value: any
    param_type: Optional[VariableType] = None
    
    def __str__(self) -> str:
        return f"{self.name} = {self.value}"
