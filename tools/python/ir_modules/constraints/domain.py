"""Domain definitions for Constraint Programming.

This module defines domains for decision variables,
including integer ranges, explicit sets, and boolean domains.
"""

from dataclasses import dataclass, field
from typing import Optional, Set, Any, Union, FrozenSet

from .constraint_ir import DomainType, InvalidDomainError


@dataclass
class Domain:
    """Domain for a decision variable.
    
    Attributes:
        domain_type: Type of domain (RANGE, SET, BOOL, UNBOUNDED)
        lower: Lower bound (for RANGE)
        upper: Upper bound (for RANGE)
        values: Explicit values (for SET)
    """
    domain_type: DomainType
    lower: Optional[Union[int, float]] = None
    upper: Optional[Union[int, float]] = None
    values: Optional[FrozenSet[Any]] = None
    
    def __post_init__(self):
        """Validate domain after initialization."""
        if self.domain_type == DomainType.RANGE:
            if self.lower is None or self.upper is None:
                raise InvalidDomainError("Range domain requires lower and upper bounds")
            if self.lower > self.upper:
                raise InvalidDomainError(f"Invalid range: {self.lower} > {self.upper}")
        elif self.domain_type == DomainType.SET:
            if self.values is None or len(self.values) == 0:
                raise InvalidDomainError("Set domain requires non-empty values")
    
    @classmethod
    def int_range(cls, lower: int, upper: int) -> 'Domain':
        """Create integer range domain.
        
        Args:
            lower: Lower bound (inclusive)
            upper: Upper bound (inclusive)
            
        Returns:
            Domain with RANGE type
        """
        return cls(DomainType.RANGE, lower=int(lower), upper=int(upper))
    
    @classmethod
    def float_range(cls, lower: float, upper: float) -> 'Domain':
        """Create float range domain.
        
        Args:
            lower: Lower bound (inclusive)
            upper: Upper bound (inclusive)
            
        Returns:
            Domain with RANGE type
        """
        return cls(DomainType.RANGE, lower=float(lower), upper=float(upper))
    
    @classmethod
    def bool_domain(cls) -> 'Domain':
        """Create boolean domain.
        
        Returns:
            Domain with BOOL type
        """
        return cls(DomainType.BOOL)
    
    @classmethod
    def set_domain(cls, values: Set[Any]) -> 'Domain':
        """Create explicit set domain.
        
        Args:
            values: Set of allowed values
            
        Returns:
            Domain with SET type
        """
        return cls(DomainType.SET, values=frozenset(values))
    
    @classmethod
    def unbounded(cls) -> 'Domain':
        """Create unbounded domain.
        
        Returns:
            Domain with UNBOUNDED type
        """
        return cls(DomainType.UNBOUNDED)
    
    def size(self) -> Optional[int]:
        """Return domain size if finite.
        
        Returns:
            Domain size or None if infinite/unbounded
        """
        if self.domain_type == DomainType.RANGE:
            if self.lower is not None and self.upper is not None:
                if isinstance(self.lower, int) and isinstance(self.upper, int):
                    return self.upper - self.lower + 1
                return None  # Float ranges are infinite
        elif self.domain_type == DomainType.SET:
            if self.values is not None:
                return len(self.values)
        elif self.domain_type == DomainType.BOOL:
            return 2
        return None
    
    def contains(self, value: Any) -> bool:
        """Check if value is in domain.
        
        Args:
            value: Value to check
            
        Returns:
            True if value is in domain
        """
        if self.domain_type == DomainType.RANGE:
            return self.lower <= value <= self.upper
        elif self.domain_type == DomainType.SET:
            return value in self.values
        elif self.domain_type == DomainType.BOOL:
            return value in (True, False, 0, 1)
        elif self.domain_type == DomainType.UNBOUNDED:
            return True
        return False
    
    def is_integer(self) -> bool:
        """Check if domain is over integers."""
        if self.domain_type == DomainType.RANGE:
            return isinstance(self.lower, int) and isinstance(self.upper, int)
        elif self.domain_type == DomainType.SET:
            return all(isinstance(v, int) for v in self.values)
        elif self.domain_type == DomainType.BOOL:
            return True
        return False
    
    def to_minizinc(self) -> str:
        """Convert to MiniZinc domain syntax.
        
        Returns:
            MiniZinc domain string
        """
        if self.domain_type == DomainType.RANGE:
            return f"{self.lower}..{self.upper}"
        elif self.domain_type == DomainType.SET:
            values_str = ", ".join(str(v) for v in sorted(self.values))
            return "{" + values_str + "}"
        elif self.domain_type == DomainType.BOOL:
            return "bool"
        elif self.domain_type == DomainType.UNBOUNDED:
            return "int"  # Default to int for unbounded
        return "int"
    
    def __str__(self) -> str:
        return self.to_minizinc()
    
    def __hash__(self):
        if self.domain_type == DomainType.RANGE:
            return hash((self.domain_type, self.lower, self.upper))
        elif self.domain_type == DomainType.SET:
            return hash((self.domain_type, self.values))
        return hash(self.domain_type)
