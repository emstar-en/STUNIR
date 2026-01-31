"""Type definitions for Semantic IR parser."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum


class ErrorType(Enum):
    """Types of parsing errors."""
    LEXICAL = "lexical"
    SYNTAX = "syntax"
    SEMANTIC = "semantic"
    CATEGORY = "category"


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class SourceLocation:
    """Location in source file."""
    file: str
    line: int
    column: int
    length: int = 1

    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.column}"


@dataclass
class ParseError:
    """Parsing error with location and suggestion."""
    location: SourceLocation
    error_type: ErrorType
    message: str
    suggestion: Optional[str] = None
    severity: ErrorSeverity = ErrorSeverity.ERROR

    def __str__(self) -> str:
        result = f"{self.severity.value.upper()}: {self.location}: {self.message}"
        if self.suggestion:
            result += f"\nSuggestion: {self.suggestion}"
        return result


@dataclass
class Type:
    """Type information."""
    name: str
    is_primitive: bool = False
    is_pointer: bool = False
    is_array: bool = False
    array_size: Optional[int] = None
    element_type: Optional['Type'] = None
    fields: Dict[str, 'Type'] = field(default_factory=dict)


@dataclass
class Parameter:
    """Function parameter."""
    name: str
    type: Type
    location: SourceLocation


@dataclass
class Expression:
    """Expression node."""
    kind: str  # literal, variable, binary_op, call, etc.
    value: Any
    type: Optional[Type] = None
    location: Optional[SourceLocation] = None


@dataclass
class Statement:
    """Statement node."""
    kind: str  # assignment, return, if, while, etc.
    expressions: List[Expression] = field(default_factory=list)
    statements: List['Statement'] = field(default_factory=list)
    location: Optional[SourceLocation] = None


@dataclass
class Function:
    """Function definition."""
    name: str
    parameters: List[Parameter]
    return_type: Type
    body: List[Statement]
    location: SourceLocation
    is_inline: bool = False
    is_static: bool = False


@dataclass
class TypeDef:
    """Type definition."""
    name: str
    type: Type
    location: SourceLocation


@dataclass
class Constant:
    """Constant definition."""
    name: str
    type: Type
    value: Any
    location: SourceLocation


@dataclass
class Import:
    """Import/dependency."""
    module: str
    items: List[str]
    location: SourceLocation


@dataclass
class ParserOptions:
    """Options for parser configuration."""
    category: str
    validate_schema: bool = True
    collect_metrics: bool = True
    enable_type_inference: bool = True
    max_errors: int = 100
    incremental: bool = False
