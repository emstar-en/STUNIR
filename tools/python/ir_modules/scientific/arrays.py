#!/usr/bin/env python3
"""STUNIR Scientific IR - Array operations and slicing.

This module defines IR nodes for array operations common in scientific
computing, including multi-dimensional arrays, slicing, whole-array
operations, and array intrinsic functions.

Usage:
    from ir.scientific.arrays import ArrayType, ArraySlice, ArrayIntrinsic
    
    # Define a 2D array
    arr = ArrayType(
        element_type=TypeRef(name='f64'),
        dimensions=[
            ArrayDimension(lower=Literal(value=1), upper=Literal(value=10)),
            ArrayDimension(lower=Literal(value=1), upper=Literal(value=10))
        ]
    )
    
    # Create an array slice
    slice_op = ArraySlice(
        array=VarRef(name='matrix'),
        slices=[
            SliceSpec(start=Literal(value=1), stop=Literal(value=5)),
            SliceSpec()  # All elements in dimension
        ]
    )
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any
from enum import Enum

from ir.scientific.scientific_ir import ScientificNode, TypeRef, Literal, ArrayOrder


# =============================================================================
# Array Types
# =============================================================================

@dataclass
class ArrayDimension(ScientificNode):
    """Single array dimension with bounds.
    
    Attributes:
        lower: Lower bound (default 1 for Fortran, 0 for Pascal)
        upper: Upper bound expression
        is_assumed: Assumed size (Fortran '*')
        is_deferred: Deferred shape (allocatable/pointer)
        is_assumed_shape: Assumed shape (:)
    """
    lower: Optional['Expr'] = None
    upper: Optional['Expr'] = None
    is_assumed: bool = False       # Fortran assumed size (*)
    is_deferred: bool = False      # Deferred shape (allocatable/pointer)
    is_assumed_shape: bool = False # Assumed shape (:)
    kind: str = 'array_dimension'


@dataclass
class ArrayType(ScientificNode):
    """Array type with dimensions and bounds.
    
    Attributes:
        element_type: Type of array elements
        dimensions: List of dimension specifications
        order: Column-major (Fortran) or row-major (Pascal)
        allocatable: Fortran ALLOCATABLE attribute
        pointer: Array is a pointer
        contiguous: Fortran CONTIGUOUS attribute
    """
    element_type: TypeRef = None
    dimensions: List[ArrayDimension] = field(default_factory=list)
    order: ArrayOrder = ArrayOrder.COLUMN_MAJOR
    allocatable: bool = False
    pointer: bool = False
    contiguous: bool = False
    kind: str = 'array_type'


@dataclass  
class CoarrayType(ScientificNode):
    """Fortran coarray type with codimensions.
    
    Attributes:
        element_type: Type of coarray elements
        dimensions: Regular dimensions
        codimensions: Codimensions for parallel access
    """
    element_type: TypeRef = None
    dimensions: List[ArrayDimension] = field(default_factory=list)
    codimensions: List[ArrayDimension] = field(default_factory=list)
    kind: str = 'coarray_type'


# =============================================================================
# Array Slicing and Sectioning
# =============================================================================

@dataclass
class SliceSpec(ScientificNode):
    """Slice specification for one dimension.
    
    Represents [start:stop:stride] or just : for all elements.
    If all are None, represents the entire dimension.
    
    Attributes:
        start: Starting index (None = lower bound)
        stop: Ending index (None = upper bound)
        stride: Step/stride (None = 1)
    """
    start: Optional['Expr'] = None
    stop: Optional['Expr'] = None
    stride: Optional['Expr'] = None
    kind: str = 'slice_spec'
    
    def is_full_slice(self) -> bool:
        """Check if this is a full dimension slice (:)."""
        return self.start is None and self.stop is None and self.stride is None


@dataclass
class ArraySlice(ScientificNode):
    """Array slice/section operation.
    
    Represents array section like A(1:5, :) or A(1:10:2).
    
    Attributes:
        array: The array expression being sliced
        slices: List of slice specs for each dimension
    """
    array: 'Expr' = None
    slices: List[SliceSpec] = field(default_factory=list)
    kind: str = 'array_slice'


@dataclass
class ArrayReshape(ScientificNode):
    """Reshape array to new dimensions.
    
    Attributes:
        array: Source array
        shape: New shape (list of dimension sizes)
        order: Optional reorder specification
    """
    array: 'Expr' = None
    shape: List['Expr'] = field(default_factory=list)
    order: Optional[List['Expr']] = None
    kind: str = 'array_reshape'


# =============================================================================
# Whole-Array Operations
# =============================================================================

class ArrayOpKind(Enum):
    """Kinds of whole-array operations."""
    ADD = 'add'
    SUBTRACT = 'subtract'
    MULTIPLY = 'multiply'
    DIVIDE = 'divide'
    POWER = 'power'
    MATMUL = 'matmul'
    TRANSPOSE = 'transpose'
    CONCAT = 'concat'


@dataclass
class ArrayOperation(ScientificNode):
    """Whole-array operation.
    
    Represents element-wise or matrix operations on entire arrays.
    
    Attributes:
        op: Operation kind (add, mul, matmul, etc.)
        operands: List of array operands
    """
    op: str = ''
    operands: List['Expr'] = field(default_factory=list)
    kind: str = 'array_operation'


@dataclass
class ArrayConstructor(ScientificNode):
    """Array constructor (Fortran [/.../, ...]).
    
    Attributes:
        elements: List of elements or implied-do loops
        element_type: Optional explicit element type
    """
    elements: List['Expr'] = field(default_factory=list)
    element_type: Optional[TypeRef] = None
    kind: str = 'array_constructor'


@dataclass
class ImpliedDo(ScientificNode):
    """Implied DO loop in array constructor.
    
    Fortran: (expr, i = start, end [, step])
    
    Attributes:
        expr: Expression to evaluate
        variable: Loop variable
        start: Start value
        end: End value
        step: Optional step value
    """
    expr: 'Expr' = None
    variable: str = ''
    start: 'Expr' = None
    end: 'Expr' = None
    step: Optional['Expr'] = None
    kind: str = 'implied_do'


# =============================================================================
# Array Intrinsic Functions
# =============================================================================

class ArrayIntrinsicKind(Enum):
    """Kinds of array intrinsic functions."""
    # Reduction operations
    SUM = 'sum'
    PRODUCT = 'product'
    MAXVAL = 'maxval'
    MINVAL = 'minval'
    ANY = 'any'
    ALL = 'all'
    COUNT = 'count'
    # Query operations
    SIZE = 'size'
    SHAPE = 'shape'
    LBOUND = 'lbound'
    UBOUND = 'ubound'
    ALLOCATED = 'allocated'
    # Transformation operations
    RESHAPE = 'reshape'
    TRANSPOSE = 'transpose'
    SPREAD = 'spread'
    PACK = 'pack'
    UNPACK = 'unpack'
    MERGE = 'merge'
    CSHIFT = 'cshift'
    EOSHIFT = 'eoshift'
    # Location operations
    MAXLOC = 'maxloc'
    MINLOC = 'minloc'
    FINDLOC = 'findloc'
    # Matrix operations
    MATMUL = 'matmul'
    DOT_PRODUCT = 'dot_product'


@dataclass
class ArrayIntrinsic(ScientificNode):
    """Array intrinsic function call.
    
    Attributes:
        name: Intrinsic function name (SUM, MAXVAL, RESHAPE, etc.)
        array: Primary array argument
        dim: Optional dimension argument
        mask: Optional mask argument
        back: Optional back argument (for MAXLOC, etc.)
        extra_args: Additional arguments specific to function
    """
    name: str = ''
    array: 'Expr' = None
    dim: Optional['Expr'] = None
    mask: Optional['Expr'] = None
    back: Optional['Expr'] = None
    extra_args: List['Expr'] = field(default_factory=list)
    kind: str = 'array_intrinsic'


@dataclass
class MatrixOperation(ScientificNode):
    """Matrix-specific operation.
    
    Attributes:
        op: Operation name (matmul, transpose, inverse, etc.)
        operands: Matrix operands
    """
    op: str = ''
    operands: List['Expr'] = field(default_factory=list)
    kind: str = 'matrix_operation'


# =============================================================================
# Array Assignment Statements
# =============================================================================

@dataclass
class ArrayAssignment(ScientificNode):
    """Array assignment (whole array or section).
    
    Attributes:
        target: Target array or section
        value: Source value (scalar broadcasts to all elements)
    """
    target: 'Expr' = None
    value: 'Expr' = None
    kind: str = 'array_assignment'


@dataclass
class WhereStatement(ScientificNode):
    """Fortran WHERE statement for masked array assignment.
    
    Attributes:
        mask: Logical mask array
        body: Assignments to perform where mask is true
        elsewhere_body: Assignments where mask is false
    """
    mask: 'Expr' = None
    body: List['Statement'] = field(default_factory=list)
    elsewhere_body: List['Statement'] = field(default_factory=list)
    kind: str = 'where_statement'


@dataclass
class ForallStatement(ScientificNode):
    """Fortran FORALL statement.
    
    Attributes:
        indices: List of index specifications
        mask: Optional mask expression
        body: Assignments to perform
    """
    indices: List['ForallIndex'] = field(default_factory=list)
    mask: Optional['Expr'] = None
    body: List['Statement'] = field(default_factory=list)
    kind: str = 'forall_statement'


@dataclass
class ForallIndex(ScientificNode):
    """FORALL index specification.
    
    Attributes:
        variable: Index variable name
        lower: Lower bound
        upper: Upper bound
        stride: Optional stride
    """
    variable: str = ''
    lower: 'Expr' = None
    upper: 'Expr' = None
    stride: Optional['Expr'] = None
    kind: str = 'forall_index'


# =============================================================================
# Pascal-Specific Array Operations
# =============================================================================

@dataclass
class PascalArrayType(ScientificNode):
    """Pascal array type with explicit index types.
    
    Pascal arrays use index type ranges: array[1..10] of Integer
    
    Attributes:
        element_type: Type of array elements
        index_ranges: List of index range specifications
        is_packed: PACKED array
    """
    element_type: TypeRef = None
    index_ranges: List['IndexRange'] = field(default_factory=list)
    is_packed: bool = False
    kind: str = 'pascal_array_type'


@dataclass
class IndexRange(ScientificNode):
    """Pascal array index range.
    
    Can be a subrange (1..10), enumeration, or char range ('a'..'z').
    
    Attributes:
        lower: Lower bound
        upper: Upper bound
        base_type: Base ordinal type (optional)
    """
    lower: 'Expr' = None
    upper: 'Expr' = None
    base_type: Optional[TypeRef] = None
    kind: str = 'index_range'


@dataclass
class DynamicArray(ScientificNode):
    """Pascal/Delphi dynamic array.
    
    Dynamic arrays are allocated at runtime with SetLength.
    
    Attributes:
        element_type: Type of array elements
        dimensions: Number of dimensions (nested dynamic arrays)
    """
    element_type: TypeRef = None
    dimensions: int = 1
    kind: str = 'dynamic_array'


@dataclass
class SetLength(ScientificNode):
    """Pascal SetLength call for dynamic arrays.
    
    Attributes:
        array: Array to resize
        sizes: New size for each dimension
    """
    array: 'Expr' = None
    sizes: List['Expr'] = field(default_factory=list)
    kind: str = 'set_length'


@dataclass
class Length(ScientificNode):
    """Pascal Length function for arrays/strings.
    
    Attributes:
        target: Array or string to get length of
        dimension: Optional dimension (for multi-dim arrays)
    """
    target: 'Expr' = None
    dimension: Optional[int] = None
    kind: str = 'length'


@dataclass
class High(ScientificNode):
    """Pascal High function - upper bound of array/ordinal.
    
    Attributes:
        target: Array or ordinal type
    """
    target: 'Expr' = None
    kind: str = 'high'


@dataclass
class Low(ScientificNode):
    """Pascal Low function - lower bound of array/ordinal.
    
    Attributes:
        target: Array or ordinal type
    """
    target: 'Expr' = None
    kind: str = 'low'


# =============================================================================
# String Operations (often related to arrays)
# =============================================================================

@dataclass
class StringConcat(ScientificNode):
    """String concatenation.
    
    Fortran: // operator
    Pascal: + operator
    
    Attributes:
        parts: Strings to concatenate
    """
    parts: List['Expr'] = field(default_factory=list)
    kind: str = 'string_concat'


@dataclass
class Substring(ScientificNode):
    """Substring extraction.
    
    Fortran: string(start:end)
    Pascal: Copy(string, start, count)
    
    Attributes:
        string: Source string
        start: Starting position
        end: Ending position (or length for Pascal)
    """
    string: 'Expr' = None
    start: 'Expr' = None
    end: 'Expr' = None
    kind: str = 'substring'


# Expression type alias
Expr = Any  # Forward reference to expression types
