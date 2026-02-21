#!/usr/bin/env python3
"""STUNIR Scientific IR - Numerical computing primitives.

This module defines IR nodes for numerical computing, including
numeric types, mathematical intrinsic functions, complex numbers,
and Fortran parallel constructs.

Usage:
    from ir.scientific.numerical import NumericType, MathIntrinsic, DoConcurrent
    
    # Create a complex type
    cplx = ComplexType(
        component_type=NumericType(base='real', kind_param=8)
    )
    
    # Create a math intrinsic call
    sin_call = MathIntrinsic(name='sin', arguments=[VarRef(name='x')])
    
    # Create a DO CONCURRENT loop
    do_conc = DoConcurrent(
        indices=[LoopIndex(variable='i', start=1, end='n')],
        body=[...]
    )
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from ir.scientific.scientific_ir import ScientificNode, TypeRef


# =============================================================================
# Numeric Types
# =============================================================================

@dataclass
class NumericType(ScientificNode):
    """Numeric type with precision specification.
    
    Attributes:
        base: Base type ('integer', 'real', 'complex')
        kind_param: Fortran KIND parameter (1, 2, 4, 8, 16)
        precision: Decimal precision (for REAL with SELECTED_REAL_KIND)
        range: Exponent range (for SELECTED_REAL_KIND)
    """
    base: str = 'real'
    kind_param: Optional[int] = None
    precision: Optional[int] = None
    range: Optional[int] = None
    kind: str = 'numeric_type'


@dataclass
class ComplexType(ScientificNode):
    """Complex number type.
    
    Attributes:
        component_type: Type of real and imaginary parts
    """
    component_type: NumericType = None
    kind: str = 'complex_type'


@dataclass
class ComplexLiteral(ScientificNode):
    """Complex number literal.
    
    Attributes:
        real_part: Real component
        imag_part: Imaginary component
    """
    real_part: 'Expr' = None
    imag_part: 'Expr' = None
    kind: str = 'complex_literal'


@dataclass
class ComplexOp(ScientificNode):
    """Complex number operation.
    
    Attributes:
        op: Operation ('real', 'imag', 'conjg', 'abs')
        operand: Complex number expression
    """
    op: str = ''
    operand: 'Expr' = None
    kind: str = 'complex_op'


# =============================================================================
# Mathematical Intrinsic Functions
# =============================================================================

# Comprehensive intrinsic mapping for Fortran and Pascal
INTRINSIC_MAP: Dict[str, Dict[str, str]] = {
    # Trigonometric functions
    'sin': {'fortran': 'SIN', 'pascal': 'Sin'},
    'cos': {'fortran': 'COS', 'pascal': 'Cos'},
    'tan': {'fortran': 'TAN', 'pascal': 'Tan'},
    'asin': {'fortran': 'ASIN', 'pascal': 'ArcSin'},
    'acos': {'fortran': 'ACOS', 'pascal': 'ArcCos'},
    'atan': {'fortran': 'ATAN', 'pascal': 'ArcTan'},
    'atan2': {'fortran': 'ATAN2', 'pascal': 'ArcTan2'},
    'sinh': {'fortran': 'SINH', 'pascal': 'Sinh'},
    'cosh': {'fortran': 'COSH', 'pascal': 'Cosh'},
    'tanh': {'fortran': 'TANH', 'pascal': 'Tanh'},
    'asinh': {'fortran': 'ASINH', 'pascal': 'ArcSinh'},
    'acosh': {'fortran': 'ACOSH', 'pascal': 'ArcCosh'},
    'atanh': {'fortran': 'ATANH', 'pascal': 'ArcTanh'},
    
    # Exponential and logarithmic
    'exp': {'fortran': 'EXP', 'pascal': 'Exp'},
    'log': {'fortran': 'LOG', 'pascal': 'Ln'},
    'log10': {'fortran': 'LOG10', 'pascal': 'Log10'},
    'log2': {'fortran': 'LOG2', 'pascal': 'Log2'},
    
    # Power and roots
    'sqrt': {'fortran': 'SQRT', 'pascal': 'Sqrt'},
    'pow': {'fortran': '**', 'pascal': 'Power'},
    'cbrt': {'fortran': 'CBRT', 'pascal': 'Cbrt'},
    
    # Absolute value and sign
    'abs': {'fortran': 'ABS', 'pascal': 'Abs'},
    'sign': {'fortran': 'SIGN', 'pascal': 'Sign'},
    
    # Rounding
    'floor': {'fortran': 'FLOOR', 'pascal': 'Floor'},
    'ceil': {'fortran': 'CEILING', 'pascal': 'Ceil'},
    'round': {'fortran': 'NINT', 'pascal': 'Round'},
    'trunc': {'fortran': 'INT', 'pascal': 'Trunc'},
    
    # Modulo and remainder
    'mod': {'fortran': 'MOD', 'pascal': 'Mod'},
    'modulo': {'fortran': 'MODULO', 'pascal': 'Mod'},
    
    # Min/Max
    'min': {'fortran': 'MIN', 'pascal': 'Min'},
    'max': {'fortran': 'MAX', 'pascal': 'Max'},
    
    # Type conversion
    'int': {'fortran': 'INT', 'pascal': 'Trunc'},
    'real': {'fortran': 'REAL', 'pascal': 'Extended'},
    'dble': {'fortran': 'DBLE', 'pascal': 'Double'},
    'cmplx': {'fortran': 'CMPLX', 'pascal': 'Complex'},
    
    # Complex number operations
    'conjg': {'fortran': 'CONJG', 'pascal': 'Conjugate'},
    'aimag': {'fortran': 'AIMAG', 'pascal': 'GetImag'},
    
    # Bit operations (integers)
    'iand': {'fortran': 'IAND', 'pascal': 'And'},
    'ior': {'fortran': 'IOR', 'pascal': 'Or'},
    'ieor': {'fortran': 'IEOR', 'pascal': 'Xor'},
    'not': {'fortran': 'NOT', 'pascal': 'Not'},
    'ishft': {'fortran': 'ISHFT', 'pascal': 'Shl'},
    'ishftc': {'fortran': 'ISHFTC', 'pascal': 'RolX'},
    'btest': {'fortran': 'BTEST', 'pascal': 'TestBit'},
    'ibset': {'fortran': 'IBSET', 'pascal': 'SetBit'},
    'ibclr': {'fortran': 'IBCLR', 'pascal': 'ClearBit'},
    
    # Random numbers
    'random': {'fortran': 'RANDOM_NUMBER', 'pascal': 'Random'},
}

# Fortran-specific array intrinsics
FORTRAN_ARRAY_INTRINSICS = [
    'SUM', 'PRODUCT', 'MAXVAL', 'MINVAL', 'ANY', 'ALL', 'COUNT',
    'MAXLOC', 'MINLOC', 'FINDLOC',
    'SIZE', 'SHAPE', 'LBOUND', 'UBOUND',
    'RESHAPE', 'TRANSPOSE', 'SPREAD', 'PACK', 'UNPACK', 'MERGE',
    'CSHIFT', 'EOSHIFT',
    'MATMUL', 'DOT_PRODUCT',
    'ALLOCATED', 'ASSOCIATED',
]


@dataclass
class MathIntrinsic(ScientificNode):
    """Mathematical intrinsic function call.
    
    Attributes:
        name: Intrinsic function name (IR standard name)
        arguments: Function arguments
    """
    name: str = ''
    arguments: List['Expr'] = field(default_factory=list)
    kind: str = 'math_intrinsic'
    
    def fortran_name(self) -> str:
        """Get Fortran intrinsic name."""
        return INTRINSIC_MAP.get(self.name, {}).get('fortran', self.name.upper())
    
    def pascal_name(self) -> str:
        """Get Pascal intrinsic name."""
        return INTRINSIC_MAP.get(self.name, {}).get('pascal', self.name.capitalize())


@dataclass
class TypeIntrinsic(ScientificNode):
    """Type inquiry intrinsic function.
    
    Fortran: KIND, SELECTED_INT_KIND, SELECTED_REAL_KIND, etc.
    
    Attributes:
        name: Intrinsic name
        arguments: Arguments
    """
    name: str = ''
    arguments: List['Expr'] = field(default_factory=list)
    kind: str = 'type_intrinsic'


@dataclass
class CharIntrinsic(ScientificNode):
    """Character intrinsic function.
    
    Attributes:
        name: Function name (CHAR, ICHAR, LEN, TRIM, etc.)
        arguments: Arguments
    """
    name: str = ''
    arguments: List['Expr'] = field(default_factory=list)
    kind: str = 'char_intrinsic'


# =============================================================================
# Fortran Parallel Constructs
# =============================================================================

@dataclass
class LoopIndex(ScientificNode):
    """Loop index specification.
    
    Attributes:
        variable: Index variable name
        start: Start value
        end: End value
        stride: Optional stride (default 1)
    """
    variable: str = ''
    start: 'Expr' = None
    end: 'Expr' = None
    stride: Optional['Expr'] = None
    kind: str = 'loop_index'


@dataclass
class ReduceSpec(ScientificNode):
    """Reduction specification for parallel loops.
    
    Attributes:
        op: Reduction operation ('+', '*', 'max', 'min', etc.)
        variable: Variable being reduced
    """
    op: str = ''
    variable: str = ''
    kind: str = 'reduce_spec'


@dataclass
class LocalitySpec(ScientificNode):
    """DO CONCURRENT locality specification.
    
    Attributes:
        local_vars: LOCAL variables (private to each iteration)
        local_init: LOCAL_INIT variables (initialized from outer scope)
        shared: SHARED variables (shared across iterations)
        reduce_ops: REDUCE specifications
        default_none: DEFAULT(NONE) specified
    """
    local_vars: List[str] = field(default_factory=list)
    local_init: List[str] = field(default_factory=list)
    shared: List[str] = field(default_factory=list)
    reduce_ops: List[ReduceSpec] = field(default_factory=list)
    default_none: bool = False
    kind: str = 'locality_spec'


@dataclass
class DoConcurrent(ScientificNode):
    """Fortran DO CONCURRENT loop.
    
    Modern Fortran parallel loop construct.
    
    Attributes:
        indices: List of index specifications
        mask: Optional mask expression
        locality: Locality specification
        body: Loop body statements
    """
    indices: List[LoopIndex] = field(default_factory=list)
    mask: Optional['Expr'] = None
    locality: Optional[LocalitySpec] = None
    body: List['Statement'] = field(default_factory=list)
    kind: str = 'do_concurrent'


# =============================================================================
# Coarray Constructs (Fortran 2008+)
# =============================================================================

@dataclass
class Coarray(ScientificNode):
    """Fortran coarray declaration.
    
    Coarrays are arrays that exist on multiple images (processes).
    
    Attributes:
        name: Coarray name
        element_type: Type of elements
        dimensions: Array dimensions
        codimensions: Codimensions (image extents)
    """
    name: str = ''
    element_type: TypeRef = None
    dimensions: List['ArrayDimension'] = field(default_factory=list)
    codimensions: List['ArrayDimension'] = field(default_factory=list)
    kind: str = 'coarray'


@dataclass
class CoarrayAccess(ScientificNode):
    """Coarray element access with image index.
    
    Fortran: array(i,j)[image_num]
    
    Attributes:
        coarray: Coarray reference
        indices: Array indices
        coindices: Coarray indices (image selection)
    """
    coarray: 'Expr' = None
    indices: List['Expr'] = field(default_factory=list)
    coindices: List['Expr'] = field(default_factory=list)
    kind: str = 'coarray_access'


@dataclass
class SyncAll(ScientificNode):
    """Fortran SYNC ALL statement.
    
    Synchronize all images.
    
    Attributes:
        stat_var: Optional STAT variable
        errmsg_var: Optional ERRMSG variable
    """
    stat_var: Optional[str] = None
    errmsg_var: Optional[str] = None
    kind: str = 'sync_all'


@dataclass
class SyncImages(ScientificNode):
    """Fortran SYNC IMAGES statement.
    
    Synchronize specific images.
    
    Attributes:
        images: Image list or '*' for all
        stat_var: Optional STAT variable
        errmsg_var: Optional ERRMSG variable
    """
    images: List['Expr'] = field(default_factory=list)
    stat_var: Optional[str] = None
    errmsg_var: Optional[str] = None
    kind: str = 'sync_images'


@dataclass
class CriticalBlock(ScientificNode):
    """Fortran CRITICAL block.
    
    Mutual exclusion block for coarray programs.
    
    Attributes:
        name: Optional block name
        body: Statements in critical section
    """
    name: Optional[str] = None
    body: List['Statement'] = field(default_factory=list)
    kind: str = 'critical_block'


@dataclass
class ImageIntrinsic(ScientificNode):
    """Coarray image intrinsic function.
    
    Functions: THIS_IMAGE(), NUM_IMAGES(), IMAGE_INDEX(), etc.
    
    Attributes:
        name: Intrinsic name
        arguments: Arguments
    """
    name: str = ''
    arguments: List['Expr'] = field(default_factory=list)
    kind: str = 'image_intrinsic'


# =============================================================================
# IEEE Floating Point Operations
# =============================================================================

@dataclass
class IEEEIntrinsic(ScientificNode):
    """IEEE floating-point intrinsic.
    
    Fortran IEEE_ARITHMETIC module functions.
    
    Attributes:
        name: Function name (IEEE_IS_NAN, IEEE_IS_FINITE, etc.)
        arguments: Arguments
    """
    name: str = ''
    arguments: List['Expr'] = field(default_factory=list)
    kind: str = 'ieee_intrinsic'


@dataclass
class IEEEFlag(ScientificNode):
    """IEEE exception flag.
    
    Attributes:
        flag: Flag name (OVERFLOW, UNDERFLOW, DIVIDE_BY_ZERO, etc.)
    """
    flag: str = ''
    kind: str = 'ieee_flag'


@dataclass
class IEEERoundingMode(ScientificNode):
    """IEEE rounding mode.
    
    Attributes:
        mode: Mode name (NEAREST, TO_ZERO, UP, DOWN)
    """
    mode: str = ''
    kind: str = 'ieee_rounding_mode'


# =============================================================================
# Namelist I/O (Fortran)
# =============================================================================

@dataclass
class NamelistGroup(ScientificNode):
    """Fortran NAMELIST group.
    
    Attributes:
        name: Namelist group name
        variables: Variables in the namelist
    """
    name: str = ''
    variables: List[str] = field(default_factory=list)
    kind: str = 'namelist_group'


@dataclass
class NamelistRead(ScientificNode):
    """Read from namelist.
    
    Attributes:
        unit: I/O unit
        nml: Namelist group name
    """
    unit: 'Expr' = None
    nml: str = ''
    kind: str = 'namelist_read'


@dataclass
class NamelistWrite(ScientificNode):
    """Write to namelist.
    
    Attributes:
        unit: I/O unit
        nml: Namelist group name
    """
    unit: 'Expr' = None
    nml: str = ''
    kind: str = 'namelist_write'


# =============================================================================
# Numeric Conversion Utilities
# =============================================================================

def get_fortran_kind(precision_bits: int, is_real: bool = True) -> int:
    """Get Fortran KIND parameter from precision bits.
    
    Args:
        precision_bits: Number of bits (8, 16, 32, 64, 128)
        is_real: True for REAL, False for INTEGER
        
    Returns:
        Fortran KIND value
    """
    kind_map = {8: 1, 16: 2, 32: 4, 64: 8, 128: 16}
    return kind_map.get(precision_bits, 4)


def get_pascal_type(base: str, precision_bits: int) -> str:
    """Get Pascal type name from base type and precision.
    
    Args:
        base: Base type ('integer', 'real')
        precision_bits: Number of bits
        
    Returns:
        Pascal type name
    """
    if base == 'integer':
        int_map = {8: 'ShortInt', 16: 'SmallInt', 32: 'LongInt', 64: 'Int64'}
        return int_map.get(precision_bits, 'Integer')
    elif base == 'real':
        real_map = {32: 'Single', 64: 'Double', 80: 'Extended'}
        return real_map.get(precision_bits, 'Real')
    return 'Variant'


# Expression type alias
Expr = Any  # Forward reference
Statement = Any  # Forward reference
ArrayDimension = Any  # Forward reference
