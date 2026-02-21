"""STUNIR Scientific IR - Scientific and legacy language constructs.

This package provides IR nodes for scientific computing and legacy
programming languages including Fortran and Pascal.

Core modules:
- scientific_ir: Core IR classes for modules, subprograms, types, statements
- arrays: Array operations, slicing, and whole-array operations
- numerical: Numerical computing primitives, intrinsics, parallel constructs

Usage:
    from ir.scientific import Module, Subprogram, Parameter, TypeRef
    from ir.scientific import ArrayType, ArrayDimension, ArraySlice
    from ir.scientific import MathIntrinsic, DoConcurrent
    
    # Create a module
    mod = Module(
        name='math_utils',
        subprograms=[
            Subprogram(name='add', is_function=True)
        ]
    )
"""

# Core IR classes
from ir.scientific.scientific_ir import (
    # Enumerations
    Visibility,
    Intent,
    ParameterMode,
    ArrayOrder,
    
    # Base class
    ScientificNode,
    
    # Type references
    TypeRef,
    
    # Module/Program structure
    Import,
    Module,
    Program,
    
    # Subprograms
    Parameter,
    Interface,
    Subprogram,
    
    # Variable declarations
    VariableDecl,
    ConstantDecl,
    
    # Type declarations
    TypeDecl,
    FieldDecl,
    RecordType,
    RecordVariant,
    VariantRecord,
    EnumType,
    SetType,
    RangeType,
    PointerType,
    FileType,
    
    # Object Pascal
    MethodDecl,
    PropertyDecl,
    ClassType,
    
    # Statements
    Assignment,
    IfStatement,
    ElseIfPart,
    ForLoop,
    WhileLoop,
    RepeatLoop,
    CaseItem,
    CaseStatement,
    CallStatement,
    ReturnStatement,
    BlockStatement,
    NullStatement,
    ExitStatement,
    ContinueStatement,
    GotoStatement,
    LabeledStatement,
    WithStatement,
    ExceptionHandler,
    TryStatement,
    RaiseStatement,
    
    # File I/O
    OpenStatement,
    CloseStatement,
    ReadStatement,
    WriteStatement,
    
    # Expressions
    Literal,
    VarRef,
    BinaryOp,
    UnaryOp,
    FunctionCall,
    ArrayAccess,
    FieldAccess,
    SetExpr,
    SetOp,
    RangeExpr,
    TypeCast,
    SizeOfExpr,
    AllocateExpr,
    DeallocateExpr,
    PointerDeref,
    AddressOf,
    
    # Fortran allocation
    AllocateStatement,
    AllocationSpec,
    DeallocateStatement,
)

# Array operations
from ir.scientific.arrays import (
    ArrayDimension,
    ArrayType,
    CoarrayType,
    SliceSpec,
    ArraySlice,
    ArrayReshape,
    ArrayOpKind,
    ArrayOperation,
    ArrayConstructor,
    ImpliedDo,
    ArrayIntrinsicKind,
    ArrayIntrinsic,
    MatrixOperation,
    ArrayAssignment,
    WhereStatement,
    ForallStatement,
    ForallIndex,
    
    # Pascal-specific
    PascalArrayType,
    IndexRange,
    DynamicArray,
    SetLength,
    Length,
    High,
    Low,
    
    # String operations
    StringConcat,
    Substring,
)

# Numerical computing
from ir.scientific.numerical import (
    NumericType,
    ComplexType,
    ComplexLiteral,
    ComplexOp,
    INTRINSIC_MAP,
    FORTRAN_ARRAY_INTRINSICS,
    MathIntrinsic,
    TypeIntrinsic,
    CharIntrinsic,
    
    # Parallel constructs
    LoopIndex,
    ReduceSpec,
    LocalitySpec,
    DoConcurrent,
    
    # Coarray
    Coarray,
    CoarrayAccess,
    SyncAll,
    SyncImages,
    CriticalBlock,
    ImageIntrinsic,
    
    # IEEE
    IEEEIntrinsic,
    IEEEFlag,
    IEEERoundingMode,
    
    # Namelist
    NamelistGroup,
    NamelistRead,
    NamelistWrite,
    
    # Utilities
    get_fortran_kind,
    get_pascal_type,
)

__all__ = [
    # Enumerations
    'Visibility', 'Intent', 'ParameterMode', 'ArrayOrder',
    'ArrayOpKind', 'ArrayIntrinsicKind',
    
    # Base
    'ScientificNode',
    
    # Types
    'TypeRef', 'NumericType', 'ComplexType', 'ArrayType', 'CoarrayType',
    'RecordType', 'VariantRecord', 'EnumType', 'SetType', 'RangeType',
    'PointerType', 'FileType', 'ClassType', 'PascalArrayType', 'DynamicArray',
    
    # Structure
    'Module', 'Program', 'Import', 'Interface', 'Subprogram', 'Parameter',
    
    # Declarations
    'VariableDecl', 'ConstantDecl', 'TypeDecl', 'FieldDecl',
    'MethodDecl', 'PropertyDecl', 'RecordVariant',
    
    # Arrays
    'ArrayDimension', 'SliceSpec', 'ArraySlice', 'ArrayReshape',
    'ArrayOperation', 'ArrayConstructor', 'ImpliedDo', 'ArrayIntrinsic',
    'MatrixOperation', 'ArrayAssignment', 'WhereStatement', 'ForallStatement',
    'ForallIndex', 'IndexRange', 'SetLength', 'Length', 'High', 'Low',
    
    # Statements
    'Assignment', 'IfStatement', 'ElseIfPart', 'ForLoop', 'WhileLoop',
    'RepeatLoop', 'CaseItem', 'CaseStatement', 'CallStatement',
    'ReturnStatement', 'BlockStatement', 'NullStatement', 'ExitStatement',
    'ContinueStatement', 'GotoStatement', 'LabeledStatement', 'WithStatement',
    'ExceptionHandler', 'TryStatement', 'RaiseStatement',
    'OpenStatement', 'CloseStatement', 'ReadStatement', 'WriteStatement',
    'AllocateStatement', 'AllocationSpec', 'DeallocateStatement',
    
    # Expressions
    'Literal', 'VarRef', 'BinaryOp', 'UnaryOp', 'FunctionCall',
    'ArrayAccess', 'FieldAccess', 'SetExpr', 'SetOp', 'RangeExpr',
    'TypeCast', 'SizeOfExpr', 'AllocateExpr', 'DeallocateExpr',
    'PointerDeref', 'AddressOf', 'ComplexLiteral', 'ComplexOp',
    'StringConcat', 'Substring',
    
    # Numerical
    'INTRINSIC_MAP', 'FORTRAN_ARRAY_INTRINSICS',
    'MathIntrinsic', 'TypeIntrinsic', 'CharIntrinsic',
    
    # Parallel
    'LoopIndex', 'ReduceSpec', 'LocalitySpec', 'DoConcurrent',
    'Coarray', 'CoarrayAccess', 'SyncAll', 'SyncImages',
    'CriticalBlock', 'ImageIntrinsic',
    
    # IEEE
    'IEEEIntrinsic', 'IEEEFlag', 'IEEERoundingMode',
    
    # Namelist
    'NamelistGroup', 'NamelistRead', 'NamelistWrite',
    
    # Utilities
    'get_fortran_kind', 'get_pascal_type',
]
