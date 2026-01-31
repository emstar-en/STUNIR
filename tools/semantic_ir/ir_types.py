"""STUNIR Semantic IR Core Types.

DO-178C Level A Compliant
Pydantic-based type definitions with runtime validation.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, constr, conint


# Type aliases with validation
IRName = constr(min_length=1, max_length=256)
IRHash = constr(pattern=r"^sha256:[a-f0-9]{64}$")
IRPath = constr(min_length=1, max_length=512)
NodeID = constr(pattern=r"^n_[a-zA-Z0-9_]+$")


class IRPrimitiveType(str, Enum):
    """Primitive type enumeration."""
    VOID = "void"
    BOOL = "bool"
    I8 = "i8"
    I16 = "i16"
    I32 = "i32"
    I64 = "i64"
    U8 = "u8"
    U16 = "u16"
    U32 = "u32"
    U64 = "u64"
    F32 = "f32"
    F64 = "f64"
    STRING = "string"
    CHAR = "char"


class IRNodeKind(str, Enum):
    """Node kind discriminator."""
    # Module
    MODULE = "module"
    # Declarations
    FUNCTION_DECL = "function_decl"
    TYPE_DECL = "type_decl"
    CONST_DECL = "const_decl"
    VAR_DECL = "var_decl"
    # Statements
    BLOCK_STMT = "block_stmt"
    EXPR_STMT = "expr_stmt"
    IF_STMT = "if_stmt"
    WHILE_STMT = "while_stmt"
    FOR_STMT = "for_stmt"
    RETURN_STMT = "return_stmt"
    BREAK_STMT = "break_stmt"
    CONTINUE_STMT = "continue_stmt"
    VAR_DECL_STMT = "var_decl_stmt"
    ASSIGN_STMT = "assign_stmt"
    # Expressions
    INTEGER_LITERAL = "integer_literal"
    FLOAT_LITERAL = "float_literal"
    STRING_LITERAL = "string_literal"
    BOOL_LITERAL = "bool_literal"
    VAR_REF = "var_ref"
    BINARY_EXPR = "binary_expr"
    UNARY_EXPR = "unary_expr"
    FUNCTION_CALL = "function_call"
    MEMBER_EXPR = "member_expr"
    ARRAY_ACCESS = "array_access"
    CAST_EXPR = "cast_expr"
    TERNARY_EXPR = "ternary_expr"
    ARRAY_INIT = "array_init"
    STRUCT_INIT = "struct_init"


class BinaryOperator(str, Enum):
    """Binary operators."""
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    MOD = "%"
    EQ = "=="
    NEQ = "!="
    LT = "<"
    LEQ = "<="
    GT = ">"
    GEQ = ">="
    AND = "&&"
    OR = "||"
    BIT_AND = "&"
    BIT_OR = "|"
    BIT_XOR = "^"
    SHL = "<<"
    SHR = ">>"
    ASSIGN = "="


class UnaryOperator(str, Enum):
    """Unary operators."""
    NEG = "-"
    NOT = "!"
    BIT_NOT = "~"
    PRE_INC = "++"
    PRE_DEC = "--"
    POST_INC = "++"
    POST_DEC = "--"
    DEREF = "*"
    ADDR_OF = "&"


class StorageClass(str, Enum):
    """Storage class."""
    AUTO = "auto"
    STATIC = "static"
    EXTERN = "extern"
    REGISTER = "register"
    STACK = "stack"
    HEAP = "heap"
    GLOBAL = "global"


class VisibilityKind(str, Enum):
    """Visibility kind."""
    PUBLIC = "public"
    PRIVATE = "private"
    PROTECTED = "protected"
    INTERNAL = "internal"


class MutabilityKind(str, Enum):
    """Mutability kind."""
    MUTABLE = "mutable"
    IMMUTABLE = "immutable"
    CONST = "const"


class InlineHint(str, Enum):
    """Inline hint."""
    ALWAYS = "always"
    NEVER = "never"
    HINT = "hint"
    NONE = "none"


class TargetCategory(str, Enum):
    """Target categories."""
    EMBEDDED = "embedded"
    REALTIME = "realtime"
    SAFETY_CRITICAL = "safety_critical"
    GPU = "gpu"
    WASM = "wasm"
    NATIVE = "native"
    JIT = "jit"
    INTERPRETER = "interpreter"
    FUNCTIONAL = "functional"
    LOGIC = "logic"
    CONSTRAINT = "constraint"
    DATAFLOW = "dataflow"
    REACTIVE = "reactive"
    QUANTUM = "quantum"
    NEUROMORPHIC = "neuromorphic"


class SafetyLevel(str, Enum):
    """Safety level."""
    NONE = "None"
    DO178C_D = "DO-178C_Level_D"
    DO178C_C = "DO-178C_Level_C"
    DO178C_B = "DO-178C_Level_B"
    DO178C_A = "DO-178C_Level_A"


class SourceLocation(BaseModel):
    """Source location information."""
    file: IRPath
    line: conint(ge=1)
    column: conint(ge=1)
    length: conint(ge=0) = 0

    class Config:
        frozen = True
