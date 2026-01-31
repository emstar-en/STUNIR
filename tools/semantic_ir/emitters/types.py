"""STUNIR Semantic IR Types - Python Reference Implementation

Core types and enumerations for semantic IR emitters.
Based on Ada SPARK Emitter_Types package.
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional


class IRDataType(Enum):
    """IR data types matching SPARK IR_Data_Type enumeration."""
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
    CHAR = "char"
    STRING = "string"
    POINTER = "pointer"
    ARRAY = "array"
    STRUCT = "struct"


class IRStatementType(Enum):
    """IR statement types matching SPARK IR_Statement_Type enumeration."""
    NOP = "nop"
    VAR_DECL = "var_decl"
    ASSIGN = "assign"
    RETURN = "return"
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    CALL = "call"
    IF = "if"
    LOOP = "loop"
    BREAK = "break"
    CONTINUE = "continue"
    BLOCK = "block"


class Architecture(Enum):
    """Architecture types matching SPARK Architecture_Type enumeration."""
    ARM = "arm"
    ARM64 = "arm64"
    AVR = "avr"
    MIPS = "mips"
    RISCV = "riscv"
    X86 = "x86"
    X86_64 = "x86_64"
    POWERPC = "powerpc"
    GENERIC = "generic"


class Endianness(Enum):
    """Endianness types matching SPARK Endianness_Type enumeration."""
    LITTLE = "little"
    BIG = "big"


@dataclass
class ArchConfig:
    """Architecture configuration matching SPARK Arch_Config_Type record."""
    word_size: int  # 8 to 64 bits
    endianness: Endianness
    alignment: int  # 1 to 16 bytes
    stack_grows_down: bool

    def __post_init__(self):
        """Validate configuration values."""
        if not (8 <= self.word_size <= 64):
            raise ValueError(f"Invalid word_size: {self.word_size} (must be 8-64)")
        if not (1 <= self.alignment <= 16):
            raise ValueError(f"Invalid alignment: {self.alignment} (must be 1-16)")


@dataclass
class IRStatement:
    """IR statement representation."""
    stmt_type: IRStatementType
    data_type: Optional[IRDataType] = None
    target: Optional[str] = None
    value: Optional[str] = None
    left_op: Optional[str] = None
    right_op: Optional[str] = None


@dataclass
class IRParameter:
    """Function parameter representation."""
    name: str
    param_type: IRDataType


@dataclass
class IRFunction:
    """Function representation."""
    name: str
    return_type: IRDataType
    parameters: List[IRParameter]
    statements: List[IRStatement]
    docstring: Optional[str] = None


@dataclass
class IRTypeField:
    """Type field representation."""
    name: str
    field_type: str
    optional: bool = False


@dataclass
class IRType:
    """Custom type/struct definition."""
    name: str
    fields: List[IRTypeField]
    docstring: Optional[str] = None


@dataclass
class IRModule:
    """Complete IR module representation matching STUNIR IR schema."""
    ir_version: str
    module_name: str
    types: List[IRType]
    functions: List[IRFunction]
    docstring: Optional[str] = None


@dataclass
class GeneratedFile:
    """Generated file record matching SPARK Generated_File_Record."""
    path: str
    hash: str
    size: int


# Architecture configuration presets
ARCH_CONFIGS = {
    Architecture.ARM: ArchConfig(
        word_size=32,
        endianness=Endianness.LITTLE,
        alignment=4,
        stack_grows_down=True,
    ),
    Architecture.ARM64: ArchConfig(
        word_size=64,
        endianness=Endianness.LITTLE,
        alignment=8,
        stack_grows_down=True,
    ),
    Architecture.AVR: ArchConfig(
        word_size=8,
        endianness=Endianness.LITTLE,
        alignment=1,
        stack_grows_down=True,
    ),
    Architecture.MIPS: ArchConfig(
        word_size=32,
        endianness=Endianness.BIG,
        alignment=4,
        stack_grows_down=True,
    ),
    Architecture.RISCV: ArchConfig(
        word_size=32,
        endianness=Endianness.LITTLE,
        alignment=4,
        stack_grows_down=True,
    ),
    Architecture.X86: ArchConfig(
        word_size=32,
        endianness=Endianness.LITTLE,
        alignment=4,
        stack_grows_down=True,
    ),
    Architecture.X86_64: ArchConfig(
        word_size=64,
        endianness=Endianness.LITTLE,
        alignment=8,
        stack_grows_down=True,
    ),
    Architecture.POWERPC: ArchConfig(
        word_size=32,
        endianness=Endianness.BIG,
        alignment=4,
        stack_grows_down=True,
    ),
    Architecture.GENERIC: ArchConfig(
        word_size=32,
        endianness=Endianness.LITTLE,
        alignment=4,
        stack_grows_down=True,
    ),
}


def map_ir_type_to_c(ir_type: IRDataType) -> str:
    """Map IR data type to C type name (matching SPARK Map_IR_Type_To_C)."""
    mapping = {
        IRDataType.VOID: "void",
        IRDataType.BOOL: "bool",
        IRDataType.I8: "int8_t",
        IRDataType.I16: "int16_t",
        IRDataType.I32: "int32_t",
        IRDataType.I64: "int64_t",
        IRDataType.U8: "uint8_t",
        IRDataType.U16: "uint16_t",
        IRDataType.U32: "uint32_t",
        IRDataType.U64: "uint64_t",
        IRDataType.F32: "float",
        IRDataType.F64: "double",
        IRDataType.CHAR: "char",
        IRDataType.STRING: "char*",
        IRDataType.POINTER: "void*",
        IRDataType.ARRAY: "array",
        IRDataType.STRUCT: "struct",
    }
    return mapping.get(ir_type, "void")


def get_arch_config(arch: Architecture) -> ArchConfig:
    """Get architecture configuration (matching SPARK Get_Arch_Config)."""
    return ARCH_CONFIGS.get(arch, ARCH_CONFIGS[Architecture.GENERIC])
