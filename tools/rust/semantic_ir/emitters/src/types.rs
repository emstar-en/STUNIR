//! STUNIR Semantic IR Types - Rust Implementation
//!
//! Core types and enumerations for semantic IR emitters.
//! Based on Ada SPARK Emitter_Types package.

use serde::{Deserialize, Serialize};
use std::fmt;

/// IR data types matching SPARK IR_Data_Type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IRDataType {
    /// Void type
    Void,
    /// Boolean type
    Bool,
    /// 8-bit signed integer
    I8,
    /// 16-bit signed integer
    I16,
    /// 32-bit signed integer
    I32,
    /// 64-bit signed integer
    I64,
    /// 8-bit unsigned integer
    U8,
    /// 16-bit unsigned integer
    U16,
    /// 32-bit unsigned integer
    U32,
    /// 64-bit unsigned integer
    U64,
    /// 32-bit floating point
    F32,
    /// 64-bit floating point
    F64,
    /// Character type
    Char,
    /// String type
    String,
    /// Pointer type
    Pointer,
    /// Array type
    Array,
    /// Struct type
    Struct,
}

/// IR statement types matching SPARK IR_Statement_Type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IRStatementType {
    /// No operation
    Nop,
    /// Variable declaration
    VarDecl,
    /// Assignment
    Assign,
    /// Return statement
    Return,
    /// Addition
    Add,
    /// Subtraction
    Sub,
    /// Multiplication
    Mul,
    /// Division
    Div,
    /// Function call
    Call,
    /// If statement
    If,
    /// Loop
    Loop,
    /// Break
    Break,
    /// Continue
    Continue,
    /// Block
    Block,
}

/// Architecture types matching SPARK Architecture_Type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Architecture {
    /// ARM 32-bit
    ARM,
    /// ARM 64-bit
    ARM64,
    /// AVR microcontroller
    AVR,
    /// MIPS
    MIPS,
    /// RISC-V
    RISCV,
    /// x86 32-bit
    X86,
    /// x86 64-bit
    X86_64,
    /// PowerPC
    PowerPC,
    /// Generic architecture
    Generic,
}

/// Endianness types matching SPARK Endianness_Type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Endianness {
    /// Little-endian
    Little,
    /// Big-endian
    Big,
}

/// Architecture configuration matching SPARK Arch_Config_Type record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchConfig {
    /// Word size in bits (8-64)
    pub word_size: u8,
    /// Endianness
    pub endianness: Endianness,
    /// Alignment in bytes (1-16)
    pub alignment: u8,
    /// Stack grows downward
    pub stack_grows_down: bool,
}

/// IR statement representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRStatement {
    /// Statement type
    pub stmt_type: IRStatementType,
    /// Data type (optional)
    pub data_type: Option<IRDataType>,
    /// Target identifier (optional)
    pub target: Option<String>,
    /// Value (optional)
    pub value: Option<String>,
    /// Left operand (optional)
    pub left_op: Option<String>,
    /// Right operand (optional)
    pub right_op: Option<String>,
}

/// Function parameter representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRParameter {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub param_type: IRDataType,
}

/// Function representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRFunction {
    /// Function name
    pub name: String,
    /// Return type
    pub return_type: IRDataType,
    /// Parameters
    pub parameters: Vec<IRParameter>,
    /// Statements
    pub statements: Vec<IRStatement>,
    /// Docstring (optional)
    pub docstring: Option<String>,
}

/// Type field representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRTypeField {
    /// Field name
    pub name: String,
    /// Field type
    pub field_type: String,
    /// Is optional
    pub optional: bool,
}

/// Custom type/struct definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRType {
    /// Type name
    pub name: String,
    /// Fields
    pub fields: Vec<IRTypeField>,
    /// Docstring (optional)
    pub docstring: Option<String>,
}

/// Complete IR module representation matching STUNIR IR schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRModule {
    /// IR version
    pub ir_version: String,
    /// Module name
    pub module_name: String,
    /// Types
    pub types: Vec<IRType>,
    /// Functions
    pub functions: Vec<IRFunction>,
    /// Docstring (optional)
    pub docstring: Option<String>,
}

/// Generated file record matching SPARK Generated_File_Record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedFile {
    /// File path
    pub path: String,
    /// SHA-256 hash
    pub hash: String,
    /// File size in bytes
    pub size: usize,
}

/// Map IR data type to C type name (matching SPARK Map_IR_Type_To_C)
pub fn map_ir_type_to_c(ir_type: IRDataType) -> &'static str {
    match ir_type {
        IRDataType::Void => "void",
        IRDataType::Bool => "bool",
        IRDataType::I8 => "int8_t",
        IRDataType::I16 => "int16_t",
        IRDataType::I32 => "int32_t",
        IRDataType::I64 => "int64_t",
        IRDataType::U8 => "uint8_t",
        IRDataType::U16 => "uint16_t",
        IRDataType::U32 => "uint32_t",
        IRDataType::U64 => "uint64_t",
        IRDataType::F32 => "float",
        IRDataType::F64 => "double",
        IRDataType::Char => "char",
        IRDataType::String => "char*",
        IRDataType::Pointer => "void*",
        IRDataType::Array => "array",
        IRDataType::Struct => "struct",
    }
}

/// Get architecture configuration (matching SPARK Get_Arch_Config)
pub fn get_arch_config(arch: Architecture) -> ArchConfig {
    match arch {
        Architecture::ARM => ArchConfig {
            word_size: 32,
            endianness: Endianness::Little,
            alignment: 4,
            stack_grows_down: true,
        },
        Architecture::ARM64 => ArchConfig {
            word_size: 64,
            endianness: Endianness::Little,
            alignment: 8,
            stack_grows_down: true,
        },
        Architecture::AVR => ArchConfig {
            word_size: 8,
            endianness: Endianness::Little,
            alignment: 1,
            stack_grows_down: true,
        },
        Architecture::MIPS => ArchConfig {
            word_size: 32,
            endianness: Endianness::Big,
            alignment: 4,
            stack_grows_down: true,
        },
        Architecture::RISCV => ArchConfig {
            word_size: 32,
            endianness: Endianness::Little,
            alignment: 4,
            stack_grows_down: true,
        },
        Architecture::X86 => ArchConfig {
            word_size: 32,
            endianness: Endianness::Little,
            alignment: 4,
            stack_grows_down: true,
        },
        Architecture::X86_64 => ArchConfig {
            word_size: 64,
            endianness: Endianness::Little,
            alignment: 8,
            stack_grows_down: true,
        },
        Architecture::PowerPC => ArchConfig {
            word_size: 32,
            endianness: Endianness::Big,
            alignment: 4,
            stack_grows_down: true,
        },
        Architecture::Generic => ArchConfig {
            word_size: 32,
            endianness: Endianness::Little,
            alignment: 4,
            stack_grows_down: true,
        },
    }
}
