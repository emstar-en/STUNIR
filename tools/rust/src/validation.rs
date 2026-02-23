//! STUNIR Semantic IR Validation
//!
//! Validation functions with detailed error reporting.

use serde::{Deserialize, Serialize};

use crate::types::*;
use crate::nodes::*;
use crate::modules::*;

/// Validation status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ValidationStatus {
    Valid,
    Invalid,
    Warning,
}

/// Validation result
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidationResult {
    pub status: ValidationStatus,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub message: String,
}

impl ValidationResult {
    /// Create a valid result
    pub fn valid() -> Self {
        Self {
            status: ValidationStatus::Valid,
            message: String::new(),
        }
    }

    /// Create an invalid result
    pub fn invalid(message: impl Into<String>) -> Self {
        Self {
            status: ValidationStatus::Invalid,
            message: message.into(),
        }
    }

    /// Create a warning result
    pub fn warning(message: impl Into<String>) -> Self {
        Self {
            status: ValidationStatus::Warning,
            message: message.into(),
        }
    }

    /// Check if valid
    pub fn is_valid(&self) -> bool {
        self.status == ValidationStatus::Valid
    }
}

/// Validate node ID format
pub fn validate_node_id(node_id: &str) -> ValidationResult {
    if IRNodeBase::is_valid_node_id(node_id) {
        ValidationResult::valid()
    } else {
        ValidationResult::invalid(
            format!("Invalid node ID '{}': must start with 'n_'", node_id)
        )
    }
}

/// Validate hash format
pub fn validate_hash(hash: &str) -> ValidationResult {
    if IRNodeBase::is_valid_hash(hash) {
        ValidationResult::valid()
    } else {
        ValidationResult::invalid(
            format!("Invalid hash '{}': must be 'sha256:' + 64 hex chars", hash)
        )
    }
}

/// Validate type reference
pub fn validate_type_reference(type_ref: &TypeReference) -> ValidationResult {
    match type_ref {
        TypeReference::PrimitiveType { .. } => ValidationResult::valid(),
        TypeReference::TypeRef { name, .. } => {
            if name.is_empty() {
                ValidationResult::invalid("Type reference must have non-empty name")
            } else {
                ValidationResult::valid()
            }
        }
    }
}

/// Validate module structure
pub fn validate_module(module: &IRModule) -> ValidationResult {
    // Validate node ID
    let id_result = validate_node_id(&module.base.node_id);
    if !id_result.is_valid() {
        return id_result;
    }

    // Validate hash if present
    if let Some(ref hash) = module.base.hash {
        let hash_result = validate_hash(hash);
        if !hash_result.is_valid() {
            return hash_result;
        }
    }

    // Validate module name
    if module.name.is_empty() {
        return ValidationResult::invalid("Module must have non-empty name");
    }

    ValidationResult::valid()
}

/// Check type compatibility
pub fn check_type_compatibility(type1: &TypeReference, type2: &TypeReference) -> bool {
    match (type1, type2) {
        (
            TypeReference::PrimitiveType { primitive: p1 },
            TypeReference::PrimitiveType { primitive: p2 },
        ) => p1 == p2,
        (
            TypeReference::TypeRef { name: n1, .. },
            TypeReference::TypeRef { name: n2, .. },
        ) => n1 == n2,
        _ => false,
    }
}

/// Check binary operator type validity
pub fn check_binary_op_types(
    op: BinaryOperator,
    left: &TypeReference,
    right: &TypeReference,
) -> ValidationResult {
    // Both must be primitive types
    let (left_prim, right_prim) = match (left, right) {
        (
            TypeReference::PrimitiveType { primitive: l },
            TypeReference::PrimitiveType { primitive: r },
        ) => (l, r),
        _ => return ValidationResult::invalid("Binary operators require primitive types"),
    };

    use IRPrimitiveType::*;

    match op {
        BinaryOperator::Add
        | BinaryOperator::Sub
        | BinaryOperator::Mul
        | BinaryOperator::Div
        | BinaryOperator::Mod => {
            // Arithmetic operators need numeric types
            let numeric_types = [I8, I16, I32, I64, U8, U16, U32, U64, F32, F64];
            if numeric_types.contains(left_prim) && numeric_types.contains(right_prim) {
                ValidationResult::valid()
            } else {
                ValidationResult::invalid("Arithmetic operators require numeric types")
            }
        }
        BinaryOperator::Eq
        | BinaryOperator::Neq
        | BinaryOperator::Lt
        | BinaryOperator::Leq
        | BinaryOperator::Gt
        | BinaryOperator::Geq => {
            // Comparison operators need compatible types
            if check_type_compatibility(left, right) {
                ValidationResult::valid()
            } else {
                ValidationResult::invalid("Comparison operators require compatible types")
            }
        }
        BinaryOperator::And | BinaryOperator::Or => {
            // Logical operators need boolean types
            if *left_prim == Bool && *right_prim == Bool {
                ValidationResult::valid()
            } else {
                ValidationResult::invalid("Logical operators require boolean types")
            }
        }
        BinaryOperator::BitAnd
        | BinaryOperator::BitOr
        | BinaryOperator::BitXor
        | BinaryOperator::Shl
        | BinaryOperator::Shr => {
            // Bitwise operators need integer types
            let integer_types = [I8, I16, I32, I64, U8, U16, U32, U64];
            if integer_types.contains(left_prim) && integer_types.contains(right_prim) {
                ValidationResult::valid()
            } else {
                ValidationResult::invalid("Bitwise operators require integer types")
            }
        }
        BinaryOperator::Assign => {
            // Assignment requires compatible types
            if check_type_compatibility(left, right) {
                ValidationResult::valid()
            } else {
                ValidationResult::invalid("Assignment requires compatible types")
            }
        }
    }
}
