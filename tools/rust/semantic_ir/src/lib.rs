//! STUNIR Semantic IR - Rust Implementation
//!
//! DO-178C Level A Compliant
//! Type-safe intermediate representation with serde serialization.

pub mod types;
pub mod nodes;
pub mod expressions;
pub mod statements;
pub mod declarations;
pub mod modules;
pub mod validation;

pub use types::*;
pub use nodes::*;
pub use expressions::*;
pub use statements::*;
pub use declarations::*;
pub use modules::*;
pub use validation::*;

/// Schema version
pub const SCHEMA_VERSION: &str = "1.0.0";

/// IR version
pub const IR_VERSION: (u8, u8, u8) = (1, 0, 0);
