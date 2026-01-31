//! STUNIR Language Family Emitters - Rust Implementation
//!
//! Emitters for language families (Lisp, Prolog, etc.).

pub mod lisp;
pub mod prolog;

// Re-export emitter types
pub use lisp::{LispConfig, LispDialect, LispEmitter};
pub use prolog::{PrologConfig, PrologDialect, PrologEmitter};
