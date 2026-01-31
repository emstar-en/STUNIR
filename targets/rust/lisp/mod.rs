//! Lisp family emitters
//!
//! Supports: Common Lisp, Scheme, Clojure, Racket, Emacs Lisp, Guile, Hy, Janet

use crate::types::*;

pub mod common_lisp;
pub mod scheme;
pub mod clojure;
pub mod racket;
pub mod emacs_lisp;
pub mod guile;
pub mod hy;
pub mod janet;

/// Lisp dialect enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LispDialect {
    CommonLisp,
    Scheme,
    Clojure,
    Racket,
    EmacsLisp,
    Guile,
    Hy,
    Janet,
}

impl LispDialect {
    /// Get comment prefix for dialect
    pub fn comment_prefix(&self) -> &'static str {
        match self {
            LispDialect::CommonLisp => ";;; ",
            LispDialect::Scheme => ";; ",
            LispDialect::Clojure => ";; ",
            LispDialect::Racket => ";; ",
            LispDialect::EmacsLisp => ";;; ",
            LispDialect::Guile => ";; ",
            LispDialect::Hy => "; ",
            LispDialect::Janet => "# ",
        }
    }
    
    /// Get file extension for dialect
    pub fn file_extension(&self) -> &'static str {
        match self {
            LispDialect::CommonLisp => ".lisp",
            LispDialect::Scheme => ".scm",
            LispDialect::Clojure => ".clj",
            LispDialect::Racket => ".rkt",
            LispDialect::EmacsLisp => ".el",
            LispDialect::Guile => ".scm",
            LispDialect::Hy => ".hy",
            LispDialect::Janet => ".janet",
        }
    }
}

/// Emit code for specified Lisp dialect
pub fn emit(dialect: LispDialect, module_name: &str) -> EmitterResult<String> {
    match dialect {
        LispDialect::CommonLisp => common_lisp::emit(module_name),
        LispDialect::Scheme => scheme::emit(module_name),
        LispDialect::Clojure => clojure::emit(module_name),
        LispDialect::Racket => racket::emit(module_name),
        LispDialect::EmacsLisp => emacs_lisp::emit(module_name),
        LispDialect::Guile => guile::emit(module_name),
        LispDialect::Hy => hy::emit(module_name),
        LispDialect::Janet => janet::emit(module_name),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_dialects() {
        let dialects = vec![
            LispDialect::CommonLisp,
            LispDialect::Scheme,
            LispDialect::Clojure,
            LispDialect::Racket,
            LispDialect::EmacsLisp,
            LispDialect::Guile,
            LispDialect::Hy,
            LispDialect::Janet,
        ];
        
        for dialect in dialects {
            let result = emit(dialect, "test_module");
            assert!(result.is_ok(), "Failed to emit {:?}", dialect);
        }
    }
    
    #[test]
    fn test_comment_prefixes() {
        assert_eq!(LispDialect::CommonLisp.comment_prefix(), ";;; ");
        assert_eq!(LispDialect::Janet.comment_prefix(), "# ");
    }
    
    #[test]
    fn test_file_extensions() {
        assert_eq!(LispDialect::CommonLisp.file_extension(), ".lisp");
        assert_eq!(LispDialect::Racket.file_extension(), ".rkt");
    }
}
