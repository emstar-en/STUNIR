//! Lisp family emitters
//!
//! Supports: Common Lisp, Scheme, Clojure, Racket, Emacs Lisp, Guile, Hy, Janet

use crate::types::*;

pub mod common_lisp;
pub mod scheme;
pub mod clojure;

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
}

pub fn emit(dialect: LispDialect, module_name: &str) -> EmitterResult<String> {
    match dialect {
        LispDialect::CommonLisp => common_lisp::emit(module_name),
        LispDialect::Scheme => scheme::emit(module_name),
        LispDialect::Clojure => clojure::emit(module_name),
        _ => Err(EmitterError::UnsupportedTarget(format!("{:?}", dialect))),
    }
}
