//! Functional language emitters
//!
//! Supports: Haskell, Scala, F#, OCaml, Erlang, Elixir

use crate::types::*;

/// Functional language
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FunctionalLanguage {
    Haskell,
    Scala,
    FSharp,
    OCaml,
    Erlang,
    Elixir,
}

impl std::fmt::Display for FunctionalLanguage {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            FunctionalLanguage::Haskell => write!(f, "Haskell"),
            FunctionalLanguage::Scala => write!(f, "Scala"),
            FunctionalLanguage::FSharp => write!(f, "F#"),
            FunctionalLanguage::OCaml => write!(f, "OCaml"),
            FunctionalLanguage::Erlang => write!(f, "Erlang"),
            FunctionalLanguage::Elixir => write!(f, "Elixir"),
        }
    }
}

/// Emit functional code
pub fn emit(language: FunctionalLanguage, module_name: &str) -> EmitterResult<String> {
    match language {
        FunctionalLanguage::Haskell => emit_haskell(module_name),
        FunctionalLanguage::Scala => emit_scala(module_name),
        FunctionalLanguage::FSharp => emit_fsharp(module_name),
        FunctionalLanguage::OCaml => emit_ocaml(module_name),
        FunctionalLanguage::Erlang => emit_erlang(module_name),
        FunctionalLanguage::Elixir => emit_elixir(module_name),
    }
}

fn emit_haskell(module_name: &str) -> EmitterResult<String> {
    let mut code = String::new();
    
    code.push_str("-- STUNIR Generated Haskell\n");
    code.push_str(&format!("-- Module: {}\n", module_name));
    code.push_str("-- Generator: Rust Pipeline\n\n");
    
    code.push_str(&format!("module {} where\n\n", module_name));
    
    code.push_str("-- Example function\n");
    code.push_str("factorial :: Integer -> Integer\n");
    code.push_str("factorial 0 = 1\n");
    code.push_str("factorial n = n * factorial (n - 1)\n\n");
    
    code.push_str("-- Map example\n");
    code.push_str("doubleList :: [Int] -> [Int]\n");
    code.push_str("doubleList = map (*2)\n");
    
    Ok(code)
}

fn emit_scala(module_name: &str) -> EmitterResult<String> {
    let mut code = String::new();
    
    code.push_str("// STUNIR Generated Scala\n");
    code.push_str(&format!("// Module: {}\n", module_name));
    code.push_str("// Generator: Rust Pipeline\n\n");
    
    code.push_str(&format!("object {} {{\n", module_name));
    code.push_str("  // Example function\n");
    code.push_str("  def factorial(n: Int): Int = n match {\n");
    code.push_str("    case 0 => 1\n");
    code.push_str("    case _ => n * factorial(n - 1)\n");
    code.push_str("  }\n\n");
    
    code.push_str("  // Map example\n");
    code.push_str("  def doubleList(xs: List[Int]): List[Int] =\n");
    code.push_str("    xs.map(_ * 2)\n");
    code.push_str("}\n");
    
    Ok(code)
}

fn emit_fsharp(module_name: &str) -> EmitterResult<String> {
    let mut code = String::new();
    
    code.push_str("// STUNIR Generated F#\n");
    code.push_str(&format!("// Module: {}\n", module_name));
    code.push_str("// Generator: Rust Pipeline\n\n");
    
    code.push_str(&format!("module {}\n\n", module_name));
    
    code.push_str("// Example function\n");
    code.push_str("let rec factorial n =\n");
    code.push_str("    match n with\n");
    code.push_str("    | 0 -> 1\n");
    code.push_str("    | _ -> n * factorial (n - 1)\n\n");
    
    code.push_str("// Map example\n");
    code.push_str("let doubleList xs =\n");
    code.push_str("    List.map (fun x -> x * 2) xs\n");
    
    Ok(code)
}

fn emit_ocaml(module_name: &str) -> EmitterResult<String> {
    let mut code = String::new();
    
    code.push_str("(* STUNIR Generated OCaml *)\n");
    code.push_str(&format!("(* Module: {} *)\n", module_name));
    code.push_str("(* Generator: Rust Pipeline *)\n\n");
    
    code.push_str("(* Example function *)\n");
    code.push_str("let rec factorial n =\n");
    code.push_str("  match n with\n");
    code.push_str("  | 0 -> 1\n");
    code.push_str("  | _ -> n * factorial (n - 1)\n\n");
    
    code.push_str("(* Map example *)\n");
    code.push_str("let double_list xs =\n");
    code.push_str("  List.map (fun x -> x * 2) xs\n");
    
    Ok(code)
}

fn emit_erlang(module_name: &str) -> EmitterResult<String> {
    let mut code = String::new();
    
    code.push_str("% STUNIR Generated Erlang\n");
    code.push_str(&format!("% Module: {}\n", module_name));
    code.push_str("% Generator: Rust Pipeline\n\n");
    
    code.push_str(&format!("-module({}).\n", module_name.to_lowercase()));
    code.push_str("-export([factorial/1, double_list/1]).\n\n");
    
    code.push_str("% Example function\n");
    code.push_str("factorial(0) -> 1;\n");
    code.push_str("factorial(N) when N > 0 -> N * factorial(N - 1).\n\n");
    
    code.push_str("% Map example\n");
    code.push_str("double_list(Xs) ->\n");
    code.push_str("    lists:map(fun(X) -> X * 2 end, Xs).\n");
    
    Ok(code)
}

fn emit_elixir(module_name: &str) -> EmitterResult<String> {
    let mut code = String::new();
    
    code.push_str("# STUNIR Generated Elixir\n");
    code.push_str(&format!("# Module: {}\n", module_name));
    code.push_str("# Generator: Rust Pipeline\n\n");
    
    code.push_str(&format!("defmodule {} do\n", module_name));
    code.push_str("  # Example function\n");
    code.push_str("  def factorial(0), do: 1\n");
    code.push_str("  def factorial(n) when n > 0, do: n * factorial(n - 1)\n\n");
    
    code.push_str("  # Map example\n");
    code.push_str("  def double_list(xs) do\n");
    code.push_str("    Enum.map(xs, fn x -> x * 2 end)\n");
    code.push_str("  end\n");
    code.push_str("end\n");
    
    Ok(code)
}
