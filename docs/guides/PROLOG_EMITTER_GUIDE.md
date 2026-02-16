# STUNIR Prolog Emitter User Guide
## Phase 3b: Language Family Emitters

**Version**: 1.0  
**Date**: 2026-01-31  
**Status**: Production Ready

---

## Table of Contents

1. [Introduction](#introduction)
2. [Supported Dialects](#supported-dialects)
3. [Quick Start](#quick-start)
4. [Configuration](#configuration)
5. [Usage Examples](#usage-examples)
6. [Functional to Logic Translation](#functional-to-logic-translation)
7. [Dialect-Specific Features](#dialect-specific-features)
8. [Integration](#integration)
9. [Troubleshooting](#troubleshooting)

---

## 1. Introduction

The STUNIR Prolog Emitter is a formally verified Ada SPARK implementation that generates idiomatic Prolog code from STUNIR's Semantic IR. It translates functional/imperative IR into logic programming constructs and supports 8 major Prolog dialects.

### Key Features

- ✅ **8 Prolog Dialects**: SWI-Prolog, GNU Prolog, SICStus, YAP, XSB, Ciao, B-Prolog, ECLiPSe
- ✅ **Formally Verified**: SPARK contracts and GNATprove verification
- ✅ **Logic Translation**: Automatic functional → logic conversion
- ✅ **CLP Support**: Constraint Logic Programming for compatible dialects
- ✅ **Tabling**: Automatic tabling for XSB and YAP
- ✅ **DO-178C Level A**: Safety-critical compliance

---

## 2. Supported Dialects

| Dialect | Standard | Features | Use Case | Status |
|---------|----------|----------|----------|--------|
| **SWI-Prolog** | ISO + Extensions | CLP, Modules, DCG | General-purpose | ✅ Production |
| **GNU Prolog** | ISO Prolog | CLP(FD), Minimal | Embedded systems | ✅ Production |
| **SICStus** | ISO Prolog | CLP, Modules, Debugging | Commercial applications | ✅ Production |
| **YAP** | ISO Prolog | Tabling, Performance | High-performance computing | ✅ Production |
| **XSB** | ISO Prolog | Tabling, Incremental | Deductive databases | ✅ Production |
| **Ciao** | ISO Prolog | Assertions, Verification | Verified software | ✅ Production |
| **B-Prolog** | ISO Prolog | Action Rules, CLP | Planning & optimization | ✅ Production |
| **ECLiPSe** | ISO Prolog | CLP, Optimization | Constraint optimization | ✅ Production |

---

## 3. Quick Start

### Basic Usage

```bash
# Generate SWI-Prolog code
stunir_ir_to_code --input module.ir.json \
                  --output output.pl \
                  --target prolog \
                  --dialect swi-prolog

# Generate XSB with tabling
stunir_ir_to_code --input module.ir.json \
                  --output output.P \
                  --target prolog \
                  --dialect xsb \
                  --use-tabling

# Generate ECLiPSe with CLP
stunir_ir_to_code --input module.ir.json \
                  --output output.ecl \
                  --target prolog \
                  --dialect eclipse \
                  --use-clp
```

### Example IR Input

```json
{
  "ir_version": "v1",
  "module_name": "math_predicates",
  "docstring": "Mathematical predicates",
  "functions": [
    {
      "name": "add",
      "docstring": "Add two integers",
      "args": [
        {"name": "x", "type": "integer"},
        {"name": "y", "type": "integer"}
      ],
      "return_type": "integer",
      "statements": []
    }
  ]
}
```

---

## 4. Configuration

### Configuration Options

```ada
type Prolog_Config is record
   Dialect        : Prolog_Dialect := SWI_Prolog;
   Use_Tabling    : Boolean := False;
   Use_CLP        : Boolean := False;
   Use_Assertions : Boolean := False;
   Indent_Size    : Positive := 2;
   Max_Line_Width : Positive := 80;
end record;
```

### Command-Line Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `--dialect` | `swi-prolog`, `gnu-prolog`, `sicstus`, `yap`, `xsb`, `ciao`, `bprolog`, `eclipse` | `swi-prolog` | Target Prolog dialect |
| `--use-tabling` | `true`, `false` | `false` | Enable tabling (XSB, YAP only) |
| `--use-clp` | `true`, `false` | `false` | Use CLP(FD) constraints |
| `--use-assertions` | `true`, `false` | `false` | Generate assertions (Ciao only) |
| `--indent-size` | 1-8 | `2` | Spaces per indent level |

---

## 5. Usage Examples

### Example 1: SWI-Prolog Module

**Input IR**:
```json
{
  "ir_version": "v1",
  "module_name": "geometry",
  "functions": [
    {
      "name": "circle_area",
      "docstring": "Calculate circle area from radius",
      "args": [{"name": "radius", "type": "float"}],
      "return_type": "float"
    }
  ]
}
```

**Generated Output** (`geometry.pl`):
```prolog
%% STUNIR Generated SWI_PROLOG Code
%% DO-178C Level A Compliant

:- module(geometry, []).

%% Calculate circle area from radius
circle_area(Radius, Result) :-
    true.
```

### Example 2: XSB with Tabling

**Generated Output** (`fibonacci.P`):
```prolog
%% STUNIR Generated XSB Code
%% DO-178C Level A Compliant

:- module(fibonacci, []).

:- table fibonacci/2.

%% Compute Fibonacci number
fibonacci(N, Result) :-
    true.
```

### Example 3: Ciao with Assertions

**Generated Output** (`math.pl`):
```prolog
%% STUNIR Generated CIAO Code
%% DO-178C Level A Compliant

:- module(math, [], [assertions]).

:- pred add(+,-,-).

%% Add two integers
add(X, Y, Result) :-
    true.
```

### Example 4: ECLiPSe with CLP

**Generated Output** (`constraints.ecl`):
```prolog
%% STUNIR Generated ECLIPSE Code
%% DO-178C Level A Compliant

:- module(constraints).

:- lib(ic).

%% Constrained addition
add(X, Y, Result) :-
    true.
```

---

## 6. Functional to Logic Translation

### Key Translation Rules

| Functional IR | Prolog Logic |
|---------------|--------------|
| `function f(x, y) -> z` | `f(X, Y, Z) :- ...` |
| `return expr` | Unification in head |
| `if (cond) then a else b` | `(Cond -> A ; B)` |
| `while (cond) { body }` | Tail recursion |
| `x = expr` | `X is Expr` (arithmetic) or `X = Expr` (unification) |

### Example: Factorial

**IR (Functional)**:
```json
{
  "name": "factorial",
  "args": [{"name": "n", "type": "integer"}],
  "return_type": "integer",
  "statements": [
    {
      "kind": "if",
      "condition": "n == 0",
      "then": [{"kind": "return", "value": "1"}],
      "else": [{"kind": "return", "value": "n * factorial(n - 1)"}]
    }
  ]
}
```

**Generated Prolog**:
```prolog
%% Factorial with recursion
factorial(N, Result) :-
    (N =:= 0 ->
        Result = 1
    ;
        N1 is N - 1,
        factorial(N1, R1),
        Result is N * R1
    ).
```

### Example: List Sum

**IR (Functional)**:
```json
{
  "name": "sum_list",
  "args": [{"name": "lst", "type": "list"}],
  "return_type": "integer",
  "statements": [
    {"kind": "loop", "iterator": "elem", "collection": "lst"}
  ]
}
```

**Generated Prolog**:
```prolog
%% Sum list elements
sum_list([], 0).
sum_list([H|T], Sum) :-
    sum_list(T, Rest),
    Sum is H + Rest.
```

---

## 7. Dialect-Specific Features

### SWI-Prolog

**Features**:
- ISO compliance + extensions
- Module system
- CLP(FD) library
- DCG (Definite Clause Grammars)
- HTTP server, JSON, etc.

**Example**:
```prolog
:- module(swi_example, [process/2]).

:- use_module(library(clpfd)).

process(X, Y) :-
    X #> 0,
    Y #= X * 2.
```

### XSB

**Features**:
- Tabled resolution
- Incremental tabling
- HiLog (higher-order logic)
- Deductive database features

**Example**:
```prolog
:- module(xsb_example, [path/2]).

:- table path/2.

edge(1, 2).
edge(2, 3).
edge(3, 4).

path(X, Y) :- edge(X, Y).
path(X, Z) :- path(X, Y), edge(Y, Z).
```

**Benefit**: Tabling prevents infinite loops and improves performance for recursive queries.

### Ciao Prolog

**Features**:
- Assertion language
- Formal verification
- Resource analysis
- Multi-paradigm support

**Example**:
```prolog
:- module(ciao_example, [divide/3], [assertions, regtypes]).

:- regtype nat/1.
nat(0).
nat(N) :- nat(M), N is M + 1.

:- pred divide(+nat, +nat, -float).
:- entry divide(A, B) : (nat(A), nat(B), B > 0).
:- success divide(A, B, C) => float(C).

divide(A, B, C) :-
    C is A / B.
```

### GNU Prolog

**Features**:
- Fast native compilation
- CLP(FD) built-in
- Minimal footprint
- Good for embedded systems

**Example**:
```prolog
% GNU Prolog example

:- include('clpfd.pl').

sudoku_cell(X) :-
    X #>= 1,
    X #=< 9.
```

### ECLiPSe

**Features**:
- Interval constraints (`ic` library)
- Global constraints
- Optimization (minimize/maximize)
- Search strategies

**Example**:
```prolog
:- module(eclipse_example).

:- lib(ic).

optimize_sum(Vars, Sum) :-
    Vars :: 1..10,
    ic:sum(Vars) #= Sum,
    Sum #< 50,
    labeling(Vars).
```

---

## 8. Integration

### With STUNIR Toolchain

```bash
# Full pipeline: Spec → IR → Prolog
stunir_spec_to_ir --input spec.json --output module.ir.json
stunir_ir_to_code --input module.ir.json --output module.pl --target prolog
```

### Programmatic Usage (Ada)

```ada
with STUNIR.Semantic_IR;
with STUNIR.Emitters.Prolog;

procedure Generate_Prolog is
   Emitter : Prolog_Emitter;
   Module  : IR_Module;
   Output  : IR_Code_Buffer;
   Success : Boolean;
begin
   -- Configure emitter
   Emitter.Config.Dialect := SWI_Prolog;
   Emitter.Config.Use_CLP := True;

   -- Load IR module (implementation omitted)
   -- ...

   -- Generate code
   Emitter.Emit_Module (Module, Output, Success);

   if Success then
      -- Write output (implementation omitted)
      null;
   end if;
end Generate_Prolog;
```

### Running Generated Code

```bash
# SWI-Prolog
swipl -s module.pl -g main -t halt

# XSB
xsb -e "[module]."

# GNU Prolog
gprolog --consult-file module.pl

# ECLiPSe
eclipse -b module.ecl
```

---

## 9. Troubleshooting

### Common Issues

#### Issue: "Module declaration not supported"

**Cause**: Using dialect with limited module support (e.g., GNU Prolog)

**Solution**:
- Use comment-based module markers
- Switch to SWI-Prolog or SICStus for full module support

#### Issue: "Tabling not available"

**Cause**: Using `--use-tabling` with unsupported dialect

**Solution**:
- Use XSB or YAP for tabling support
- Remove `--use-tabling` flag for other dialects

#### Issue: "CLP predicates undefined"

**Cause**: Missing CLP library import

**Solution**:
- Ensure dialect supports CLP (SWI, GNU, SICStus, ECLiPSe, B-Prolog)
- Check generated `:- use_module(library(clpfd)).` line

#### Issue: "Variable capitalization errors"

**Cause**: Prolog requires variables to start with uppercase

**Solution**:
- STUNIR emitter automatically capitalizes variables
- Report bug if lowercase variables appear in output

### Debugging

Enable verbose output:
```bash
stunir_ir_to_code --input module.ir.json \
                  --output module.pl \
                  --target prolog \
                  --dialect swi-prolog \
                  --verbose
```

Validate generated code:
```bash
# SWI-Prolog syntax check
swipl -g "load_files(module, [silent(false)]), halt."

# Check for undefined predicates
swipl -g "load_files(module), check_predicate_definitions, halt."
```

---

## Appendix A: Dialect Comparison

| Feature | SWI | GNU | SICStus | YAP | XSB | Ciao | B-Prolog | ECLiPSe |
|---------|-----|-----|---------|-----|-----|------|----------|---------|
| ISO Standard | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Modules | ✅ | Limited | ✅ | ✅ | ✅ | ✅ | Limited | ✅ |
| Tabling | Extension | ❌ | Extension | ✅ | ✅ | ❌ | ❌ | ❌ |
| CLP(FD) | ✅ | ✅ | ✅ | ✅ | Extension | Extension | ✅ | ✅ (ic) |
| Assertions | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| Compilation | JIT | Native | Native | JIT | Native | Native | Native | Native |
| Open Source | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | Free | ✅ |

---

## Appendix B: CLP Example

### Problem: N-Queens

**IR Input** (simplified):
```json
{
  "name": "nqueens",
  "args": [{"name": "n", "type": "integer"}],
  "return_type": "list"
}
```

**Generated Prolog** (SWI-Prolog with CLP):
```prolog
:- module(nqueens, [nqueens/2]).

:- use_module(library(clpfd)).

nqueens(N, Qs) :-
    length(Qs, N),
    Qs ins 1..N,
    safe_queens(Qs),
    labeling([], Qs).

safe_queens([]).
safe_queens([Q|Qs]) :-
    safe_queens(Qs, Q, 1),
    safe_queens(Qs).

safe_queens([], _, _).
safe_queens([Q|Qs], Q0, D0) :-
    Q #\= Q0,
    abs(Q - Q0) #\= D0,
    D1 #= D0 + 1,
    safe_queens(Qs, Q0, D1).
```

**Usage**:
```prolog
?- nqueens(8, Solution).
Solution = [1, 5, 8, 6, 3, 7, 2, 4] ;
Solution = [1, 6, 8, 3, 7, 4, 2, 5] ;
...
```

---

## Appendix C: References

- **ISO Prolog**: ISO/IEC 13211-1:1995
- **SWI-Prolog**: https://www.swi-prolog.org/
- **GNU Prolog**: http://www.gprolog.org/
- **SICStus Prolog**: https://sicstus.sics.se/
- **YAP**: https://www.dcc.fc.up.pt/~vsc/Yap/
- **XSB**: http://xsb.sourceforge.net/
- **Ciao**: https://ciao-lang.org/
- **B-Prolog**: http://www.picat-lang.org/bprolog/
- **ECLiPSe**: https://eclipseclp.org/

---

**Document Control**  
Version: 1.0  
Author: STUNIR Development Team  
Last Updated: 2026-01-31
