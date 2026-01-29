# STUNIR Lisp Target Emitters

## Overview

This package provides code emitters for Lisp-family languages, converting STUNIR IR to idiomatic code in:

- **Common Lisp** - ANSI CL with CLOS, ASDF, and type declarations
- **Scheme** - R7RS with define-library and tail-call optimization
- **Clojure** - JVM-based with type hints and clojure.spec
- **Racket** - With contracts and optional Typed Racket

## Architecture

```
targets/lisp/
├── __init__.py          # Package exports
├── base.py              # LispEmitterBase class
├── common_lisp/
│   ├── __init__.py
│   ├── emitter.py       # CommonLispEmitter
│   └── types.py         # CL type mapping
├── scheme/
│   ├── __init__.py
│   ├── emitter.py       # SchemeEmitter (R7RS)
│   └── types.py         # Scheme type mapping
├── clojure/
│   ├── __init__.py
│   ├── emitter.py       # ClojureEmitter
│   └── types.py         # Clojure type hints/specs
└── racket/
    ├── __init__.py
    ├── emitter.py       # RacketEmitter
    └── types.py         # Racket types/contracts
```

## Usage

### Common Lisp

```python
from targets.lisp.common_lisp import CommonLispEmitter, CommonLispConfig
from pathlib import Path

config = CommonLispConfig(
    target_dir=Path("./output"),
    use_declarations=True,
    emit_asdf=True
)
emitter = CommonLispEmitter(config)
result = emitter.emit(ir_data)
print(result.code)
```

Output:
```lisp
(defpackage #:stunir.math
  (:use #:cl)
  (:export #:add))

(in-package #:stunir.math)

(declaim (ftype (function (fixnum fixnum) fixnum) add))
(defun add (a b)
  (declare (type fixnum a) (type fixnum b))
  (the fixnum (+ a b)))
```

### Scheme (R7RS)

```python
from targets.lisp.scheme import SchemeEmitter, SchemeEmitterConfig

config = SchemeEmitterConfig(
    target_dir=Path("./output"),
    r7rs_library=True
)
emitter = SchemeEmitter(config)
result = emitter.emit(ir_data)
```

Output:
```scheme
(define-library (stunir math)
  (import (scheme base))
  (export add)
  
  (begin
    ;; add : integer integer -> integer
    (define (add a b)
      (+ a b))))
```

### Clojure

```python
from targets.lisp.clojure import ClojureEmitter, ClojureEmitterConfig

config = ClojureEmitterConfig(
    target_dir=Path("./output"),
    use_spec=True,
    use_type_hints=True
)
emitter = ClojureEmitter(config)
result = emitter.emit(ir_data)
```

Output:
```clojure
(ns stunir.math
  (:require [clojure.spec.alpha :as s]))

(s/fdef add
  :args (s/cat :arg0 int? :arg1 int?)
  :ret int?)

(defn ^long add
  [^long a ^long b]
  (+ a b))
```

### Racket

```python
from targets.lisp.racket import RacketEmitter, RacketEmitterConfig

config = RacketEmitterConfig(
    target_dir=Path("./output"),
    use_contracts=True
)
emitter = RacketEmitter(config)
result = emitter.emit(ir_data)
```

Output:
```racket
#lang racket

(provide
 (contract-out
  [add (-> integer? integer? integer?)]))

(define (add a b)
  (+ a b))
```

### Typed Racket

```python
config = RacketEmitterConfig(
    target_dir=Path("./output"),
    use_typed=True,
    lang="typed/racket"
)
emitter = RacketEmitter(config)
```

Output:
```racket
#lang typed/racket

(: add (-> Integer Integer Integer))
(define (add [a : Integer] [b : Integer])
  (+ a b))
```

## Type Mapping

| IR Type | Common Lisp | Scheme | Clojure | Racket |
|---------|-------------|--------|---------|--------|
| i32 | fixnum | integer | int?/^int | Integer |
| i64 | (signed-byte 64) | integer | int?/^long | Integer |
| f64 | double-float | real | double?/^double | Flonum |
| bool | boolean | boolean | boolean? | Boolean |
| string | string | string | string?/^String | String |
| void | (values) | void | nil? | Void |

## Symbolic IR Extensions

The emitters support symbolic IR features:

- **Symbols**: `{"kind": "symbol", "name": "hello"}`
- **Quotes**: `{"kind": "quote", "value": ...}`
- **Quasiquotes**: `{"kind": "quasiquote", "value": ...}`
- **Lambdas**: `{"kind": "lambda", "params": [...], "body": [...]}`
- **Macros**: `{"kind": "defmacro", "name": "...", ...}`

## Testing

```bash
cd /home/ubuntu/stunir_repo
python -m pytest tests/ir/test_symbolic_ir.py -v
python -m pytest tests/codegen/test_common_lisp_generator.py -v
python -m pytest tests/codegen/test_scheme_generator.py -v
python -m pytest tests/codegen/test_clojure_generator.py -v
python -m pytest tests/codegen/test_racket_generator.py -v
```

## Part of Phase 5A

This implementation follows the HLI (High-Level Implementation) framework:
- `HLI_SYMBOLIC_IR.md` - Symbolic IR extensions
- `HLI_COMMON_LISP_EMITTER.md` - Common Lisp emitter
- `HLI_SCHEME_EMITTER.md` - Scheme emitter
- `HLI_CLOJURE_EMITTER.md` - Clojure emitter
- `HLI_RACKET_EMITTER.md` - Racket emitter
