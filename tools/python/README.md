# STUNIR Python Tools

> **Status:** Secondary to SPARK. Use `tools/spark/` for production.
> **Authority:** `tools/spark/ARCHITECTURE.md` is the canonical SSoT.

---

## Directory Structure

```
tools/python/
├── targets/            # Code emitters for various target languages
│   ├── asm_ir/         # Assembly IR emitters
│   ├── asp/            # Answer Set Programming emitters
│   ├── assembly/       # x86, ARM assembly emitters
│   ├── beam/           # BEAM VM (Erlang, Elixir) emitters
│   ├── business/       # COBOL, BASIC emitters
│   ├── bytecode/       # Bytecode emitters
│   ├── c/              # C/C++ emitters
│   ├── constraints/    # MiniZinc, CHR emitters
│   ├── embedded/       # Embedded systems emitters
│   ├── expert_systems/ # CLIPS, JESS emitters
│   ├── fpga/           # FPGA emitters
│   ├── functional/     # Haskell, F#, OCaml emitters
│   ├── gpu/            # CUDA, OpenCL emitters
│   ├── grammar/        # ANTLR, BNF emitters
│   ├── json/           # JSON emitters
│   ├── lexer/          # Lexer generators
│   ├── lisp/           # Lisp family emitters
│   ├── mobile/         # Mobile platform emitters
│   ├── native/         # Native code emitters
│   ├── oop/            # OOP language emitters
│   ├── parser/         # Parser generators
│   ├── planning/       # PDDL emitters
│   ├── polyglot/       # Multi-language emitters
│   ├── prolog/         # Prolog emitters
│   ├── scientific/     # Fortran, Pascal emitters
│   ├── systems/        # Systems language emitters
│   └── wasm/           # WebAssembly emitters
├── manifests/          # IR manifest generation and verification
├── ir/                 # IR utilities
├── semantic/           # Semantic analysis
├── semantic_ir/        # Semantic IR implementation
├── emitters/           # Emitter utilities
├── codegen/            # Code generation
├── validators/         # Validation utilities
├── validation/         # Validation logic
├── parsers/            # Parsing utilities
├── scripts/            # Pipeline scripts
├── integration/        # Integration tests
├── integrations/       # Integration utilities
├── optimize/           # Optimization utilities
├── security/           # Security utilities
├── resilience/         # Resilience patterns
├── retry/              # Retry logic
├── ratelimit/          # Rate limiting
├── telemetry/          # Telemetry utilities
├── resources/          # Resource management
├── memory/             # Memory utilities
├── common/             # Common utilities
├── config/             # Configuration
├── platform/           # Platform utilities
├── lib/                # Library utilities
├── serializers/        # Serialization
├── stunir_logging/     # Logging
├── stunir_types/       # Type definitions
├── receipt_emitter/    # Receipt emission
├── manifest/           # Manifest utilities
├── ir_emitter/         # IR emission
├── canonicalizers/     # Canonicalization
├── conformance/        # Conformance testing
├── *.py                # Top-level Python utilities (49 files)
└── stunir_minimal.py   # Minimal pipeline runner
```

> **Note:** SPARK/Ada code previously in `targets/spark/` and `core/` has been archived to `docs/archive/spark_deprecated/`. Use `tools/spark/` for SPARK implementation.

---

## Usage

These Python tools are **reference implementations** and **secondary** to the SPARK pipeline.

### When to Use Python Tools

- Learning and understanding pipeline logic
- Rapid prototyping (when receipts not required)
- When GNAT/SPARK toolchain is unavailable

### When to Use SPARK Tools

- All production use cases
- Safety-critical applications
- Systems requiring formal verification
- Reproducible builds with audit receipts

---

## Policy Reference

See `docs/archive/ARCHIVE_POLICY.md` for:
- Shell offloading deprecation rationale
- Python patch fallback policy with receipt requirements
- SPARK-first policy details
