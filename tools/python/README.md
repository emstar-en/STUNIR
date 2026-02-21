# STUNIR Python Tools

> **Status:** Secondary to SPARK. Use `tools/spark/` for production.
> **Authority:** `tools/spark/ARCHITECTURE.md` is the canonical SSoT.

---

## Directory Structure

```
tools/python/
├── targets/            # Code emitters for various target languages
├── bootstrap/          # STUNIR bootstrap compiler, lexer, grammar
├── spec/               # Spec utilities (validation, templates)
├── ir_modules/         # IR modules (actor, asp, business, etc.)
├── manifests/          # IR manifest generation and verification
├── ir/                 # IR utilities
├── semantic/           # Semantic analysis
├── semantic_ir/        # Semantic IR implementation
├── emitters/           # Emitter utilities
├── codegen/            # Code generation
├── validators/         # Validation utilities
├── validation/         # Validation logic
├── parsers/            # Parsing utilities
├── scripts/            # Pipeline scripts (including unified_analysis)
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
