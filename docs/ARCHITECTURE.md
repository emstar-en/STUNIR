# STUNIR Architecture

Technical architecture documentation for the STUNIR (Structured Toolchain for Unified Native IR) system.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Component Architecture](#component-architecture)
3. [Data Flow](#data-flow)
4. [Design Decisions](#design-decisions)
5. [Technology Stack](#technology-stack)
6. [Module Organization](#module-organization)
7. [Security Architecture](#security-architecture)

---

## System Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           STUNIR System                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                │
│  │   Input      │   │   Core       │   │   Output     │                │
│  │   Layer      │   │   Pipeline   │   │   Layer      │                │
│  ├──────────────┤   ├──────────────┤   ├──────────────┤                │
│  │ • Specs      │──▶│ • Parser     │──▶│ • Targets    │                │
│  │ • Configs    │   │ • IR Emitter │   │ • Manifests  │                │
│  │ • Validation │   │ • Transform  │   │ • Receipts   │                │
│  └──────────────┘   └──────────────┘   └──────────────┘                │
│          │                  │                  │                        │
│          └──────────────────┼──────────────────┘                        │
│                             │                                            │
│                    ┌────────▼────────┐                                  │
│                    │  Verification   │                                  │
│                    │     Layer       │                                  │
│                    ├─────────────────┤                                  │
│                    │ • Hash Verify   │                                  │
│                    │ • Manifest Check│                                  │
│                    │ • Receipt Valid │                                  │
│                    └─────────────────┘                                  │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                        Native Tools Layer                                │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                │
│  │    Rust      │   │   Haskell    │   │      C       │                │
│  │  (crypto,    │   │  (manifest,  │   │ (provenance) │                │
│  │   canon)     │   │   IR gen)    │   │              │                │
│  └──────────────┘   └──────────────┘   └──────────────┘                │
└─────────────────────────────────────────────────────────────────────────┘
```

### Core Principles

1. **Determinism**: All operations produce identical outputs for identical inputs
2. **Verifiability**: Every artifact can be cryptographically verified
3. **Polyglot**: Support for multiple programming languages
4. **Security**: Input validation and protection against common attacks
5. **Modularity**: Components can be used independently or together

---

## Component Architecture

### Input Layer

```
┌─────────────────────────────────────────┐
│              Input Layer                 │
├─────────────────────────────────────────┤
│                                          │
│  ┌─────────────┐    ┌─────────────┐     │
│  │    Spec     │    │   Config    │     │
│  │   Parser    │    │   Loader    │     │
│  └──────┬──────┘    └──────┬──────┘     │
│         │                  │            │
│         ▼                  ▼            │
│  ┌─────────────────────────────────┐    │
│  │        Input Validator          │    │
│  │  • Path sanitization            │    │
│  │  • JSON validation              │    │
│  │  • Schema checking              │    │
│  │  • Size limits                  │    │
│  └─────────────────────────────────┘    │
│                                          │
└─────────────────────────────────────────┘
```

**Components:**

| Component | Responsibility | Implementation |
|-----------|----------------|----------------|
| Spec Parser | Parse spec JSON files | Python + Rust |
| Config Loader | Load configuration | Python |
| Input Validator | Validate and sanitize inputs | Python (`tools/security/validation.py`) |

### Core Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Core Pipeline                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│    ┌─────────┐      ┌─────────┐      ┌─────────┐      ┌─────────┐  │
│    │  Spec   │ ──▶  │   IR    │ ──▶  │ Target  │ ──▶  │Manifest │  │
│    │ Parse   │      │  Emit   │      │  Emit   │      │  Gen    │  │
│    └─────────┘      └─────────┘      └─────────┘      └─────────┘  │
│         │                │                │                │        │
│         ▼                ▼                ▼                ▼        │
│    ┌─────────────────────────────────────────────────────────────┐  │
│    │                     Canonicalization                        │  │
│    │  • JSON normalization (RFC 8785)                           │  │
│    │  • Deterministic key ordering                              │  │
│    │  • Consistent serialization                                │  │
│    └─────────────────────────────────────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Pipeline Stages:**

1. **Spec Parse**: Convert input spec to internal representation
2. **IR Emit**: Generate normalized Intermediate Representation
3. **Target Emit**: Generate target-specific code
4. **Manifest Gen**: Create artifact manifests

### Cryptographic Layer

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Cryptographic Layer                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌───────────────────┐    ┌───────────────────┐                     │
│  │   File Hashing    │    │ Directory Hashing │                     │
│  │  • SHA-256        │    │  • Merkle Tree    │                     │
│  │  • Streaming      │    │  • Sorted paths   │                     │
│  │  • Size limits    │    │  • Depth limits   │                     │
│  └───────────────────┘    └───────────────────┘                     │
│            │                        │                                │
│            └────────────┬───────────┘                                │
│                         ▼                                            │
│            ┌───────────────────────┐                                │
│            │   Hash Verification   │                                │
│            │  • Manifest compare   │                                │
│            │  • Receipt validate   │                                │
│            │  • Integrity check    │                                │
│            └───────────────────────┘                                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Implementation:** Primarily in Rust (`stunir-native`) for performance and memory safety.

---

## Data Flow

### Spec to Output Flow

```
                    Input                Processing               Output
                      │                      │                      │
                      ▼                      ▼                      ▼
              ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
              │  spec.json   │      │    IR.json   │      │  target/     │
              │              │      │              │      │  ├── main.rs │
              │ {            │      │ {            │      │  ├── lib.rs  │
              │  "kind":     │──────│  "kind": "ir"│──────│  └── ...     │
              │    "spec",   │      │  "functions" │      │              │
              │  "modules":  │      │    ...       │      │              │
              │    [...]     │      │ }            │      │              │
              │ }            │      │              │      │              │
              └──────────────┘      └──────────────┘      └──────────────┘
                      │                      │                      │
                      ▼                      ▼                      ▼
              ┌──────────────────────────────────────────────────────────┐
              │                     Receipts Directory                    │
              │  receipts/                                                │
              │  ├── ir_manifest.json      (hashes of IR artifacts)      │
              │  ├── targets_manifest.json (hashes of generated code)    │
              │  └── build_receipt.json    (complete build record)       │
              └──────────────────────────────────────────────────────────┘
```

### Verification Flow

```
┌────────────────────────────────────────────────────────────────────────┐
│                         Verification Process                            │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────────┐                                                       │
│   │   Manifest  │                                                       │
│   │    JSON     │                                                       │
│   └──────┬──────┘                                                       │
│          │                                                              │
│          ▼                                                              │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │  For each entry in manifest:                                     │  │
│   │                                                                   │  │
│   │    ┌─────────┐      ┌─────────┐      ┌─────────┐                │  │
│   │    │  Load   │ ──▶  │ Compute │ ──▶  │ Compare │                │  │
│   │    │  File   │      │  Hash   │      │  Hashes │                │  │
│   │    └─────────┘      └─────────┘      └─────────┘                │  │
│   │                                             │                    │  │
│   │                                     ┌───────┴───────┐           │  │
│   │                                     ▼               ▼           │  │
│   │                               ┌─────────┐     ┌─────────┐      │  │
│   │                               │  Match  │     │Mismatch │      │  │
│   │                               │   ✓     │     │   ✗     │      │  │
│   │                               └─────────┘     └─────────┘      │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│   Result: All match → Verified ✓  |  Any mismatch → Failed ✗          │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Design Decisions

### Why Multiple Languages?

STUNIR uses multiple implementation languages for specific purposes:

| Language | Use Case | Rationale |
|----------|----------|-----------|
| **Python** | Orchestration, tooling | Rapid development, ecosystem |
| **Rust** | Crypto, performance-critical | Memory safety, speed |
| **Haskell** | IR generation, type safety | Strong types, purity |
| **C** | Provenance, low-level | Universal compatibility |

### Why Canonical JSON?

**Problem**: Standard JSON serialization is non-deterministic (key ordering varies).

**Solution**: RFC 8785 JSON Canonicalization Scheme (JCS):
- Lexicographically sorted keys
- No insignificant whitespace
- Consistent number formatting
- Minimal string escaping

```python
# Non-deterministic (standard)
{"z": 1, "a": 2}  # or {"a": 2, "z": 1} - depends on implementation

# Canonical (STUNIR)
{"a":2,"z":1}     # Always this exact output
```

### Why Merkle Trees for Directories?

**Benefits:**
1. **Partial Verification**: Verify subset without full directory scan
2. **Change Detection**: Quickly identify which files changed
3. **Parallelization**: Hash individual files concurrently
4. **Determinism**: Sorted paths ensure consistent ordering

**Structure:**
```
Directory Root Hash
         │
    ┌────┴────┐
    │         │
  file1     dir1
  (hash)      │
         ┌───┴───┐
       file2   file3
       (hash)  (hash)

Root = SHA256(SHA256(file1) || SHA256(dir1_root))
```

### Why Separate Receipts from Manifests?

| Aspect | Manifest | Receipt |
|--------|----------|---------|
| **Purpose** | Index of artifacts | Proof of build |
| **Contents** | Paths + hashes | Inputs, outputs, tools, time |
| **Mutability** | Regenerated on changes | Append-only history |
| **Scope** | Single artifact type | Entire build |

---

## Technology Stack

### Core Technologies

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Technology Stack                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Layer           Technology              Purpose                     │
│  ─────────────────────────────────────────────────────────────────  │
│  CLI             Python (argparse)       User interface              │
│  Orchestration   Python                  Workflow coordination       │
│  IR Processing   Python + Haskell        Spec → IR transformation   │
│  Cryptography    Rust (sha2, hex)        Hashing, verification      │
│  Serialization   serde (Rust), json (Py) Data encoding/decoding    │
│  Validation      Python (custom)         Input sanitization         │
│  Build           Cargo, Cabal, pip       Package management         │
│  Testing         pytest, cargo test      Automated testing          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Dependencies

**Python:**
- `hashlib` - SHA-256 hashing
- `json` - JSON parsing
- `pathlib` - Path handling
- `dataclasses` - Data structures

**Rust:**
- `sha2` - SHA-256 implementation
- `serde` / `serde_json` - Serialization
- `anyhow` - Error handling
- `thiserror` - Error types
- `clap` - CLI parsing

**Haskell:**
- `aeson` - JSON parsing
- `cryptonite` - Cryptographic primitives
- `bytestring` - Binary data

---

## Module Organization

### Repository Structure

```
stunir/
├── docs/                      # Documentation
│   ├── API_REFERENCE.md
│   ├── ARCHITECTURE.md        # This file
│   ├── USER_GUIDE.md
│   └── DEPLOYMENT.md
│
├── tools/                     # Python tooling
│   ├── ir_emitter/           # Spec → IR conversion
│   ├── emitters/             # Target code generators
│   ├── security/             # Input validation
│   │   ├── validation.py     # Sanitization
│   │   └── __init__.py
│   ├── errors.py             # Error system
│   └── native/               # Native tool sources
│       ├── rust/             # Rust implementation
│       │   └── stunir-native/
│       │       ├── src/
│       │       │   ├── lib.rs
│       │       │   ├── crypto.rs
│       │       │   ├── canonical.rs
│       │       │   └── ...
│       │       └── Cargo.toml
│       └── haskell/          # Haskell implementation
│
├── manifests/                 # Manifest generators
│   ├── base.py               # Base classes
│   ├── ir/                   # IR manifests
│   ├── receipts/             # Receipt manifests
│   └── targets/              # Target manifests
│
├── targets/                   # Target emitters
│   ├── polyglot/             # High-level languages
│   │   ├── rust/
│   │   └── c_base.py
│   └── assembly/             # Low-level targets
│       ├── x86/
│       └── arm/
│
├── tests/                     # Test suites
│   ├── security/             # Security tests
│   │   └── test_fuzzing.py
│   └── ...
│
├── scripts/                   # Build scripts
│   ├── build.sh
│   └── verify_strict.sh
│
└── receipts/                  # Build outputs (generated)
```

### Module Dependencies

```
                    ┌─────────────────────┐
                    │     CLI Entry       │
                    │   (main.py/main.rs) │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
        ┌─────────┐      ┌─────────┐      ┌─────────┐
        │  tools/ │      │manifests│      │ targets │
        │ errors  │◀─────│         │      │         │
        └────┬────┘      └────┬────┘      └────┬────┘
             │                │                │
             │                │                │
             ▼                ▼                ▼
        ┌─────────────────────────────────────────┐
        │            tools/security               │
        │         (validation, sanitization)       │
        └─────────────────────────────────────────┘
                               │
                               ▼
        ┌─────────────────────────────────────────┐
        │         tools/native (Rust/Haskell)     │
        │      (crypto, canonical, low-level)      │
        └─────────────────────────────────────────┘
```

---

## Security Architecture

### Threat Model

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Threat Model                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Threat                    │ Mitigation                             │
│  ─────────────────────────────────────────────────────────────────  │
│  Path traversal            │ Path sanitization, base dir check      │
│  Symlink attacks           │ No symlink following by default        │
│  DoS via large files       │ File size limits (1GB max)            │
│  DoS via deep nesting      │ Directory depth limit (100)           │
│  JSON bomb                 │ JSON depth limit (50), size limit     │
│  Command injection         │ No shell=True, input validation       │
│  Hash collision            │ SHA-256 (256-bit security)            │
│  Manifest tampering        │ Verification against fresh hashes     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Security Boundaries

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Security Boundaries                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   UNTRUSTED                      │  TRUSTED                         │
│   ───────────────────────────────│──────────────────────────────    │
│                                  │                                   │
│   • User-provided specs          │  • Internal IR format            │
│   • User-provided paths          │  • Verified manifests            │
│   • Config files                 │  • Hash computations             │
│   • Environment variables        │  • Canonical JSON output         │
│                                  │                                   │
│   ─────────────────────────────▶ │                                   │
│         Input Validator          │                                   │
│   ◀───────────────────────────── │                                   │
│         (sanitization)           │                                   │
│                                  │                                   │
└─────────────────────────────────────────────────────────────────────┘
```

### Input Validation Pipeline

```
User Input
     │
     ▼
┌─────────────────┐
│ Path Sanitizer  │─── Reject: null bytes, .., special chars
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Size Checker   │─── Reject: files > 1GB, JSON depth > 50
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Format Validator│─── Reject: invalid JSON, schema mismatch
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ ID Validator    │─── Reject: invalid identifiers, reserved words
└────────┬────────┘
         │
         ▼
   Safe Input ✓
```

---

## See Also

- [API Reference](API_REFERENCE.md) - Complete API documentation
- [User Guide](USER_GUIDE.md) - Getting started tutorial
- [Contributing](../CONTRIBUTING.md) - Development guidelines
- [Deployment](DEPLOYMENT.md) - Production deployment guide
- [Security Policy](../SECURITY.md) - Security guidelines

---

*Architecture version: 1.0 | Last updated: January 2026*
