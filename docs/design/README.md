# STUNIR Design Documentation

> Issues: `docs/design/pipeline/1037`, `docs/design/receipts/1038`

## Overview

This section contains design documentation for key STUNIR subsystems.

## Design Documents

| Document | Description |
|----------|-------------|
| [Pipeline Design](pipeline.md) | Build pipeline architecture |
| [Receipts Design](receipts.md) | Verification receipts system |

## Design Principles

### 1. Determinism First
Every design decision prioritizes reproducibility:
- Canonical output formats
- Fixed epochs instead of timestamps
- Sorted collections

### 2. Modular Pipeline
The pipeline is composed of independent stages:
- Each stage has clear inputs/outputs
- Stages can be run independently
- Intermediate artifacts are cacheable

### 3. Verification by Default
Every artifact is verifiable:
- SHA-256 hashes for integrity
- Manifests for completeness
- Receipts for audit trail

### 4. Multi-Target Support
Designed for cross-platform generation:
- Abstract IR format
- Target-specific emitters
- Common base classes

## Architecture Decision Records

### ADR-001: JSON as Primary Format
**Decision:** Use JSON for all machine-readable artifacts.

**Rationale:**
- Human readable
- Wide tooling support
- Canonical form well-defined (RFC 8785)

### ADR-002: SHA-256 for Hashing
**Decision:** Use SHA-256 for all content hashes.

**Rationale:**
- Widely supported
- Sufficient collision resistance
- Good performance

### ADR-003: Haskell as Reference Implementation
**Decision:** Use Haskell for reference/native tools.

**Rationale:**
- Strong type system
- Immutable by default
- Deterministic evaluation

## Related
- [Architecture](../architecture/README.md)
- [Pipeline Design](pipeline.md)
- [Receipts Design](receipts.md)

---
*STUNIR Design Documentation v1.0*
