# STUNIR Repository Index

## Overview
This repository is organized to support the **STUNIR (Standardization Theorem + Unique Normals + Intermediate Reference)** architecture.
The current structure emphasizes a **Shell Primary** workflow, where orchestration is handled by POSIX-compliant shell scripts, and Python is treated as an optional accelerator.

## 🤖 AI Agent Resources (New)
If you are an AI model working on this repo, **START HERE**:
1.  **`STUNIR_ANTI_PATTERNS.md`**: Read this to avoid breaking the build with non-deterministic code.
2.  **`meta/ai_bins/`**: This is your "Swap Space". Use `A_TASK.md` to track your state.
3.  **`spec/STUNIR_TYPES.d.ts`**: Use this to understand the JSON schemas for Manifests and Receipts.

## Key Directories

### `scripts/` - The Build System
- **`build.sh`**: The entry point. Now uses a polyglot dispatch strategy.
- **`lib/`**: Contains core shell libraries (`dispatch.sh`, `receipt.sh`) for logic that was previously Python-only.
- **`generate_manifest.sh`**: Tool to generate deterministic file manifests.

### `spec/` - The Source of Truth
- **`env/`**: Environment allowlists (Strict vs. Discovery).
- **`schemas/`**: Markdown-based specifications for file formats (Manifest V1, KV Receipts).
- **`STUNIR_TYPES.d.ts`**: TypeScript definitions for core data structures.

### `tools/` - Deterministic Toolchain
- Contains the Python reference implementations (`epoch.py`, `spec_to_ir.py`).
- `epoch.py` has been patched to output strict canonical JSON.

### `asm/` - Artifacts
- Stores the Intermediate Reference (IR) and compiled outputs.

## Navigation for Models
- **`STUNIR_REPO_INDEX.json`**: A complete machine-readable list of files and descriptions.
- **`STUNIR_REFERENCE_RULES.md`**: The "Constitution" of this repo. Read this to understand the constraints (e.g., "No Python" fallback).
