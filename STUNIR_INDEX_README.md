# STUNIR Repository Index

## Overview
This repository is organized to support the **STUNIR (Standardization Theorem + Unique Normals + Intermediate Reference)** architecture.
The current structure emphasizes a **Shell Primary** workflow, where orchestration is handled by POSIX-compliant shell scripts, and Python is treated as an optional accelerator.

## Key Directories

### `scripts/` - The Build System
- **`build.sh`**: The entry point. Now uses a polyglot dispatch strategy.
- **`lib/`**: **[NEW]** Contains core shell libraries (`dispatch.sh`, `receipt.sh`) for logic that was previously Python-only.
- **`generate_manifest.sh`**: **[NEW]** Tool to generate deterministic file manifests.

### `spec/` - The Source of Truth
- **`env/`**: **[NEW]** Environment allowlists (Strict vs. Discovery).
- **`schemas/`**: **[NEW]** Markdown-based specifications for file formats (Manifest V1, KV Receipts).

### `tools/` - Deterministic Toolchain
- Contains the Python reference implementations (`epoch.py`, `spec_to_ir.py`).
- `epoch.py` has been patched to output strict canonical JSON.

### `asm/` - Artifacts
- Stores the Intermediate Reference (IR) and compiled outputs.

## Navigation for Models
- **`STUNIR_REPO_INDEX.json`**: A complete machine-readable list of files and descriptions.
- **`STUNIR_REFERENCE_RULES.md`**: The "Constitution" of this repo. Read this to understand the constraints (e.g., "No Python" fallback).

## Recent Changes (Shell Primary)
- Added `scripts/lib/` for shell-native logic.
- Added `spec/schemas/` for explicit format definitions.
- Split environment allowlists into `discovery` (permissive) and `runtime` (strict).
