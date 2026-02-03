# STUNIR Manifests

This directory contains manifest generators and verifiers for all STUNIR pipeline stages.

## Overview

Manifests are deterministic JSON documents that track artifacts, configuration, and execution state throughout the STUNIR build pipeline.

## Manifest Types

| Type | Schema | Description |
|------|--------|-------------|
| IR | `stunir.manifest.ir.v1` | IR artifacts in `asm/ir/` |
| Receipts | `stunir.manifest.receipts.v1` | Build receipts |
| Contracts | `stunir.manifest.contracts.v1` | Validated contracts |
| Targets | `stunir.manifest.targets.v1` | Generated target code |
| Pipeline | `stunir.manifest.pipeline.v1` | Pipeline stage definitions |
| Runtime | `stunir.manifest.runtime.v1` | Runtime environment |
| Security | `stunir.manifest.security.v1` | Security attestations |
| Performance | `stunir.manifest.performance.v1` | Performance benchmarks |

## Usage

### Generate a manifest:
```bash
python3 manifests/ir/gen_ir_manifest.py --output=receipts/ir_manifest.json
```

### Verify a manifest:
```bash
python3 manifests/ir/verify_ir_manifest.py receipts/ir_manifest.json
```

## Architecture

- `base.py` - Shared utilities and base classes
- `__init__.py` - Package exports
- `<type>/gen_<type>_manifest.py` - Generator for each type
- `<type>/verify_<type>_manifest.py` - Verifier for each type

## Phase 4 Issues Resolved

- manifests/ir/1015
- manifests/receipts/1016
- manifests/contracts/1042
- manifests/targets/1043
- manifests/pipeline/1073
- manifests/runtime/1074
- manifests/security/1144
- manifests/performance/1145
