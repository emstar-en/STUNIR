# STUNIR Troubleshooting Guide

> Issue: `docs/troubleshooting/1070` - Complete docs → troubleshooting pipeline stage

## Overview

This guide helps diagnose and resolve common STUNIR issues.

## Contents

| Document | Description |
|----------|-------------|
| [Common Issues](common_issues.md) | Frequently encountered problems |

## Quick Diagnostic

### Check Version
```bash
python3 --version
git --version
./scripts/build.sh --version 2>/dev/null || echo "No version flag"
```

### Verify Installation
```bash
# Check Python modules
python3 -c "import tools; print('tools: OK')"
python3 -c "import manifests; print('manifests: OK')"

# Check scripts
test -x scripts/build.sh && echo "build.sh: OK"
test -x scripts/verify.sh && echo "verify.sh: OK"
```

### Test Basic Build
```bash
./scripts/build.sh --profile=python
```

## Common Error Categories

### 1. Build Failures
| Error | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError` | Missing Python package | `pip install -r docs/requirements.txt` |
| `Permission denied` | Script not executable | `chmod +x scripts/*.sh` |
| `command not found: stunir-native` | Native tools not built | Build Haskell tools or use Python profile |

### 2. Verification Failures
| Error | Cause | Solution |
|-------|-------|----------|
| `Hash mismatch` | Modified artifact | Rebuild from clean state |
| `Missing file` | Incomplete build | Run full build pipeline |
| `Manifest not found` | Manifest not generated | Run manifest generator |

### 3. Determinism Issues
| Error | Cause | Solution |
|-------|-------|----------|
| `Non-deterministic output` | Timestamp in output | Use epoch timestamps |
| `Different hashes on rebuild` | Unsorted keys | Use canonical JSON |

## Diagnostic Commands

### Check Manifests
```bash
# Verify IR manifest
python -m manifests.ir.verify_ir_manifest receipts/ir_manifest.json

# List manifest contents
jq '.entries[] | .name' receipts/ir_manifest.json
```

### Check Artifacts
```bash
# Compare hashes
sha256sum asm/ir/*.json
sha256sum receipts/*.json
```

### Verbose Build
```bash
STUNIR_DEBUG=1 ./scripts/build.sh 2>&1 | tee build.log
```

## Getting Help

1. Check [Common Issues](common_issues.md)
2. Review [Architecture](../architecture/README.md)
3. Search existing issues
4. Open new issue with diagnostic output

## Related
- [Common Issues](common_issues.md)
- [Deployment Guide](../deployment/README.md)

---
*STUNIR Troubleshooting v1.0*
