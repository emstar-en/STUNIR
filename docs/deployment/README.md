# STUNIR Deployment Guide

> Issue: `docs/deployment/1069` - Complete docs → deployment pipeline stage

## Overview

This guide covers deploying STUNIR in development and production environments.

## Contents

| Document | Description |
|----------|-------------|
| [Production Deployment](production.md) | Production setup guide |

## Prerequisites

### Required
- Python 3.8+
- Git 2.20+

### Optional (by profile)
- **Native Profile**: GHC 9.0+, Cabal 3.0+
- **Rust Profile**: Rust 1.60+, Cargo
- **Shell Profile**: Bash 4.0+, jq, sha256sum

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/emstar-en/STUNIR.git
cd STUNIR
```

### 2. Install Dependencies
```bash
# Python dependencies
pip install -r docs/requirements.txt

# Optional: Build native tools
cd tools/native/haskell/stunir-native
cabal build
```

### 3. Run Build
```bash
./scripts/build.sh
```

### 4. Verify Installation
```bash
./scripts/verify.sh
```

## Build Profiles

| Profile | Command | Requirements |
|---------|---------|-------------|
| Native (Haskell) | `./scripts/build.sh --profile=native` | GHC, Cabal |
| Python | `./scripts/build.sh --profile=python` | Python 3.8+ |
| Shell | `./scripts/build.sh --profile=shell` | Bash, jq |
| Rust | `./scripts/build.sh --profile=rust` | Rust, Cargo |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|--------|
| `STUNIR_EPOCH` | Build epoch timestamp | Current time |
| `STUNIR_PROFILE` | Build profile | `python` |
| `STUNIR_OUTPUT` | Output directory | `output/` |
| `STUNIR_STRICT` | Enable strict mode | `false` |

## Directory Structure

```
STUNIR/
├── scripts/           # Build and verification scripts
├── tools/             # Pipeline tools
├── targets/           # Target emitters
├── manifests/         # Manifest generators
├── contracts/         # Build contracts
├── asm/ir/            # Generated IR artifacts
├── receipts/          # Verification receipts
└── docs/              # Documentation
```

## CI/CD Integration

### GitHub Actions
```yaml
name: STUNIR Build
on: [push, pull_request]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -r docs/requirements.txt
      - run: ./scripts/build.sh
      - run: ./scripts/verify.sh
```

### GitLab CI
```yaml
stunir-build:
  image: python:3.11
  script:
    - pip install -r docs/requirements.txt
    - ./scripts/build.sh
    - ./scripts/verify.sh
```

## Related
- [Production Deployment](production.md)
- [Troubleshooting](../troubleshooting/README.md)
- [Architecture](../architecture/README.md)

---
*STUNIR Deployment Guide v1.0*
