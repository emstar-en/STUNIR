# STUNIR Docker Integration

Docker containerization for the STUNIR deterministic IR pipeline.

## Overview

This directory contains Docker configurations for running the STUNIR toolchain
in isolated, reproducible environments.

## Dockerfiles

| File | Description | Profile |
|------|-------------|----------|
| `Dockerfile.haskell` | Haskell native toolchain | native |
| `Dockerfile.python` | Python-based toolchain | python |
| `Dockerfile.full` | Multi-stage full build | all |

## Quick Start

### Single Profile Build

```bash
# Build and run native profile
docker build -f docker/Dockerfile.haskell -t stunir:native .
docker run -v $(pwd)/specs:/stunir/specs:ro \
           -v $(pwd)/outputs:/stunir/outputs \
           stunir:native

# Build and run Python profile
docker build -f docker/Dockerfile.python -t stunir:python .
docker run -v $(pwd)/specs:/stunir/specs:ro \
           -v $(pwd)/outputs:/stunir/outputs \
           stunir:python
```

### Docker Compose

```bash
# Run full pipeline
cd docker
docker-compose up stunir-full

# Run verification
docker-compose up stunir-verify

# Run all services
docker-compose up
```

## Volume Mounts

| Mount | Purpose | Mode |
|-------|---------|------|
| `/stunir/specs` | Input specifications | read-only |
| `/stunir/outputs` | Generated outputs | read-write |
| `/stunir/receipts` | Build receipts | read-write |
| `/stunir/asm` | IR artifacts | read-write |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `STUNIR_PROFILE` | Build profile (native/python/shell) | native |
| `STUNIR_STRICT` | Enable strict verification | false |
| `STUNIR_VERBOSE` | Enable verbose output | false |

## CI/CD Integration

### GitHub Actions

```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build STUNIR
        run: |
          docker build -f docker/Dockerfile.full -t stunir:ci .
          docker run stunir:ci ./scripts/build.sh --profile native
```

### GitLab CI

```yaml
build:
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker build -f docker/Dockerfile.full -t stunir:ci .
    - docker run stunir:ci ./scripts/build.sh
```

## Building Images

```bash
# Build all images
docker build -f docker/Dockerfile.haskell -t stunir:native .
docker build -f docker/Dockerfile.python -t stunir:python .
docker build -f docker/Dockerfile.full -t stunir:full .

# Build with specific tag
docker build -f docker/Dockerfile.full -t stunir:v1.0.0 .
```

## Issue Reference

Resolves: `pipeline/docker/1019`
