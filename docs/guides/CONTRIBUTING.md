# Contributing to STUNIR

Thank you for your interest in contributing to STUNIR! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Quality Standards](#code-quality-standards)
- [Pull Request Process](#pull-request-process)
- [Commit Message Guidelines](#commit-message-guidelines)

## Code of Conduct

Please read and follow our Code of Conduct. We expect all contributors to be respectful and professional.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR-USERNAME/stunir.git`
3. Add the upstream remote: `git remote add upstream https://github.com/stunir/stunir.git`
4. Create a branch: `git checkout -b feature/your-feature-name`

## Development Setup

### Prerequisites

- Python 3.9+
- Rust 1.70+
- GHC 9.4+ (for Haskell)
- Node.js 18+ (for documentation)

### Python Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
pre-commit install --hook-type commit-msg
```

### Rust Setup

```bash
cd tools/native/rust/stunir-native
cargo build
cargo test
```

### Haskell Setup

```bash
cd tools/native/haskell/stunir-native
cabal update
cabal build
```

## Code Quality Standards

### Pre-commit Hooks

We use pre-commit hooks to ensure code quality. Install them:

```bash
pre-commit install
```

Hooks will automatically run on commit. To run manually:

```bash
pre-commit run --all-files
```

### Python

- **Formatting**: Black (line length 100)
- **Linting**: Ruff with comprehensive rules
- **Type Checking**: MyPy with strict mode
- **Import Sorting**: isort (black profile)

Run checks manually:

```bash
black --check tools/ manifests/ targets/
ruff check tools/ manifests/ targets/
mypy tools/ manifests/
```

### Rust

- **Formatting**: rustfmt
- **Linting**: Clippy with pedantic lints
- **No unsafe code** without justification

Run checks:

```bash
cd tools/native/rust/stunir-native
cargo fmt --check
cargo clippy -- -D warnings
cargo test
```

### Haskell

- **Linting**: hlint
- **Strict warnings**: `-Wall -Werror`
- **Type signatures** on all functions

Run checks:

```bash
cd tools/native/haskell/stunir-native
cabal build --ghc-options="-Wall -Werror"
hlint src/
```

### Documentation

All code must be documented:

- **Python**: Google-style docstrings
- **Rust**: rustdoc comments (`///`)
- **Haskell**: Haddock comments (`-- |`)

See [Documentation Standards](docs/development/DOCUMENTATION_STANDARDS.md).

## Pull Request Process

1. **Update your branch** with the latest upstream changes:
   ```bash
   git fetch upstream
   git rebase upstream/devsite
   ```

2. **Run all checks**:
   ```bash
   pre-commit run --all-files
   pytest tests/
   ```

3. **Push your branch**:
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Create a Pull Request** targeting the `devsite` branch

5. **PR Requirements**:
   - [ ] All CI checks pass
   - [ ] Code follows style guidelines
   - [ ] Tests added/updated as needed
   - [ ] Documentation updated
   - [ ] Changelog entry added (if applicable)
   - [ ] Commit messages follow guidelines

## Commit Message Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding/updating tests
- `ci`: CI/CD changes
- `chore`: Maintenance tasks
- `deps`: Dependency updates

### Scopes

- `python`: Python code changes
- `rust`: Rust code changes
- `haskell`: Haskell code changes
- `manifests`: Manifest system
- `targets`: Target emitters
- `ci`: CI/CD
- `docs`: Documentation

### Examples

```
feat(manifests): add pipeline manifest generator

Implement gen_pipeline_manifest.py for tracking STUNIR pipeline stages.
Adds schema stunir.manifest.pipeline.v1 with stage ordering.

Closes #1073
```

```
fix(rust): resolve clippy warnings in canonical.rs

- Use `if let` instead of match for single pattern
- Remove unnecessary clone
- Add missing documentation
```

## Testing

### Running Tests

```bash
# Python tests with coverage
pytest tests/ --cov=tools --cov=manifests --cov-report=html

# Rust tests
cd tools/native/rust/stunir-native && cargo test

# Haskell tests
cd tools/native/haskell/stunir-native && cabal test
```

### Test Categories

- `unit/`: Isolated unit tests
- `integration/`: Integration tests
- `determinism/`: Determinism verification

Mark tests with appropriate markers:

```python
import pytest

@pytest.mark.unit
def test_sha256():
    pass

@pytest.mark.integration
@pytest.mark.slow
def test_full_pipeline():
    pass
```

## Questions?

Feel free to open an issue for questions or discussions!
