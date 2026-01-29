# STUNIR Interactive Examples

This directory contains interactive examples demonstrating STUNIR's capabilities across multiple programming languages.

## Directory Structure

```
examples/
├── python/              # Python examples
│   ├── basic_usage.py   # Getting started with STUNIR
│   ├── advanced_usage.py # Advanced features
│   ├── custom_emitters.py # Creating custom emitters
│   ├── batch_processing.py # Batch operations
│   └── error_handling.py # Error handling patterns
├── rust/                # Rust examples
│   ├── basic_usage.rs   # Basic Rust usage
│   └── advanced_usage.rs # Advanced Rust features
├── haskell/             # Haskell examples
│   ├── BasicUsage.hs    # Basic Haskell usage
│   └── AdvancedUsage.hs # Advanced Haskell features
└── notebooks/           # Jupyter notebooks
    ├── getting_started.ipynb # Interactive tutorial
    └── advanced_features.ipynb # Advanced workflows
```

## Quick Start

### Python Examples

```bash
# Run basic usage example
python examples/python/basic_usage.py

# Run with custom input
python examples/python/basic_usage.py --spec path/to/spec.json
```

### Rust Examples

```bash
# Compile and run
rustc examples/rust/basic_usage.rs -o basic_usage
./basic_usage
```

### Haskell Examples

```bash
# Run with GHCi
ghci examples/haskell/BasicUsage.hs
# Or compile
ghc examples/haskell/BasicUsage.hs -o basic_usage
```

### Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook examples/notebooks/
```

## Prerequisites

- **Python**: Python 3.8+ with `stunir` package installed
- **Rust**: Rust 1.70+ with Cargo
- **Haskell**: GHC 9.0+ with Cabal
- **Notebooks**: Jupyter with Python kernel

## Running All Examples

```bash
# Run all Python examples
for f in examples/python/*.py; do python "$f"; done

# Validate examples work
python -m pytest examples/python/ -v
```

## Contributing

When adding new examples:

1. Include clear comments explaining each step
2. Handle errors gracefully
3. Provide sample input/output
4. Update this README
5. Add to CI validation

## License

All examples are provided under the same license as STUNIR.
