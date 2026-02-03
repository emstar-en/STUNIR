"""STUNIR Code Generation Tests.

This package contains tests for the STUNIR code generation pipeline:
- IR to code conversion tests
- Target language emitter tests
- Code optimization tests
- Output validation tests

Test Categories:
    - Unit tests: Individual component testing
    - Integration tests: End-to-end pipeline testing
    - Confluence tests: Cross-implementation consistency

Usage:
    pytest tests/codegen/ -v
    python -m unittest discover tests/codegen/

Coverage Areas:
    - Python ir_to_code tool
    - Rust ir_to_code implementation
    - SPARK ir_to_code implementation
    - Target-specific code generation (C, Rust, etc.)
"""

# Tests for STUNIR code generation modules
