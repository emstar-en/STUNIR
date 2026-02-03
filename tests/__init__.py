"""STUNIR Test Suite.

Comprehensive test suite for the STUNIR project covering:
- Unit tests: Individual component testing
- Integration tests: Cross-component functionality
- Confluence tests: Multi-implementation consistency
- Performance tests: Benchmarking and load testing
- Security tests: Vulnerability scanning
- Fuzz tests: Input validation and edge cases

Test Organization:
    tests/
    ├── unit/           # Unit tests for individual modules
    ├── integration/    # Integration tests for pipelines
    ├── confluence/     # Cross-implementation tests
    ├── codegen/        # Code generation tests
    ├── semantic_ir/    # Semantic IR parser tests
    └── targets/        # Target language tests

Running Tests:
    # Run all tests
    pytest
    
    # Run specific test category
    pytest tests/unit/ -v
    pytest tests/integration/ -v
    pytest tests/confluence/ -v
    
    # Run with coverage
    pytest --cov=stunir --cov-report=html

Test Fixtures:
    Test data is located in test_vectors/ and includes:
    - Sample specifications
    - Expected IR outputs
    - Expected code outputs
    - Edge case inputs
"""
