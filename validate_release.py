#!/usr/bin/env python3
"""
STUNIR v0.9.0 Release Validation Script

Validates that all components are ready for release.
"""

import json
import os
import sys
from pathlib import Path


def check_file_exists(filepath, description):
    """Check if a file exists."""
    if Path(filepath).exists():
        print(f"  [OK] {description}: {filepath}")
        return True
    else:
        print(f"  [MISSING] {description}: {filepath}")
        return False


def check_version_consistency():
    """Check version consistency across files."""
    print("\n1. Version Consistency Check")
    print("-" * 40)
    
    # Check CHANGELOG.md for version 0.8.9 or 0.9.0
    changelog = Path("STUNIR-main/CHANGELOG.md")
    if changelog.exists() and ("0.8.9" in changelog.read_text() or "0.9.0" in changelog.read_text()):
        print("  [OK] CHANGELOG.md contains version info")
        return True
    else:
        print("  [MISSING] Version info not found in CHANGELOG.md")
        return False


def check_documentation():
    """Check documentation completeness."""
    print("\n2. Documentation Check")
    print("-" * 40)
    
    docs = [
        ("STUNIR-main/CHANGELOG.md", "CHANGELOG"),
        ("STUNIR-main/docs/API_REFERENCE.md", "API Reference"),
        ("STUNIR-main/docs/TROUBLESHOOTING.md", "Troubleshooting"),
        ("STUNIR-main/README.md", "README"),
    ]
    
    results = []
    for filepath, desc in docs:
        results.append(check_file_exists(filepath, desc))
    
    return all(results)


def check_rust_backend():
    """Check Rust backend implementation."""
    print("\n3. Rust Backend Check")
    print("-" * 40)
    
    files = [
        ("STUNIR-main/src/main.rs", "Main entry point"),
        ("STUNIR-main/src/ir_v1.rs", "IR module"),
        ("STUNIR-main/src/emit.rs", "Emitter trait"),
        ("STUNIR-main/src/canonical.rs", "Canonical IR"),
    ]
    
    results = []
    for filepath, desc in files:
        results.append(check_file_exists(filepath, desc))
    
    return all(results)


def check_python_tools():
    """Check Python tools implementation."""
    print("\n4. Python Tools Check")
    print("-" * 40)
    
    files = [
        ("STUNIR-main/tools/ir_to_code.py", "IR to Code CLI"),
        ("STUNIR-main/tools/semantic_ir/emitters/types.py", "Semantic IR Types"),
        ("STUNIR-main/tools/semantic_ir/emitters/base_emitter.py", "Base Emitter"),
    ]
    
    results = []
    for filepath, desc in files:
        results.append(check_file_exists(filepath, desc))
    
    return all(results)


def check_tests():
    """Check test suites."""
    print("\n5. Test Suites Check")
    print("-" * 40)
    
    files = [
        ("STUNIR-main/tests/integration/test_integration.py", "Integration Tests"),
    ]
    
    results = []
    for filepath, desc in files:
        results.append(check_file_exists(filepath, desc))
    
    return all(results)


def run_integration_tests():
    """Run integration tests."""
    print("\n6. Integration Tests")
    print("-" * 40)
    
    test_file = Path("STUNIR-main/tests/integration/test_integration.py")
    if not test_file.exists():
        print("  [MISSING] Integration test file not found")
        return False
    
    import subprocess
    result = subprocess.run(
        [sys.executable, str(test_file)],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("  [OK] All integration tests passed")
        return True
    else:
        print("  [FAIL] Integration tests failed")
        return False


def main():
    """Run all validation checks."""
    print("=" * 60)
    print("STUNIR v0.9.0 Release Validation")
    print("=" * 60)
    
    checks = [
        ("Version Consistency", check_version_consistency),
        ("Documentation", check_documentation),
        ("Rust Backend", check_rust_backend),
        ("Python Tools", check_python_tools),
        ("Test Suites", check_tests),
        ("Integration Tests", run_integration_tests),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"  [ERROR] {name} check failed: {e}")
            results[name] = False
    
    # Print summary
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")
    
    print("-" * 60)
    print(f"Overall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n[SUCCESS] All validation checks passed!")
        print("STUNIR v0.9.0 is ready for release!")
        return 0
    else:
        print("\n[WARNING] Some validation checks failed!")
        print("Please review the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
