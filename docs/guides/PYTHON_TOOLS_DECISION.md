# Python Tools Alignment Plan

## Version 0.8.9

**Date:** February 3, 2026  
**Decision:** ALIGN Python tools with Ada SPARK and Rust implementations

---

## Executive Summary

STUNIR will maintain **three parallel implementations**: Python (reference), Ada SPARK (verified), and Rust (performance). Rather than deprecating Python, we will bring all implementations to feature parity, ensuring users can choose based on their needs:

- **Python**: Rapid prototyping, readability, ecosystem integration
- **Ada SPARK**: Safety-critical, formal verification, DO-178C compliance
- **Rust**: High performance, systems programming, modern toolchain

---

## Current State Analysis

### Implementation Comparison

| Feature | Python | Ada SPARK | Rust | Gap |
|---------|--------|-----------|------|-----|
| spec_to_ir | ✅ Complete | ✅ Complete | ✅ Complete | None |
| ir_to_code (C) | ✅ Complete | ✅ Complete | ✅ Complete | None |
| ir_to_code (Rust) | ❌ Missing | ✅ Complete | ✅ Complete | Add to Python |
| ir_to_code (Python) | ❌ Missing | ✅ Complete | ✅ Complete | Add to Python |
| ir_to_code (JS) | ❌ Missing | ✅ Complete | ✅ Complete | Add to Python |
| ir_to_code (Zig) | ❌ Missing | ✅ Complete | ✅ Complete | Add to Python |
| ir_to_code (Go) | ❌ Missing | ✅ Complete | ✅ Complete | Add to Python |
| ir_to_code (Ada) | ❌ Missing | ✅ Complete | ✅ Complete | Add to Python |
| ir_optimize | ❌ Missing | ✅ Complete | ❌ Missing | Add to Python & Rust |
| Test Suite | ❌ Missing | ✅ Complete | ✅ Complete | Add Python tests |

---

## Alignment Strategy

### Phase 1: Python Feature Completion (v0.9.0)

#### 1.1 Add Missing Target Emitters to Python

**ir_to_code.py** needs emitters for:
- Rust
- Python
- JavaScript
- Zig
- Go
- Ada

**Implementation Plan:**
```python
# Add to ir_to_code.py

def emit_rust(ir: Dict[str, Any]) -> str:
    """Generate Rust code from IR."""
    # Implementation aligned with Ada SPARK output
    pass

def emit_python(ir: Dict[str, Any]) -> str:
    """Generate Python code from IR."""
    # Implementation aligned with Ada SPARK output
    pass

# ... similar for JS, Zig, Go, Ada
```

#### 1.2 Implement ir_optimizer in Python

Create `tools/ir_optimizer.py`:
```python
"""IR Optimizer - Python Implementation

Aligns with Ada SPARK optimizer in tools/spark/src/optimizer/
"""

def constant_folding(ir: Dict[str, Any]) -> Dict[str, Any]:
    """Fold constant expressions."""
    pass

def constant_propagation(ir: Dict[str, Any]) -> Dict[str, Any]:
    """Propagate constants through code."""
    pass

def dead_code_elimination(ir: Dict[str, Any]) -> Dict[str, Any]:
    """Remove dead code."""
    pass

def unreachable_code_elimination(ir: Dict[str, Any]) -> Dict[str, Any]:
    """Remove unreachable code."""
    pass
```

#### 1.3 Create Python Test Suite

Create `tests/python/test_spec_to_ir.py`:
```python
"""Python tests for spec_to_ir - aligned with SPARK tests"""

import unittest
from tools.spec_to_ir import convert_spec_to_ir, load_spec_dir

class TestSpecToIR(unittest.TestCase):
    def test_basic_conversion(self):
        """Test basic spec to IR conversion."""
        pass
    
    def test_type_conversion(self):
        """Test type mapping."""
        pass
    
    # ... align with SPARK test coverage
```

Create `tests/python/test_ir_to_code.py`:
```python
"""Python tests for ir_to_code - aligned with SPARK tests"""

import unittest
from tools.ir_to_code import emit_c, emit_rust, emit_python

class TestIRToCode(unittest.TestCase):
    def test_emit_c_function(self):
        """Test C code generation."""
        pass
    
    def test_emit_rust_function(self):
        """Test Rust code generation."""
        pass
    
    # ... all target languages
```

Create `tests/python/test_ir_optimizer.py`:
```python
"""Python tests for ir_optimizer - aligned with SPARK tests"""

import unittest
from tools.ir_optimizer import (
    constant_folding,
    constant_propagation,
    dead_code_elimination
)

class TestIROptimizer(unittest.TestCase):
    """25 tests aligned with tests/spark/optimizer/test_optimizer.adb"""
    pass
```

---

### Phase 2: Rust Feature Completion (v0.9.0)

#### 2.1 Implement ir_optimizer in Rust

Add to `tools/rust/src/optimizer.rs`:
```rust
/// IR Optimizer - Rust Implementation
/// Aligns with Ada SPARK optimizer

pub fn constant_folding(ir: &mut IRModule) -> OptimizationResult {
    // Implementation aligned with Ada SPARK
}

pub fn constant_propagation(ir: &mut IRModule) -> OptimizationResult {
    // Implementation aligned with Ada SPARK
}

pub fn dead_code_elimination(ir: &mut IRModule) -> OptimizationResult {
    // Implementation aligned with Ada SPARK
}
```

#### 2.2 Complete Rust Test Suite

Already started - complete the test implementations in `tools/rust/src/optimizer.rs`.

---

### Phase 3: Cross-Implementation Validation (v0.9.0)

#### 3.1 Golden Master Tests

Create tests that verify all three implementations produce identical output:

```python
# tests/cross_validation/test_golden_master.py

import subprocess
import json
import filecmp

def test_spec_to_ir_equivalence():
    """Verify Python, SPARK, and Rust produce identical IR."""
    # Run all three implementations
    # Compare outputs
    assert filecmp.cmp('python.ir.json', 'spark.ir.json')
    assert filecmp.cmp('spark.ir.json', 'rust.ir.json')

def test_ir_to_code_equivalence():
    """Verify all emitters produce equivalent code."""
    pass
```

#### 3.2 Round-Trip Tests

```python
# tests/cross_validation/test_roundtrip.py

def test_ir_roundtrip():
    """Spec -> IR -> Code -> IR should be consistent."""
    pass
```

---

## Implementation Timeline

### v0.9.0 (Current Sprint)

**Week 1-2: Python Completion**
- [ ] Add Rust emitter to ir_to_code.py
- [ ] Add Python emitter to ir_to_code.py
- [ ] Add JavaScript emitter to ir_to_code.py
- [ ] Add Zig emitter to ir_to_code.py
- [ ] Add Go emitter to ir_to_code.py
- [ ] Add Ada emitter to ir_to_code.py
- [ ] Create ir_optimizer.py with all passes
- [ ] Create Python test suite (25 optimizer tests)

**Week 3-4: Rust Completion**
- [ ] Implement optimizer in Rust
- [ ] Complete Rust test suite
- [ ] Fix Windows linking issues

**Week 5-6: Validation**
- [ ] Create cross-validation tests
- [ ] Run golden master comparisons
- [ ] Document any intentional divergences

### v1.0.0 (Future)

- [ ] All three implementations at 100% feature parity
- [ ] Comprehensive test coverage in all languages
- [ ] Performance benchmarks
- [ ] User can choose implementation via CLI flag

---

## CLI Alignment

### Unified Interface

All three implementations will support identical CLI:

```bash
# Python
python tools/spec_to_ir.py --spec-root specs/ --out output.ir.json

# Ada SPARK (identical interface)
tools/spark/bin/stunir_spec_to_ir_main --spec-root specs/ --out output.ir.json

# Rust (identical interface)
stunir-native spec-to-ir --spec-root specs/ --out output.ir.json
```

### Implementation Selector

Add wrapper script that selects implementation:

```bash
# stunir wrapper
stunir --impl python spec-to-ir --spec-root specs/ --out output.ir.json
stunir --impl spark spec-to-ir --spec-root specs/ --out output.ir.json
stunir --impl rust spec-to-ir --spec-root specs/ --out output.ir.json
```

---

## Benefits of Alignment

1. **User Choice**: Users select based on needs
   - Python: Development, scripting, ecosystem
   - SPARK: Safety-critical, verification
   - Rust: Performance, systems integration

2. **Validation**: Three implementations cross-validate each other

3. **Redundancy**: If one has issues, others available

4. **Ecosystem**: Python tools integrate with Python ecosystem

5. **Education**: Python code is readable reference

---

## Success Criteria

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| Feature Parity | 100% | All emitters in all languages |
| Test Coverage | >90% | Lines covered in all implementations |
| Output Equivalence | 100% | Golden master tests pass |
| Performance | <2x | Python within 2x of Rust for same task |
| Documentation | Complete | All implementations documented |

---

## Resource Requirements

| Task | Effort | Owner |
|------|--------|-------|
| Python emitters | 3 days | Python team |
| Python optimizer | 2 days | Python team |
| Python tests | 2 days | QA team |
| Rust optimizer | 3 days | Rust team |
| Cross-validation | 2 days | QA team |
| Documentation | 1 day | Docs team |
| **Total** | **13 days** | |

---

## Conclusion

Rather than deprecating Python, STUNIR will maintain three fully-featured implementations. This provides maximum flexibility for users while ensuring correctness through cross-validation.

**Decision:** Align all implementations to 100% feature parity by v1.0.0.

---

**Approved by:** STUNIR Technical Lead  
**Date:** February 3, 2026  
**Review Date:** v0.9.0 release