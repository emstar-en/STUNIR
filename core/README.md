# STUNIR Core SPARK Library

## 🎉 100% SPARK Migration Complete! 🎉

**Status:** ✅ All 4 Phases Complete  
**Total Files:** ~85 | **Total Lines:** ~13,300 | **Total Tests:** ~260

This directory contains Ada SPARK implementations of critical STUNIR components migrated from Python for formal verification and DO-178C/DO-333 compliance.

### Components

| Directory | Description | Python Source |
|-----------|-------------|---------------|
| `type_system/` | Type system with registry | `tools/stunir_types/type_system.py` |
| `ir_transform/` | Control flow analysis | `tools/ir/control_flow.py` |
| `semantic_checker/` | Dead code/semantic analysis | `tools/semantic/checker.py` |
| `ir_validator/` | IR parsing and validation | `tools/parsers/parse_ir.py`, `tools/validators/validate_ir.py` |
| `common/` | Shared utilities (strings, hashes) | Various |

### Building

```bash
# Build the library
make build

# Run SPARK proofs
make prove

# Run unit tests
make test

# Clean build artifacts
make clean
```

### Requirements

- GNAT (GNU Ada Translator) 2021+ with SPARK support
- GNATprove for formal verification

### SPARK Features

- **SPARK_Mode On** - All code is SPARK-compatible
- **No Exceptions** - Uses result types instead
- **Bounded Containers** - All containers have compile-time size limits
- **Pre/Postconditions** - Formal contracts on all operations
- **Loop Variants** - Provable loop termination

### File Structure

```
core/
├── type_system/
│   ├── stunir_types.ads/adb         -- Type definitions and operations
│   └── stunir_type_registry.ads/adb -- Type registry management
├── ir_transform/
│   ├── ir_basic_blocks.ads/adb      -- Basic block representation
│   └── ir_control_flow.ads/adb      -- CFG and dominator analysis
├── semantic_checker/
│   └── semantic_analysis.ads/adb    -- Dead code detection
├── ir_validator/
│   ├── ir_parser.ads/adb            -- IR parsing
│   └── ir_validator.ads/adb         -- IR validation
├── common/
│   ├── stunir_strings.ads/adb       -- Bounded string types
│   └── stunir_hashes.ads/adb        -- Hash utilities
├── tests/
│   ├── test_types.adb               -- Type system tests
│   ├── test_control_flow.adb        -- CFG tests
│   ├── test_semantic.adb            -- Semantic analysis tests
│   └── test_validator.adb           -- Validator tests
├── stunir_core.gpr                   -- GNAT project file
├── Makefile                          -- Build automation
└── README.md                         -- This file
```

### Proof Targets

| Component | VCs | Status |
|-----------|-----|--------|
| Type System | ~40 | ✓ |
| Control Flow | ~35 | ✓ |
| Semantic Checker | ~25 | ✓ |
| IR Validator | ~30 | ✓ |
| **Total** | **~130** | **✓** |

### Integration

These SPARK components can be:
1. Linked as a static library (`libstunir_core.a`)
2. Called via C FFI from Python
3. Used directly from Ada code

### License

Part of the STUNIR project. See repository root for license information.
