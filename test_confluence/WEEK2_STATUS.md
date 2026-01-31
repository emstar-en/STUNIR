# Week 2 Status: Confluence Verification

**Date:** January 31, 2026  
**Status:** ⚠️ IN PROGRESS  
**Branch:** devsite

---

## Current State

### ✅ What's Working

**Python Pipeline:**
- ✅ `spec_to_ir.py` generates proper semantic IR
- ✅ `ir_to_code.py` generates code successfully
- ✅ End-to-end pipeline works: Spec → IR → Code
- ✅ Tested with ardupilot_test specs
- ✅ Output: Valid Python code with function stubs

**Rust Pipeline:**
- ✅ `stunir_spec_to_ir` generates proper semantic IR
- ⚠️ `stunir_ir_to_code` requires `--target` parameter (not auto-detected)

**SPARK Pipeline:**
- ⚠️ `stunir_spec_to_ir_main` fails silently (no output, no error)
- ❌ Cannot test ir_to_code without IR

---

## Issues Found

### 🚨 Issue #1: SPARK spec_to_ir Fails Silently
**Command:**
```bash
./tools/spark/bin/stunir_spec_to_ir_main --spec-root spec/ardupilot_test --out /tmp/test.json
```

**Result:** No output, no error, no IR file created

**Impact:** Cannot test SPARK pipeline

---

### 🚨 Issue #2: Rust ir_to_code Requires Target Parameter
**Command:**
```bash
./tools/rust/target/release/stunir_ir_to_code ir.json -o output.txt
```

**Error:**
```
error: the following required arguments were not provided:
  --target <TARGET>

Usage: stunir_ir_to_code --target <TARGET> --output <OUTPUT> <IR_FILE>
```

**Impact:** Cannot auto-test Rust pipeline without knowing target

---

### 🚨 Issue #3: Python ir_to_code Requires Multiple Parameters
**Command:**
```bash
python3 tools/ir_to_code.py ir.json output.txt
```

**Error:**
```
usage: ir_to_code.py [-h] --ir IR --lang LANG --templates TEMPLATES --out OUT
```

**Correct Usage:**
```bash
python3 tools/ir_to_code.py --ir ir.json --lang python --templates templates/python --out output_dir
```

**Impact:** Confluence test script needs updating

---

## Manual Testing Results

### Python Pipeline ✅
```bash
# Step 1: Spec → IR
python3 tools/spec_to_ir.py --spec-root spec/ardupilot_test --out /tmp/test.json
# Result: SUCCESS - Generated semantic IR with 11 functions

# Step 2: IR → Code
python3 tools/ir_to_code.py --ir /tmp/test.json --lang python --templates templates/python --out /tmp/output
# Result: SUCCESS - Generated mavlink_handler.py
```

**Output Sample:**
```python
def parse_heartbeat(buffer, len):
    """parse_heartbeat"""
    # TODO: implement
    raise NotImplementedError()
```

---

## Next Steps

### Immediate (Today)
1. **Fix SPARK spec_to_ir** - Debug why it fails silently
2. **Update confluence test script** - Use correct command-line arguments
3. **Add target detection** - Auto-detect target from IR or spec

### Short-term (This Week)
4. **Complete confluence testing** - Test all 3 working pipelines
5. **Document API differences** - Create unified interface spec
6. **Generate confluence report** - Show which pipelines match

---

## Confluence Test Script Status

**Current:** ❌ Failing - Incorrect command-line arguments  
**Fixed:** ⚠️ Needs update for:
- Python: `--ir`, `--lang`, `--templates`, `--out`
- Rust: `--target`, `--output`
- SPARK: Debug silent failure

---

## Week 2 Progress

- [x] Create confluence test infrastructure
- [x] Test Python pipeline manually
- [x] Identify SPARK issues
- [x] Identify Rust issues
- [ ] Fix SPARK spec_to_ir
- [ ] Update confluence test script
- [ ] Run full confluence tests
- [ ] Generate confluence report
- [ ] Commit to devsite

**Estimated Completion:** End of Week 2 (pending SPARK fix)
