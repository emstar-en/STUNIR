# Investigation Report: Emitter Status & HLI Plans

**Date:** January 30, 2026  
**Investigator:** DeepAgent  
**Repository:** /home/ubuntu/stunir_repo/

---

## 1. Syntax Error Fix (✅ COMPLETED)

### Issue
- **File:** `/home/ubuntu/stunir_repo/targets/embedded/emitter.py`
- **Line:** 432 (original), now 433
- **Error:** `SyntaxError: f-string: single '}' is not allowed`

### Root Cause
The f-string in `_emit_readme()` method contained a dictionary lookup with nested braces that broke Python 3.12+ f-string parsing:
```python
# BROKEN:
Requires `{{'arm': 'arm-none-eabi', ...}.get(self.arch, 'arm-none-eabi')}` toolchain.
```

### Solution Applied
Extracted the dictionary lookup to a variable before the f-string:
```python
toolchain = {'arm': 'arm-none-eabi', 'avr': 'avr', 'mips': 'mips-elf'}.get(self.arch, 'arm-none-eabi')
return f"""...
Requires `{toolchain}` toolchain.
..."""
```

### Verification
```bash
python3 -m py_compile targets/embedded/emitter.py  # ✅ Syntax OK
```

### Commit
`e4353d1` - Fix f-string syntax error in embedded emitter

---

## 2. Ada SPARK Emitter Status

### Current State

| Location | Content | Implementation |
|----------|---------|----------------|
| `tools/spark/` | Core tools (spec_to_ir, ir_to_code) | ✅ Ada SPARK |
| `targets/` | **28 target emitters** | ❌ Python only |

### Analysis

The SPARK migration covered **core tools only**:
- `stunir_spec_to_ir` - converts specs to IR
- `stunir_ir_to_code` - generic code emission

However, **28 target-specific emitters remain Python-only**:

| Category | Emitters | SPARK Version |
|----------|----------|---------------|
| Assembly | arm, x86 | ❌ None |
| Embedded | embedded | ❌ None |
| GPU | gpu | ❌ None |
| WASM | wasm | ❌ None |
| Mobile | mobile | ❌ None |
| FPGA | fpga | ❌ None |
| Polyglot | c89, c99, rust | ❌ None |
| Lisp | 8 variants | ❌ None |
| Prolog | multiple | ❌ None |
| Other | asm, lexer, parser, grammar, planning, etc. | ❌ None |

### Why Python Emitter Is Being Used

The workflow falls back to Python because:
1. Ada SPARK `stunir_ir_to_code` only handles generic emission
2. **No SPARK emitters exist for specific targets** like `embedded`
3. Build scripts detect missing SPARK binaries and fall back to Python

### Migration Gap

The `MIGRATION_SUMMARY_ADA_SPARK.md` claims migration is "COMPLETED" but only covers:
- 2 core tools in `tools/spark/`
- Documentation updates
- Build script fallback logic

**NOT migrated:** All 28 target-specific emitters in `targets/` directory.

---

## 3. HLI Plans Status

### Found Documents

| Document | Phase | Status |
|----------|-------|--------|
| `HLI_SPARK_MIGRATION_PHASE1.md` | Core Utilities | ✅ Complete |
| `HLI_SPARK_MIGRATION_PHASE2.md` | Build System | ⚠️ **MISSING** |
| `HLI_SPARK_MIGRATION_PHASE3.md` | Test Infrastructure | 🔄 In Progress |
| `HLI_SPARK_MIGRATION_PHASE4.md` | Tool Integration | ✅ Complete |

### Missing Documents

1. **`HLI_SPARK_MIGRATION_PHASE2.md`** - Build System migration document is missing
   - Phase 3 document claims Phase 2 is "Complete" but no documentation exists
   
2. **`stunir_implementation_framework/`** - Referenced in multiple README files but directory doesn't exist:
   - `phase6/HLI_GRAMMAR_IR.md`
   - `phase8/HLI_BEAM_VM.md`
   - `phase8/HLI_FUNCTIONAL_LANGUAGES.md`
   - `phase8/HLI_FSHARP.md`
   - Multiple other HLI documents

### HLI References Found

Files referencing HLI documents (14 total):
- `/home/ubuntu/stunir_repo/ir/grammar/README.md`
- `/home/ubuntu/stunir_repo/ir/oop/README.md`
- `/home/ubuntu/stunir_repo/ir/actor/README.md`
- `/home/ubuntu/stunir_repo/targets/functional/README.md`
- `/home/ubuntu/stunir_repo/targets/beam/README.md`
- `/home/ubuntu/stunir_repo/targets/oop/README.md`
- `/home/ubuntu/stunir_repo/targets/systems/README.md`
- `/home/ubuntu/stunir_repo/targets/lisp/README.md`
- `/home/ubuntu/stunir_repo/targets/grammar/README.md`
- And others...

---

## 4. Recommendations

### Immediate Actions

1. **✅ DONE:** Syntax error fixed - Ardupilot test can now proceed

2. **Create Phase 2 HLI Document:**
   ```bash
   # Create the missing Phase 2 document
   touch docs/migration/HLI_SPARK_MIGRATION_PHASE2.md
   ```

### Short-Term (1-2 weeks)

3. **Prioritize Embedded SPARK Emitter:**
   - Create `tools/spark/src/stunir_embedded_emitter.ads`
   - Create `tools/spark/src/stunir_embedded_emitter.adb`
   - This unblocks Ardupilot test from using SPARK end-to-end

4. **Create Missing Framework Directory:**
   ```bash
   mkdir -p /home/ubuntu/stunir_implementation_framework/phase6
   mkdir -p /home/ubuntu/stunir_implementation_framework/phase8
   ```
   - Either create the referenced HLI documents or fix the broken links

### Medium-Term (1-3 months)

5. **Target Emitter Migration Plan:**
   
   | Priority | Emitter | Justification |
   |----------|---------|---------------|
   | HIGH | embedded | Safety-critical (Ardupilot) |
   | HIGH | c89, c99 | Common targets |
   | MEDIUM | rust | Modern systems |
   | MEDIUM | wasm | Web deployment |
   | LOW | lisp variants | Specialized use |

6. **Complete Phase 3 Migration:**
   - Test infrastructure SPARK migration is still in progress
   - Complete before adding new target emitters

### Documentation Updates Needed

7. **Update `MIGRATION_SUMMARY_ADA_SPARK.md`:**
   - Change status from "COMPLETED" to "CORE TOOLS COMPLETED"
   - Add section on target emitter migration roadmap
   - List which emitters still need SPARK versions

---

## Summary

| Item | Status |
|------|--------|
| Syntax Error | ✅ Fixed |
| Core SPARK Tools | ✅ Complete |
| Target SPARK Emitters | ❌ 0/28 migrated |
| HLI Phase 1 | ✅ Complete |
| HLI Phase 2 | ⚠️ Document missing |
| HLI Phase 3 | 🔄 In Progress |
| HLI Phase 4 | ✅ Complete |
| Implementation Framework | ❌ Directory missing |

**Next Priority:** Create SPARK version of embedded emitter for Ardupilot test.
