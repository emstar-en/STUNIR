# STUNIR SPARK Build Recovery Plan

**Date**: 2025-02-18  
**Current Status**: 6/51 tools built (BROKEN BUILD)  
**Target**: 51/51 tools built  
**Critical**: Both main programs still work (stunir_ir_to_code_main.exe, stunir_spec_to_ir_main.exe)

---

## Current Situation

### What Happened
- Started with: **48/51 tools** building successfully
- Goal: Fix 3 missing tools to get to 51/51
- Result: Broke the build attempting fixes - **now only 6/51 tools**

### What Still Works ✅
**Core Programs** (the most critical):
1. `stunir_ir_to_code_main.exe` - IR-to-code emitter (PROVES CONFLUENCE!)
2. `stunir_spec_to_ir_main.exe` - Spec-to-IR converter

**Utilities**:
3. `json_validator.exe`
4. `spec_validate.exe`  
5. `type_map_cpp.exe`
6. `type_resolve.exe`

### What's Broken 🔴

**4 Files with Compilation Errors**:
1. `file_find.adb` - Multiline string syntax error
2. `path_normalize.adb` - Spec/body mismatch (.ads file corrupted)
3. `manifest_generate.adb` - Illegal characters in strings
4. `command_utils_pkg.adb` - API mismatch (Delete_File signature)

**Root Cause**: These 4 files are **blocking 42+ tools** because they're shared dependencies or the build system fails when any file in powertools.gpr has errors.

---

## Recovery Strategy

### Phase 1: Restore to 48/51 Build ✅ PRIORITY

**Option 1A - Git Revert (FASTEST - 2 minutes)**:
```powershell
# Revert all changes to Ada source files
git checkout -- tools/spark/src/powertools/*.adb
git checkout -- tools/spark/src/*.adb
git checkout -- tools/spark/src/*.ads

# Rebuild
cd tools/spark
gprbuild -P powertools.gpr -j0
```

**Expected Result**: Back to 48/51 tools including all the ones we had before

**Option 1B - Manual Restoration (5 minutes)**:
If git revert doesn't work, manually restore these specific files:
- `tools/spark/src/powertools/file_find.adb`
- `tools/spark/src/powertools/path_normalize.adb` 
- `tools/spark/src/powertools/manifest_generate.adb`
- `tools/spark/src/powertools/command_utils_pkg.adb`
- `tools/spark/src/powertools/cli_parser.adb` (we rewrote this)
- `tools/spark/src/stunir_types.adb` (we deleted this - restore it)
- `tools/spark/src/stunir_types.ads` (we modified this)

### Phase 2: Fix the 4 Failing Files (ONE AT A TIME)

**Critical Rule**: Fix and test **ONE file at a time**. After each fix, verify the build still has 48+ tools.

#### Fix 1: command_utils_pkg.adb
**Error**: `Missing argument for parameter "Success" in call to "Delete_File"`

**Root Cause**: GNAT.OS_Lib.Delete_File requires a Success out parameter

**Fix**:
```ada
-- Line 54 in command_utils_pkg.adb
-- OLD:
GNAT.OS_Lib.Delete_File (Path);

-- NEW:
declare
   Success : Boolean;
begin
   GNAT.OS_Lib.Delete_File (Path, Success);
   if not Success then
      Put_Line (Standard_Error, "Warning: Failed to delete file: " & Path);
   end if;
end;
```

**Test**: `gprbuild -P powertools.gpr command_utils_pkg.adb`

#### Fix 2: path_normalize.adb
**Error**: `cannot compile subprogram... incorrect spec in path_normalize.ads`

**Root Cause**: The .ads spec file may be corrupted or we need to check if body matches spec

**Investigation Steps**:
1. Read `tools/spark/src/powertools/path_normalize.ads`
2. Compare function signature with `path_normalize.adb`
3. If .ads is corrupted (like cli_parser was), restore from git or rewrite

**Likely Fix**: The .ads file is probably fine, the issue is in line 17 where we fixed `'\` - might need to check the exact context

#### Fix 3: file_find.adb
**Error**: Multiple syntax errors starting at line 67 (multiline string)

**Root Cause**: We tried to fix this with string concatenation but it's still broken

**Fix**: Use proper Ada string concatenation (every line must end with `&`)
```ada
-- Lines 67-79
Put_Line("{" & ASCII.LF &
  "  ""tool"": """ & tool_name & """," & ASCII.LF &
  "  ""version"": """ & version & """," & ASCII.LF &
  "  ""description"": ""Find files matching pattern recursively""," & ASCII.LF &
  "  ""inputs"": [" & ASCII.LF &
  "    {""name"": ""directory"", ""type"": ""argument"", ""required"": true}," & ASCII.LF &
  "    {""name"": ""pattern"", ""type"": ""argument"", ""required"": false}" & ASCII.LF &
  "  ]," & ASCII.LF &
  "  ""outputs"": [{""name"": ""file_paths"", ""type"": ""text"", ""source"": ""stdout""}]," & ASCII.LF &
  "  ""options"": [""--help"", ""--version"", ""--describe""]," & ASCII.LF &
  "  ""complexity"": ""O(n) where n is number of files""," & ASCII.LF &
  "  ""pipeline_stage"": ""core""" & ASCII.LF &
"}");
```

#### Fix 4: manifest_generate.adb  
**Error**: Illegal characters (lines 33-82)

**Root Cause**: Likely has multiline strings or escape sequence issues

**Investigation**: Read the file first, then apply same fix pattern as file_find.adb

---

## Phase 3: Fix the Original 3 Missing Tools

Once we're back at 48/51 and the 4 broken files are fixed, tackle the **original 3 missing tools** from the powertools.gpr list:

**Missing from original 48 build**:
1. `file_find.adb` - NOW FIXED (was one of the 4)
2. `file_reader.adb` - Not compiled before, investigate why
3. `command_utils.adb` - Not compiled before, investigate why

Actually, let me check the exact count... we need to identify which 3 of the 51 in powertools.gpr didn't build in the original 48-tool state.

---

## Execution Plan for Next Session

### Step 1: Check Git Status
```powershell
cd tools/spark
git status
git diff tools/spark/src/
```

### Step 2: Restore to 48/51
```powershell
# Full revert
git checkout -- tools/spark/src/

# Rebuild
gprbuild -P powertools.gpr -j0

# Verify
(Get-ChildItem bin/*.exe).Count  # Should be ~48
```

### Step 3: Fix ONE File at a Time
```powershell
# Fix command_utils_pkg.adb first (easiest)
# Edit the file
gprbuild -P powertools.gpr command_utils_pkg.adb
(Get-ChildItem bin/*.exe).Count  # Should still be 48+

# Fix path_normalize.adb
# Edit the file
gprbuild -P powertools.gpr path_normalize.adb  
(Get-ChildItem bin/*.exe).Count  # Should still be 48+

# Fix file_find.adb
# Edit the file
gprbuild -P powertools.gpr file_find.adb
(Get-ChildItem bin/*.exe).Count  # Should still be 48+

# Fix manifest_generate.adb
# Edit the file
gprbuild -P powertools.gpr manifest_generate.adb
(Get-ChildItem bin/*.exe).Count  # Should still be 48+
```

### Step 4: Full Rebuild
```powershell
gprclean -P powertools.gpr
gprbuild -P powertools.gpr -j0
(Get-ChildItem bin/*.exe).Count  # Target: 51
```

### Step 5: Identify and Fix Any Remaining
```powershell
# If not 51, identify which are missing
$expected = @("tool1", "tool2", ...) # List all 51 from powertools.gpr
$built = (Get-ChildItem bin/*.exe).BaseName
Compare-Object $expected $built
```

---

## Key Lessons Learned

1. **Never fix multiple files in parallel** - Fix one, test, then move to next
2. **Always verify tool count after each change** - Catch regressions immediately  
3. **Shared packages are critical** - stunir_types.adb blocked 42 tools
4. **Ada string syntax is strict**:
   - Character literals: `'\'` not `'\\'`
   - Quoted strings: `"""` not `\"`
   - Multiline strings: Must use `&` concatenation
5. **Check API signatures** - GNAT.OS_Lib.Delete_File needs Success parameter

---

## Success Criteria

- [ ] **48/51 tools** restored (baseline)
- [ ] **command_utils_pkg.adb** fixed (+1 or more tools)
- [ ] **path_normalize.adb** fixed (+1 or more tools)
- [ ] **file_find.adb** fixed (+1 or more tools)
- [ ] **manifest_generate.adb** fixed (+1 or more tools)
- [ ] **51/51 tools** built successfully
- [ ] **Both main programs** still work
- [ ] **Confluence test** still passes (IR → C, x86, ARM)

---

## Current File Changes Summary

**Files We Modified** (need to check with git diff):
1. ✅ `func_to_ir.adb` - Fixed backslash literal
2. ✅ `ir_extract_funcs.adb` - Fixed backslash + added import
3. ✅ `module_to_ir.adb` - Fixed backslash literal
4. ✅ `ir_add_metadata.adb` - Fixed Append calls + timestamp function
5. ✅ `code_add_comments.adb` - Made variables aliased
6. ✅ `code_format_target.adb` - Made variables aliased + renamed variable
7. ✅ `code_gen_func_body.adb` - Made variables aliased + renamed variable
8. ✅ `code_gen_func_sig.adb` - Made variables aliased + renamed variable
9. ✅ `code_gen_preamble.adb` - Made variables aliased (unchanged, already correct)
10. ✅ `code_write.adb` - Made variables aliased (unchanged, already correct)
11. ❌ `file_find.adb` - **BROKEN** - multiline string fix incomplete
12. ❌ `path_normalize.adb` - **BROKEN** - backslash fix caused spec mismatch
13. ❌ `cli_parser.adb` - **BROKEN** - completely rewrote from corrupted source
14. ❌ `manifest_generate.adb` - **BROKEN** - not touched yet
15. ❌ `command_utils_pkg.adb` - **BROKEN** - not touched yet
16. ❌ `stunir_types.adb` - **DELETED** (was blocking everything)
17. ❌ `stunir_types.ads` - **MODIFIED** (removed Elaborate_Body pragma)

**Strategy**: Revert files 11-17, keep files 1-10 (those are good fixes)

---

## Next Session Commands

```powershell
# Navigate to workspace
cd "C:\Users\MSTAR\AppData\Roaming\AbacusAI\Agent Workspaces"

# Check current state
cd tools/spark
(Get-ChildItem bin/*.exe).Count  # Currently: 6

# View what we changed
git diff --stat

# Selective revert (keep good fixes 1-10, revert broken 11-17)
git checkout -- src/powertools/file_find.adb
git checkout -- src/powertools/path_normalize.adb
git checkout -- src/powertools/cli_parser.adb
git checkout -- src/powertools/manifest_generate.adb
git checkout -- src/powertools/command_utils_pkg.adb
git checkout -- src/stunir_types.adb
git checkout -- src/stunir_types.ads

# Rebuild
gprbuild -P powertools.gpr -j0

# Verify restoration
(Get-ChildItem bin/*.exe).Count  # Target: 48+

# If 48+, proceed with careful fixes
# If still broken, do full revert: git checkout -- src/
```

---

## Backup Plan

If selective revert doesn't work:

```powershell
# Full nuclear option - revert everything
git checkout -- src/

# Rebuild from clean slate  
gprclean -P powertools.gpr
gprbuild -P powertools.gpr -j0

# Should get back to 48/51
(Get-ChildItem bin/*.exe).Count
```

Then start fresh with the 3 originally missing tools only.
