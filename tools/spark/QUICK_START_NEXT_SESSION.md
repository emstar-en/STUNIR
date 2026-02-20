# STUNIR SPARK Build - Quick Start for Next Session

**Session Goal**: Restore build from 6/51 to 51/51 tools

---

## Current State
- **Tools Built**: 6/51 (BROKEN)
- **Main Programs**: ✅ Still work (code_emitter.exe, stunir_spec_to_ir_main.exe)
- **Confluence**: ✅ Already proven (C, x86, ARM from same IR)

---

## Next Session: First 5 Minutes

### 1. Restore to 48/51 Baseline
```powershell
cd "C:\Users\MSTAR\AppData\Roaming\AbacusAI\Agent Workspaces\tools\spark"

# Check what we broke
git status
git diff src/

# Revert broken files
git checkout -- src/powertools/file_find.adb
git checkout -- src/powertools/path_normalize.adb
git checkout -- src/powertools/cli_parser.adb
git checkout -- src/powertools/manifest_generate.adb
git checkout -- src/powertools/command_utils_pkg.adb
git checkout -- src/stunir_types.adb
git checkout -- src/stunir_types.ads

# Rebuild
gprbuild -P powertools.gpr -j0

# Verify
(Get-ChildItem bin/*.exe).Count  # Target: 48
```

### 2. If Still Broken, Full Revert
```powershell
# Nuclear option
git checkout -- src/
gprbuild -P powertools.gpr -j0
(Get-ChildItem bin/*.exe).Count  # Should be 48
```

---

## Then Fix ONE File at a Time

### Fix 1: command_utils_pkg.adb (Easiest)
**Line 54** - Add Success parameter:
```ada
declare
   Success : Boolean;
begin
   GNAT.OS_Lib.Delete_File (Path, Success);
   if not Success then
      Put_Line (Standard_Error, "Warning: Failed to delete: " & Path);
   end if;
end;
```

**Test**: 
```powershell
gprbuild -P powertools.gpr command_utils_pkg.adb
(Get-ChildItem bin/*.exe).Count  # Should increase
```

### Fix 2-4: Follow BUILD_RECOVERY_PLAN.md

---

## Key Commands

**Check build status**:
```powershell
cd tools/spark
(Get-ChildItem bin/*.exe).Count
Get-ChildItem bin/*.exe | Select-Object -ExpandProperty Name
```

**Test confluence** (verify still works):
```powershell
cd ..\..
.\tools\spark\bin\code_emitter.exe -i test_ir.json -o out -t c
.\tools\spark\bin\code_emitter.exe -i test_ir.json -o out -t x86
```

**View changes**:
```powershell
git diff src/
git status
```

---

## Reference Documents
- **Full recovery plan**: `tools/spark/docs/BUILD_RECOVERY_PLAN.md`
- **Session summary**: `tools/spark/docs/SESSION_SUMMARY.md`
- **Alpha status**: `tools/spark/docs/SPARK_ALPHA_STATUS.md`

---

## Success Criteria
- [ ] 48+ tools restored
- [ ] Both main programs work
- [ ] Confluence test passes
- [ ] Working toward 51/51

---

**Remember**: Fix ONE file at a time, rebuild after each, verify tool count!
