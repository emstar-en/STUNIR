# STUNIR Pipeline Fix Plan

## Current Status

The STUNIR pipeline has been run on `ardupilot_crc.cpp` but has several issues:

1. **Function count mismatch**: Original has 25 functions, extraction had 12 (now fixed to 25)
2. **Return types incorrect**: All set to "auto" instead of actual types (uint8_t, uint16_t, etc.)
3. **Parameters missing**: All functions show empty parameter lists
4. **Indexer bug**: `stunir_code_index_main.exe` produces empty indexes (file_count: 0)
5. **Spec assemble limitations**: Doesn't parse signatures to extract return types and parameters

## Baseline (from ardupilot_crc.cpp)

**25 functions found:**
1. `uint16_t crc_crc4(uint16_t *data)`
2. `uint8_t crc_crc8(const uint8_t *p, uint8_t len)`
3. `uint8_t crc8_generic(const uint8_t *buf, const uint16_t buf_len, const uint8_t polynomial, uint8_t initial_value)`
4. `uint8_t crc8_dvb_s2(uint8_t crc, uint8_t a)`
5. `uint8_t crc8_dvb(uint8_t crc, uint8_t a, uint8_t seed)`
6. `uint8_t crc8_dvb_s2_update(uint8_t crc, const void *data, uint32_t length)`
7. `uint8_t crc8_dvb_update(uint8_t crc, const uint8_t* buf, const uint16_t buf_len)`
8. `uint8_t crc8_maxim(const uint8_t *data, uint16_t length)`
9. `uint8_t crc8_sae(const uint8_t *data, uint16_t length)`
10. `uint8_t crc8_rds02uf(const uint8_t *data, uint16_t length)`
11. `uint16_t crc_xmodem_update(uint16_t crc, uint8_t data)`
12. `uint16_t crc_xmodem(const uint8_t *data, uint16_t len)`
13. `uint32_t crc32_small(uint32_t crc, const uint8_t *buf, uint32_t size)`
14. `uint16_t crc16_ccitt(const uint8_t *buf, uint32_t len, uint16_t crc)`
15. `uint16_t crc16_ccitt_r(const uint8_t *buf, uint32_t len, uint16_t crc, uint16_t out)`
16. `uint16_t crc16_ccitt_GDL90(const uint8_t *buf, uint32_t len, uint16_t crc)`
17. `uint16_t calc_crc_modbus(const uint8_t *buf, uint16_t len)`
18. `uint16_t crc_fletcher16(const uint8_t *buffer, uint32_t len)`
19. `void hash_fnv_1a(uint32_t len, const uint8_t* buf, uint64_t* hash)`
20. `uint32_t crc_crc24q(const uint8_t *bytes, uint16_t len)`
21. `uint8_t crc_sum8_with_carry(const uint8_t *p, uint8_t len)`
22. `uint16_t crc_crc16_ibm(uint16_t crc_accum, uint8_t *data_blk_ptr, uint16_t data_blk_size)`
23. `uint64_t crc_crc64(const uint32_t *data, uint16_t num_words)`
24. `uint8_t parity(uint8_t byte)`
25. `uint16_t crc_sum_of_bytes_16(const uint8_t *data, uint16_t count)`

## Issues to Fix

### Issue 1: Fix stunir_code_index_main Directory Traversal Bug

**Location**: `STUNIR-main/tools/spark/src/stunir_code_index.adb`

**Problem**: The indexer produces `file_count: 0` even when run on directories with files.

**Root Cause**: Likely in the `Index_Directory` procedure - path handling or bounded string issues.

**Fix Steps**:
1. Add debug output to trace directory traversal
2. Check if `Start_Search` is finding files
3. Verify path concatenation logic
4. Check bounded string length limits (`Max_Path_Length = 512`)
5. Ensure proper handling of Windows path separators

**Test Command**:
```powershell
& "STUNIR-main\tools\spark\bin\stunir_code_index_main.exe" --input "STUNIR-main\tools\spark\src" --output "test_index.json"
```

**Expected Result**: Non-zero file_count

### Issue 2: Fix stunir_spec_assemble to Parse Signatures

**Location**: `STUNIR-main/tools/spark/src/stunir_spec_assemble.adb`

**Problem**: Lines 84-88 hardcode return_type to "void" then "auto", ignoring actual signature:
```ada
Module.Functions (Idx).Return_Type := Name_Strings.To_Bounded_String ("void");
if Signature'Length > 0 then
   Module.Functions (Idx).Return_Type := Name_Strings.To_Bounded_String ("auto");
end if;
```

**Fix Steps**:
1. Parse the signature string to extract:
   - Return type (before function name)
   - Function name
   - Parameters (inside parentheses)
2. Update the Function_Record type to store parameters
3. Modify Write_Spec_JSON to output parameters

**Example Signature Parsing**:
```
"uint8_t crc_crc8(const uint8_t *p, uint8_t len)"
  ↓
Return: "uint8_t"
Name: "crc_crc8"
Params: [{"name": "p", "type": "const uint8_t *"}, {"name": "len", "type": "uint8_t"}]
```

### Issue 3: Update extraction.json Format

**Current Format** (in `stunir_runs/ardupilot_full/extraction.json`):
```json
{
  "elements": [
    {
      "name": "crc_crc4",
      "type": "function",
      "signature": "uint16_t crc_crc4(uint16_t *data)",
      "return_type": "uint16_t",
      "args": [{"name": "data", "type": "uint16_t *"}]
    }
  ]
}
```

The extraction.json already has the correct data. The issue is spec_assemble doesn't use the `return_type` and `args` fields.

**Fix**: Update spec_assemble to use these fields if present, fall back to signature parsing if not.

### Issue 4: Verify spec_to_ir Handles Parameters

**Location**: `STUNIR-main/tools/spark/src/stunir_spec_to_ir.adb`

**Check**: Does spec_to_ir read parameters from spec.json and include them in ir.json?

**Expected ir.json structure**:
```json
{
  "schema": "stunir_flat_ir_v1",
  "ir_version": "v1",
  "module_name": "crc",
  "functions": [
    {
      "name": "crc_crc8",
      "return_type": "uint8_t",
      "args": [
        {"name": "p", "type": "const uint8_t *"},
        {"name": "len", "type": "uint8_t"}
      ],
      "steps": []
    }
  ]
}
```

### Issue 5: Verify ir_to_code Emits Correct Signatures

**Location**: `STUNIR-main/tools/spark/src/stunir_ir_to_code.adb`

**Check**: Does ir_to_code read return_type and args from ir.json?

**Current Output** (wrong):
```cpp
void crc_crc8(void) {
    /* TODO: Implement */
    return;
}
```

**Expected Output**:
```cpp
uint8_t crc_crc8(const uint8_t *p, uint8_t len) {
    /* TODO: Implement */
    return 0;
}
```

## Implementation Priority

1. **High Priority**: Fix spec_assemble signature parsing (Issue 2)
   - This is the root cause of incorrect output
   
2. **Medium Priority**: Fix code_index_main (Issue 1)
   - Needed for proper indexing but workaround exists
   
3. **Medium Priority**: Verify spec_to_ir parameter handling (Issue 4)
   - May already work once spec.json is correct
   
4. **Medium Priority**: Verify ir_to_code parameter emission (Issue 5)
   - May already work once ir.json is correct

## Files to Modify

1. `STUNIR-main/tools/spark/src/stunir_spec_assemble.adb`
   - Parse signature to extract return type and parameters
   - Update Function_Record type
   - Update Write_Spec_JSON to include parameters

2. `STUNIR-main/tools/spark/src/stunir_code_index.adb` (optional)
   - Add debug output
   - Fix directory traversal

3. `STUNIR-main/tools/spark/src/stunir_spec_to_ir.adb` (verify)
   - Ensure it reads parameters from spec.json

4. `STUNIR-main/tools/spark/src/stunir_ir_to_code.adb` (verify)
   - Ensure it emits parameters in generated code

## Testing Checklist

- [ ] spec_assemble produces correct return types
- [ ] spec_assemble produces correct parameter lists
- [ ] spec_to_ir includes parameters in ir.json
- [ ] ir_to_code emits correct function signatures
- [ ] All 25 functions from ardupilot_crc.cpp are present
- [ ] Function signatures match original (name, return type, parameters)

## Current Workarounds in Place

1. **Indexer**: Using `update_indexes.py` Python script to generate index.machine.json
2. **Extraction**: Using `create_extraction.py` Python script to generate extraction.json from ardupilot_crc.cpp
3. **Lockfile**: Dummy `local_toolchain.lock.json` created for spec_to_ir

These workarounds should be removed once the Ada tools are fixed.