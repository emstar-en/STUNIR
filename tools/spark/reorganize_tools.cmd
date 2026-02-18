@echo off
REM STUNIR Tool Reorganization Batch Script
REM Moves 73 tools from powertools\ to orthogonal directory structure

setlocal enabledelayedexpansion
set SRC=C:\Users\MSTAR\AppData\Roaming\AbacusAI\Agent Workspaces\tools\spark\src

echo ========================================
echo STUNIR Tool Reorganization
echo ========================================
echo.

cd "%SRC%\powertools"

REM Core files (2 + headers)
echo [1/12] Moving core files...
move /Y stunir_json_parser.adb "%SRC%\core\" 2>nul
move /Y stunir_json_parser.ads "%SRC%\core\" 2>nul
move /Y command_utils.adb "%SRC%\core\" 2>nul
move /Y command_utils.ads "%SRC%\core\" 2>nul
echo    ✓ Core files moved

REM JSON files (12)
echo [2/12] Moving JSON tools...
move /Y json_extract.adb "%SRC%\json\" 2>nul
move /Y json_formatter.adb "%SRC%\json\" 2>nul
move /Y json_merge.adb "%SRC%\json\" 2>nul
move /Y json_merge_arrays.adb "%SRC%\json\" 2>nul
move /Y json_merge_objects.adb "%SRC%\json\" 2>nul
move /Y json_path_eval.adb "%SRC%\json\" 2>nul
move /Y json_path_parser.adb "%SRC%\json\" 2>nul
move /Y json_read.adb "%SRC%\json\" 2>nul
move /Y json_validate.adb "%SRC%\json\" 2>nul
move /Y json_validator.adb "%SRC%\json\" 2>nul
move /Y json_value_format.adb "%SRC%\json\" 2>nul
move /Y json_write.adb "%SRC%\json\" 2>nul
echo    ✓ JSON tools moved

REM Type files (9)
echo [3/12] Moving type tools...
move /Y type_dependency.adb "%SRC%\types\" 2>nul
move /Y type_expand.adb "%SRC%\types\" 2>nul
move /Y type_lookup.adb "%SRC%\types\" 2>nul
move /Y type_map.adb "%SRC%\types\" 2>nul
move /Y type_map_cpp.adb "%SRC%\types\" 2>nul
move /Y type_map_target.adb "%SRC%\types\" 2>nul
move /Y type_normalize.adb "%SRC%\types\" 2>nul
move /Y type_resolve.adb "%SRC%\types\" 2>nul
move /Y type_resolver.adb "%SRC%\types\" 2>nul
echo    ✓ Type tools moved

REM Function files (4)
echo [4/12] Moving function tools...
move /Y func_dedup.adb "%SRC%\functions\" 2>nul
move /Y func_parse_body.adb "%SRC%\functions\" 2>nul
move /Y func_parse_sig.adb "%SRC%\functions\" 2>nul
move /Y func_to_ir.adb "%SRC%\functions\" 2>nul
echo    ✓ Function tools moved

REM Detection files (2)
echo [5/12] Moving detection tools...
move /Y format_detect.adb "%SRC%\detection\" 2>nul
move /Y lang_detect.adb "%SRC%\detection\" 2>nul
echo    ✓ Detection tools moved

REM Spec files (6)
echo [6/12] Moving spec tools...
move /Y extraction_to_spec.adb "%SRC%\spec\" 2>nul
move /Y spec_extract_funcs.adb "%SRC%\spec\" 2>nul
move /Y spec_extract_module.adb "%SRC%\spec\" 2>nul
move /Y spec_extract_types.adb "%SRC%\spec\" 2>nul
move /Y spec_validate.adb "%SRC%\spec\" 2>nul
move /Y spec_validate_schema.adb "%SRC%\spec\" 2>nul
echo    ✓ Spec tools moved

REM IR files (10)
echo [7/12] Moving IR tools...
move /Y ir_add_metadata.adb "%SRC%\ir\" 2>nul
move /Y ir_check_functions.adb "%SRC%\ir\" 2>nul
move /Y ir_check_required.adb "%SRC%\ir\" 2>nul
move /Y ir_check_types.adb "%SRC%\ir\" 2>nul
move /Y ir_extract_funcs.adb "%SRC%\ir\" 2>nul
move /Y ir_extract_module.adb "%SRC%\ir\" 2>nul
move /Y ir_gen_functions.adb "%SRC%\ir\" 2>nul
move /Y ir_merge_funcs.adb "%SRC%\ir\" 2>nul
move /Y ir_validate.adb "%SRC%\ir\" 2>nul
move /Y ir_validate_schema.adb "%SRC%\ir\" 2>nul
echo    ✓ IR tools moved

REM Codegen files (12)
echo [8/12] Moving codegen tools...
move /Y cpp_header_gen.adb "%SRC%\codegen\" 2>nul
move /Y cpp_impl_gen.adb "%SRC%\codegen\" 2>nul
move /Y cpp_sig_normalize.adb "%SRC%\codegen\" 2>nul
move /Y sig_gen_cpp.adb "%SRC%\codegen\" 2>nul
move /Y sig_gen_python.adb "%SRC%\codegen\" 2>nul
move /Y sig_gen_rust.adb "%SRC%\codegen\" 2>nul
move /Y code_add_comments.adb "%SRC%\codegen\" 2>nul
move /Y code_format_target.adb "%SRC%\codegen\" 2>nul
move /Y code_gen_func_body.adb "%SRC%\codegen\" 2>nul
move /Y code_gen_func_sig.adb "%SRC%\codegen\" 2>nul
move /Y code_gen_preamble.adb "%SRC%\codegen\" 2>nul
move /Y code_write.adb "%SRC%\codegen\" 2>nul
echo    ✓ Codegen tools moved

REM Validation files (4)
echo [9/12] Moving validation tools...
move /Y schema_check_format.adb "%SRC%\validation\" 2>nul
move /Y schema_check_required.adb "%SRC%\validation\" 2>nul
move /Y schema_check_types.adb "%SRC%\validation\" 2>nul
move /Y validation_reporter.adb "%SRC%\validation\" 2>nul
echo    ✓ Validation tools moved

REM File utilities (5)
echo [10/12] Moving file tools...
move /Y file_find.adb "%SRC%\files\" 2>nul
move /Y file_hash.adb "%SRC%\files\" 2>nul
move /Y file_indexer.adb "%SRC%\files\" 2>nul
move /Y file_reader.adb "%SRC%\files\" 2>nul
move /Y file_writer.adb "%SRC%\files\" 2>nul
echo    ✓ File tools moved

REM Verification files (3)
echo [11/12] Moving verification tools...
move /Y hash_compute.adb "%SRC%\verification\" 2>nul
move /Y receipt_generate.adb "%SRC%\verification\" 2>nul
move /Y manifest_generate.adb "%SRC%\verification\" 2>nul
echo    ✓ Verification tools moved

REM Utility files (4)
echo [12/12] Moving utility tools...
move /Y cli_parser.adb "%SRC%\utils\" 2>nul
move /Y module_to_ir.adb "%SRC%\utils\" 2>nul
move /Y path_normalize.adb "%SRC%\utils\" 2>nul
move /Y toolchain_verify.adb "%SRC%\utils\" 2>nul
echo    ✓ Utility tools moved

echo.
echo ========================================
echo Verification
echo ========================================
cd "%SRC%\powertools"
set /a remaining=0
for %%f in (*.adb) do set /a remaining+=1
echo Remaining .adb files in powertools: %remaining%

if %remaining%==0 (
    echo [SUCCESS] All 73 tools reorganized successfully!
) else (
    echo [WARNING] %remaining% files remain in powertools directory
)

echo.
echo Directory structure is now orthogonal and micronized!
pause
