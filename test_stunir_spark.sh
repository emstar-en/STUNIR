"#!/bin/bash
# Test script for STUNIR-generated SPARK/Ada code
# Usage: ./test_stunir_spark.sh

set -euo pipefail  # Fail on errors, undefined variables, or pipeline failures

# Configuration
STUNIR_SPEC=\"spark_arith.stunir\"
EXPECTED_OUTPUT=\"main.adb\"
EXPECTED_RESULT=20  # (2 + 3) * 4 = 20
TEMP_DIR=$(mktemp -d)
GPR_FILE=\"$TEMP_DIR/main.gpr\"

# Cleanup function
cleanup() {
    rm -rf \"$TEMP_DIR\"
}
trap cleanup EXIT

echo \"=== STUNIR SPARK Test Script ===\"

# Step 1: Generate Ada code using STUNIR tools
echo \"Step 1/4: Generating Ada code from $STUNIR_SPEC...\"
if [ ! -f \"$STUNIR_SPEC\" ]; then
    echo \"Error: Spec file '$STUNIR_SPEC' not found!\"
    exit 1
fi

# Use the specific STUNIR tools we found in the workspace
tools/spark/bin/stunir_spec_to_ir_main.exe --input $STUNIR_SPEC --output $TEMP_DIR/ir.json
if [ ! -f \"$TEMP_DIR/ir.json\" ]; then
    echo \"Error: Failed to generate IR from spec!\"
    exit 1
fi

# Convert IR to Ada code
tools/spark/bin/stunir_ir_to_code_main.exe --input $TEMP_DIR/ir.json --output $TEMP_DIR/main.adb --target Spark
if [ ! -f \"$TEMP_DIR/main.adb\" ]; then
    echo \"Error: Failed to generate Ada code from IR!\"
    exit 1
fi

# Step 2: Create GPR project file for compilation
echo \"Step 2/4: Creating GPR project...\"
cat > \"$GPR_FILE\" <<EOF
project Main is
   for Source_Dirs use (\"$TEMP_DIR\");
   for Object_Dir use \"obj\";
   for Executable (\"main\") use \"main.exe\";
end Main;
EOF

# Step 3: Compile with SPARK (if available) or GNAT
echo \"Step 3/4: Compiling Ada code...\"
gprbuild -P \"$GPR_FILE\" -f || {
    echo \"Compilation failed! Check for SPARK toolchain issues.\"
    exit 1
}

# Step 4: Run the executable and validate output
echo \"Step 4/4: Running and validating output...\"
if [ -x \"$TEMP_DIR/main.exe\" ]; then
    ACTUAL_RESULT=$(\"$TEMP_DIR/main.exe\")
    echo \"Generated code returned: $ACTUAL_RESULT\"

    if [ \"$ACTUAL_RESULT\" -eq \"$EXPECTED_RESULT\" ]; then
        echo \"✅ Test PASSED: Output matches expected result ($EXPECTED_RESULT).\"
    else
        echo \"❌ Test FAILED: Expected $EXPECTED_RESULT, got $ACTUAL_RESULT.\"
        exit 1
    fi
else
    echo \"Error: Executable not found in '$TEMP_DIR'!\"
    exit 1
fi

echo \"=== All tests completed successfully! ===\""
