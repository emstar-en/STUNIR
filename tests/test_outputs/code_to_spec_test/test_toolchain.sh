#!/bin/bash
# Test script for the STUNIR Code-to-Spec Toolchain
# This demonstrates the workflow from source code to spec with provenance

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Paths
SPARK_BIN="../../tools/spark/bin"
TEST_DIR="."
OUTPUT_DIR="./output"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}STUNIR Code-to-Spec Toolchain Test${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Step 1: Index the source code
echo -e "${YELLOW}Step 1: Indexing source code...${NC}"
if [ -f "$SPARK_BIN/stunir_code_index" ]; then
    "$SPARK_BIN/stunir_code_index" \
        --input "$TEST_DIR" \
        --output "$OUTPUT_DIR/code_index.json"
    echo -e "${GREEN}✓ Code index created${NC}"
else
    echo -e "${RED}✗ stunir_code_index not built yet${NC}"
    echo "  Run: cd ../../tools/spark && make"
fi
echo ""

# Step 2: Slice the source file
echo -e "${YELLOW}Step 2: Slicing source file...${NC}"
if [ -f "$SPARK_BIN/stunir_code_slice" ]; then
    "$SPARK_BIN/stunir_code_slice" \
        --input "$TEST_DIR/example_crc.c" \
        --output "$OUTPUT_DIR/code_slice.json" \
        --index "$OUTPUT_DIR/code_index.json"
    echo -e "${GREEN}✓ Code slice created${NC}"
else
    echo -e "${RED}✗ stunir_code_slice not built yet${NC}"
fi
echo ""

# Step 3: Show AI guidance
echo -e "${YELLOW}Step 3: AI Extraction Guidance${NC}"
echo "At this point, an AI would:"
echo "  1. Read the code slice: $OUTPUT_DIR/code_slice.json"
echo "  2. Read the source file: $TEST_DIR/example_crc.c"
echo "  3. Extract function signatures and types"
echo "  4. Write extractions to: $OUTPUT_DIR/extracted/"
echo ""

# Create mock AI extraction for demonstration
mkdir -p "$OUTPUT_DIR/extracted"
cat > "$OUTPUT_DIR/extracted/crc_extraction.json" << 'EOF'
{
  "kind": "stunir.extraction.v1",
  "source_file": "example_crc.c",
  "elements": [
    {
      "name": "crc8_dvb_s2",
      "type": "function",
      "signature": "uint8_t crc8_dvb_s2(uint8_t crc, uint8_t a)",
      "source_lines": "8-18",
      "source_hash": "a1b2c3d4e5f6..."
    },
    {
      "name": "crc8_dvb_s2_update",
      "type": "function",
      "signature": "uint8_t crc8_dvb_s2_update(uint8_t crc, const void *data, uint32_t length)",
      "source_lines": "21-28",
      "source_hash": "b2c3d4e5f6a7..."
    },
    {
      "name": "crc16_ccitt",
      "type": "function",
      "signature": "uint16_t crc16_ccitt(const uint8_t *buf, uint32_t len, uint16_t crc)",
      "source_lines": "42-46",
      "source_hash": "c3d4e5f6a7b8..."
    }
  ]
}
EOF
echo -e "${GREEN}✓ Mock AI extraction created${NC}"
echo ""

# Step 4: Assemble spec
echo -e "${YELLOW}Step 4: Assembling spec from extractions...${NC}"
if [ -f "$SPARK_BIN/stunir_spec_assemble" ]; then
    "$SPARK_BIN/stunir_spec_assemble" \
        --input "$OUTPUT_DIR/extracted" \
        --output "$OUTPUT_DIR/crc_spec.json" \
        --index "$OUTPUT_DIR/code_index.json" \
        --name "example_crc"
    echo -e "${GREEN}✓ Spec assembled${NC}"
else
    echo -e "${RED}✗ stunir_spec_assemble not built yet${NC}"
fi
echo ""

# Step 5: Create provenance links
cat > "$OUTPUT_DIR/links.json" << 'EOF'
{
  "links": [
    {
      "spec_element": "crc8_dvb_s2",
      "source_file": "example_crc.c",
      "source_hash": "a1b2c3d4e5f6...",
      "source_lines": "8-18",
      "confidence": 95
    }
  ]
}
EOF

# Step 6: Generate receipt
echo -e "${YELLOW}Step 5: Generating provenance receipt...${NC}"
if [ -f "$SPARK_BIN/stunir_receipt_link" ]; then
    "$SPARK_BIN/stunir_receipt_link" \
        --spec "$OUTPUT_DIR/crc_spec.json" \
        --index "$OUTPUT_DIR/code_index.json" \
        --links "$OUTPUT_DIR/links.json" \
        --output "$OUTPUT_DIR/provenance_receipt.json"
    echo -e "${GREEN}✓ Provenance receipt created${NC}"
else
    echo -e "${RED}✗ stunir_receipt_link not built yet${NC}"
fi
echo ""

# Summary
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Toolchain Test Complete${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Generated artifacts:"
echo "  - Code Index:       $OUTPUT_DIR/code_index.json"
echo "  - Code Slice:       $OUTPUT_DIR/code_slice.json"
echo "  - AI Extraction:    $OUTPUT_DIR/extracted/crc_extraction.json"
echo "  - STUNIR Spec:      $OUTPUT_DIR/crc_spec.json"
echo "  - Provenance Links: $OUTPUT_DIR/links.json"
echo "  - Receipt:          $OUTPUT_DIR/provenance_receipt.json"
echo ""
echo "Next steps:"
echo "  1. Build the SPARK tools: cd ../../tools/spark && make"
echo "  2. Run this script again to execute the full pipeline"
echo "  3. Use the generated spec with: stunir_spec_to_ir"
echo ""
