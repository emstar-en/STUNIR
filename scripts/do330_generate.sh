#!/bin/bash
# STUNIR DO-330 Tool Qualification Package Generator
# Shell Script Wrapper
# Copyright (C) 2026 STUNIR Project
# SPDX-License-Identifier: Apache-2.0

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DO330_DIR="${SCRIPT_DIR}/../tools/do330"
GENERATOR="${DO330_DIR}/bin/do330_generator"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print banner
print_banner() {
    echo "============================================================"
    echo "STUNIR DO-330 Tool Qualification Package Generator"
    echo "============================================================"
    echo ""
}

# Print usage
print_usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --tool=<name>      Tool name to qualify (required)"
    echo "  --tql=<1-5>        TQL level (default: 5)"
    echo "  --dal=<A-E>        DAL level (default: E)"
    echo "  --output=<dir>     Output directory (default: ./certification_package)"
    echo "  --build            Build generator before running"
    echo "  --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --tool=verify_build --tql=4 --output=./pkg"
    echo "  $0 --tool=ir_emitter --tql=5 --dal=C --output=./pkg --build"
}

# Build generator if needed
build_generator() {
    echo -e "${YELLOW}Building DO-330 generator...${NC}"
    cd "${DO330_DIR}"
    make build
    cd - > /dev/null
    echo -e "${GREEN}Build complete.${NC}"
    echo ""
}

# Main
main() {
    print_banner

    # Parse arguments
    TOOL_NAME=""
    TQL_LEVEL="5"
    DAL_LEVEL="E"
    OUTPUT_DIR="./certification_package"
    DO_BUILD=false

    for arg in "$@"; do
        case $arg in
            --tool=*)
                TOOL_NAME="${arg#*=}"
                ;;
            --tql=*)
                TQL_LEVEL="${arg#*=}"
                ;;
            --dal=*)
                DAL_LEVEL="${arg#*=}"
                ;;
            --output=*)
                OUTPUT_DIR="${arg#*=}"
                ;;
            --build)
                DO_BUILD=true
                ;;
            --help)
                print_usage
                exit 0
                ;;
            *)
                echo -e "${RED}Unknown argument: $arg${NC}"
                print_usage
                exit 1
                ;;
        esac
    done

    # Validate required arguments
    if [ -z "$TOOL_NAME" ]; then
        echo -e "${RED}Error: --tool argument is required${NC}"
        print_usage
        exit 1
    fi

    # Build if requested
    if [ "$DO_BUILD" = true ]; then
        build_generator
    fi

    # Check if generator exists
    if [ ! -f "$GENERATOR" ]; then
        echo -e "${YELLOW}Generator not found. Building...${NC}"
        build_generator
    fi

    # Create output directory
    mkdir -p "$OUTPUT_DIR"

    # Run generator
    echo "Generating DO-330 package for: $TOOL_NAME"
    echo "TQL Level: TQL-$TQL_LEVEL"
    echo "DAL Level: DAL-$DAL_LEVEL"
    echo "Output: $OUTPUT_DIR"
    echo ""

    "$GENERATOR" \
        --tool="$TOOL_NAME" \
        --tql="$TQL_LEVEL" \
        --dal="$DAL_LEVEL" \
        --output="$OUTPUT_DIR"

    # Check result
    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}DO-330 package generated successfully!${NC}"
        echo "Output: $OUTPUT_DIR"
    else
        echo ""
        echo -e "${RED}Package generation failed.${NC}"
        exit 1
    fi
}

main "$@"
