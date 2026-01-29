#!/bin/bash
# STUNIR DO-333 Formal Methods Verification Script
# Copyright (C) 2026 STUNIR Project
# SPDX-License-Identifier: Apache-2.0
#
# Shell script wrapper for DO-333 formal verification tools

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOL_DIR="${SCRIPT_DIR}/../tools/do333"
BIN_DIR="${TOOL_DIR}/bin"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Usage
usage() {
    echo "STUNIR DO-333 Formal Methods Verification"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  build        Build DO-333 tools"
    echo "  prove        Run SPARK proofs on tools"
    echo "  analyze      Analyze a project for formal verification"
    echo "  report       Generate certification reports"
    echo "  demo         Run demonstration"
    echo "  help         Show this help"
    echo ""
    echo "Environment:"
    echo "  STUNIR_ENABLE_COMPLIANCE=1  Enable compliance tools (required)"
    echo ""
}

# Check if compliance is enabled
check_compliance() {
    if [ "${STUNIR_ENABLE_COMPLIANCE:-0}" != "1" ]; then
        echo -e "${YELLOW}DO-333 support is disabled.${NC}"
        echo "Set STUNIR_ENABLE_COMPLIANCE=1 to enable."
        exit 0
    fi
}

# Build tools
build_tools() {
    check_compliance
    echo -e "${GREEN}Building DO-333 tools...${NC}"
    cd "${TOOL_DIR}"
    
    if command -v gprbuild &> /dev/null; then
        make build
        echo -e "${GREEN}Build complete.${NC}"
    else
        echo -e "${YELLOW}gprbuild not found. Install GNAT to build DO-333 tools.${NC}"
        exit 1
    fi
}

# Run SPARK proofs
run_proofs() {
    check_compliance
    echo -e "${GREEN}Running SPARK proofs on DO-333 tools...${NC}"
    cd "${TOOL_DIR}"
    
    if command -v gnatprove &> /dev/null; then
        make prove
        echo -e "${GREEN}Proofs complete.${NC}"
    else
        echo -e "${YELLOW}gnatprove not found. Install SPARK to run proofs.${NC}"
        exit 1
    fi
}

# Analyze project
analyze_project() {
    check_compliance
    local project="${1:-}"
    
    if [ -z "$project" ]; then
        echo -e "${RED}Error: No project file specified.${NC}"
        echo "Usage: $0 analyze <project.gpr>"
        exit 1
    fi
    
    if [ ! -f "$project" ]; then
        echo -e "${RED}Error: Project file not found: $project${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Analyzing project: $project${NC}"
    
    if command -v gnatprove &> /dev/null; then
        gnatprove -P "$project" --mode=all --level=2 --prover=z3,cvc5,altergo
    else
        echo -e "${YELLOW}gnatprove not found. Install SPARK to analyze projects.${NC}"
        exit 1
    fi
}

# Generate reports
generate_reports() {
    check_compliance
    local format="${1:-text}"
    
    echo -e "${GREEN}Generating DO-333 reports (format: $format)...${NC}"
    
    if [ -x "${BIN_DIR}/do333_analyzer" ]; then
        "${BIN_DIR}/do333_analyzer" report "$format"
    else
        echo -e "${YELLOW}DO-333 analyzer not built. Run '$0 build' first.${NC}"
        exit 1
    fi
}

# Run demo
run_demo() {
    check_compliance
    echo -e "${GREEN}Running DO-333 demonstration...${NC}"
    
    if [ -x "${BIN_DIR}/do333_analyzer" ]; then
        "${BIN_DIR}/do333_analyzer" demo
    else
        echo -e "${YELLOW}DO-333 analyzer not built. Run '$0 build' first.${NC}"
        exit 1
    fi
}

# Main
main() {
    local cmd="${1:-help}"
    shift || true
    
    case "$cmd" in
        build)
            build_tools
            ;;
        prove)
            run_proofs
            ;;
        analyze)
            analyze_project "$@"
            ;;
        report)
            generate_reports "$@"
            ;;
        demo)
            run_demo
            ;;
        help|--help|-h)
            usage
            ;;
        *)
            echo -e "${RED}Unknown command: $cmd${NC}"
            usage
            exit 1
            ;;
    esac
}

main "$@"
