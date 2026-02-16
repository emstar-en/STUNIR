# STUNIR SPARK Components Makefile
# DO-333 / SPARK 2014 compliant build system
# Copyright (C) 2026 STUNIR Project
# SPDX-License-Identifier: Apache-2.0

# Compiler settings
GNATMAKE = gnatmake
GNATPROVE = gnatprove
GPRBUILD = gprbuild

# Directories
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin
TEST_DIR = tests

# Build modes
MODE = prove  # prove, debug, release

# Proof level (0-4)
PROOF_LEVEL = 4

# Colors for output
RED = \033[0;31m
GREEN = \033[0;32m
YELLOW = \033[1;33m
NC = \033[0m # No Color

# Targets
.PHONY: all clean tools core parsing validation analysis testing utils prove test install

# Default target
all: directories tools
	@echo "$(GREEN)SPARK components built successfully$(NC)"

# Create necessary directories
directories:
	@powershell -NoProfile -Command "if (-not (Test-Path '$(OBJ_DIR)')) { New-Item -ItemType Directory -Path '$(OBJ_DIR)' | Out-Null }"
	@powershell -NoProfile -Command "if (-not (Test-Path '$(BIN_DIR)')) { New-Item -ItemType Directory -Path '$(BIN_DIR)' | Out-Null }"

# ============================================================================
# Toolchain Build
# ============================================================================

tools: directories
	@echo "$(YELLOW)Building STUNIR tools...$(NC)"
	$(GPRBUILD) -P stunir_tools.gpr -Xmode=$(MODE)
	@echo "$(GREEN)STUNIR tools built$(NC)"

# Compatibility targets
core: tools
parsing: tools
validation: tools
analysis: tools
testing: tools
utils: tools

# ============================================================================
# Formal Verification
# ============================================================================

prove:
	@echo "$(YELLOW)Running formal verification...$(NC)"
	$(GNATPROVE) -P stunir_tools.gpr --level=$(PROOF_LEVEL) --checks-as-errors
	@echo "$(GREEN)Formal verification complete$(NC)"

# ============================================================================
# Testing
# ============================================================================

test: all
	@echo "$(YELLOW)Running SPARK test suite...$(NC)"
	$(BIN_DIR)/test_runner
	@echo "$(GREEN)Tests complete$(NC)"

# ============================================================================
# Installation
# ============================================================================

install: all
	@echo "$(YELLOW)Installing SPARK binaries...$(NC)"
	install -d $(DESTDIR)/usr/local/bin
	install -m 755 $(BIN_DIR)/spec_assembler $(DESTDIR)/usr/local/bin/
	install -m 755 $(BIN_DIR)/ir_converter $(DESTDIR)/usr/local/bin/
	install -m 755 $(BIN_DIR)/code_emitter $(DESTDIR)/usr/local/bin/
	install -m 755 $(BIN_DIR)/pipeline_driver $(DESTDIR)/usr/local/bin/
	@echo "$(GREEN)Installation complete$(NC)"

# ============================================================================
# Cleanup
# ============================================================================

clean:
	@echo "$(YELLOW)Cleaning build artifacts...$(NC)"
	rm -rf $(OBJ_DIR)/*
	rm -rf $(BIN_DIR)/*
	find . -name "*.ali" -delete
	find . -name "*.o" -delete
	find . -name "*~" -delete
	@echo "$(GREEN)Clean complete$(NC)"

distclean: clean
	rm -rf $(OBJ_DIR)
	rm -rf $(BIN_DIR)

# ============================================================================
# Help
# ============================================================================

help:
	@echo "STUNIR SPARK Components - Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all          - Build all components (default)"
	@echo "  core         - Build core pipeline components"
	@echo "  parsing      - Build parsing components"
	@echo "  validation   - Build validation components"
	@echo "  analysis     - Build analysis components"
	@echo "  testing      - Build testing components"
	@echo "  utils        - Build utility components"
	@echo "  prove        - Run formal verification on all components"
	@echo "  test         - Run test suite"
	@echo "  install      - Install binaries to system"
	@echo "  clean        - Remove build artifacts"
	@echo "  distclean    - Remove all generated files"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Variables:"
	@echo "  MODE=prove|debug|release  - Build mode (default: prove)"
	@echo "  PROOF_LEVEL=0-4           - Proof effort level (default: 4)"
