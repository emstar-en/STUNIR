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
.PHONY: all clean core parsing validation analysis testing utils prove test install

# Default target
all: directories core parsing validation
	@echo "$(GREEN)SPARK components built successfully$(NC)"

# Create necessary directories
directories:
	@mkdir -p $(OBJ_DIR)
	@mkdir -p $(BIN_DIR)

# ============================================================================
# Core Components (Phase 1)
# ============================================================================

core: directories
	@echo "$(YELLOW)Building Core Components...$(NC)"
	$(GNATMAKE) -P core.gpr -Xmode=$(MODE) -aO$(OBJ_DIR)
	@echo "$(GREEN)Core components built$(NC)"

# Core formal verification
core-prove: core
	@echo "$(YELLOW)Running formal verification on Core...$(NC)"
	$(GNATPROVE) -P core.gpr --level=$(PROOF_LEVEL) --checks-as-errors
	@echo "$(GREEN)Core verification complete$(NC)"

# ============================================================================
# Parsing Components (Phase 2)
# ============================================================================

parsing: core
	@echo "$(YELLOW)Building Parsing Components...$(NC)"
	$(GNATMAKE) -P parsing.gpr -Xmode=$(MODE) -aO$(OBJ_DIR)
	@echo "$(GREEN)Parsing components built$(NC)"

parsing-prove: parsing
	@echo "$(YELLOW)Running formal verification on Parsing...$(NC)"
	$(GNATPROVE) -P parsing.gpr --level=3 --checks-as-errors
	@echo "$(GREEN)Parsing verification complete$(NC)"

# ============================================================================
# Validation Components (Phase 3)
# ============================================================================

validation: core
	@echo "$(YELLOW)Building Validation Components...$(NC)"
	$(GNATMAKE) -P validation.gpr -Xmode=$(MODE) -aO$(OBJ_DIR)
	@echo "$(GREEN)Validation components built$(NC)"

validation-prove: validation
	@echo "$(YELLOW)Running formal verification on Validation...$(NC)"
	$(GNATPROVE) -P validation.gpr --level=$(PROOF_LEVEL) --checks-as-errors
	@echo "$(GREEN)Validation verification complete$(NC)"

# ============================================================================
# Analysis Components (Phase 4)
# ============================================================================

analysis: core validation
	@echo "$(YELLOW)Building Analysis Components...$(NC)"
	$(GNATMAKE) -P analysis.gpr -Xmode=$(MODE) -aO$(OBJ_DIR)
	@echo "$(GREEN)Analysis components built$(NC)"

analysis-prove: analysis
	@echo "$(YELLOW)Running formal verification on Analysis...$(NC)"
	$(GNATPROVE) -P analysis.gpr --level=2 --checks-as-errors
	@echo "$(GREEN)Analysis verification complete$(NC)"

# ============================================================================
# Testing Components (Phase 5)
# ============================================================================

testing: core parsing validation
	@echo "$(YELLOW)Building Testing Components...$(NC)"
	$(GNATMAKE) -P testing.gpr -Xmode=$(MODE) -aO$(OBJ_DIR)
	@echo "$(GREEN)Testing components built$(NC)"

testing-prove: testing
	@echo "$(YELLOW)Running formal verification on Testing...$(NC)"
	$(GNATPROVE) -P testing.gpr --level=2 --checks-as-errors
	@echo "$(GREEN)Testing verification complete$(NC)"

# ============================================================================
# Utility Components (Phase 6)
# ============================================================================

utils: core
	@echo "$(YELLOW)Building Utility Components...$(NC)"
	$(GNATMAKE) -P utils.gpr -Xmode=$(MODE) -aO$(OBJ_DIR)
	@echo "$(GREEN)Utility components built$(NC)"

utils-prove: utils
	@echo "$(YELLOW)Running formal verification on Utils...$(NC)"
	$(GNATPROVE) -P utils.gpr --level=1 --checks-as-errors
	@echo "$(GREEN)Utils verification complete$(NC)"

# ============================================================================
# Full Verification
# ============================================================================

prove: core-prove parsing-prove validation-prove analysis-prove testing-prove utils-prove
	@echo "$(GREEN)All formal verification complete$(NC)"

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
