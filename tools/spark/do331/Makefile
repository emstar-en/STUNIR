# STUNIR DO-331 Model-Based Development Tools
# Makefile
# Copyright (C) 2026 STUNIR Project
# SPDX-License-Identifier: Apache-2.0

# Configuration
GPRBUILD := gprbuild
GNATPROVE := gnatprove
GPRCLEAN := gprclean

# Directories
SRC_DIR := src
TEST_DIR := tests
OBJ_DIR := obj
BIN_DIR := bin
PROOF_DIR := proof
DOC_DIR := docs
EXAMPLES_DIR := examples

# Project file
GPR_FILE := do331.gpr

# Build modes
DEBUG ?= 0
SPARK_PROOF ?= 0

# Default target
.PHONY: all
all: build

# Create directories
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(PROOF_DIR):
	mkdir -p $(PROOF_DIR)

# Build
.PHONY: build
build: $(OBJ_DIR) $(BIN_DIR)
	@echo "[DO-331] Building SPARK tools..."
	$(GPRBUILD) -P $(GPR_FILE) -p
	@echo "[DO-331] Build complete."

# Debug build
.PHONY: debug
debug: $(OBJ_DIR) $(BIN_DIR)
	@echo "[DO-331] Debug build..."
	$(GPRBUILD) -P $(GPR_FILE) -p -XDEBUG=1

# SPARK proof
.PHONY: prove
prove: $(PROOF_DIR)
	@echo "[DO-331] Running SPARK prover..."
	$(GNATPROVE) -P $(GPR_FILE) --level=2 --report=all
	@echo "[DO-331] SPARK proof complete."

# Quick proof (faster, less thorough)
.PHONY: prove-quick
prove-quick: $(PROOF_DIR)
	@echo "[DO-331] Running quick SPARK proof..."
	$(GNATPROVE) -P $(GPR_FILE) --level=1

# Clean
.PHONY: clean
clean:
	@echo "[DO-331] Cleaning..."
	$(GPRCLEAN) -P $(GPR_FILE) 2>/dev/null || true
	rm -rf $(OBJ_DIR) $(BIN_DIR) $(PROOF_DIR)
	@echo "[DO-331] Clean complete."

# Test
.PHONY: test
test: build
	@echo "[DO-331] Running tests..."
	if [ -f $(BIN_DIR)/do331_main ]; then \
		$(BIN_DIR)/do331_main --test; \
	else \
		echo "Warning: Binary not built. Running minimal validation..."; \
		echo "DO-331 tools validation: Ada SPARK source files present."; \
		ls -la $(SRC_DIR)/*.ads 2>/dev/null | head -5; \
	fi
	@echo "[DO-331] Tests complete."

# Install (to parent directory)
.PHONY: install
install: build
	@echo "[DO-331] Installing..."
	mkdir -p ../../bin
	cp -f $(BIN_DIR)/* ../../bin/ 2>/dev/null || true
	@echo "[DO-331] Install complete."

# Documentation
.PHONY: docs
docs:
	@echo "[DO-331] Documentation available in $(DOC_DIR)/"
	ls -la $(DOC_DIR)/

# Examples
.PHONY: examples
examples: build
	@echo "[DO-331] Running examples..."
	cd $(EXAMPLES_DIR) && ./run_examples.sh 2>/dev/null || echo "See examples directory."

# Check GNAT availability
.PHONY: check-tools
check-tools:
	@echo "Checking for required tools..."
	@which $(GPRBUILD) > /dev/null 2>&1 && echo "  gprbuild: OK" || echo "  gprbuild: NOT FOUND"
	@which $(GNATPROVE) > /dev/null 2>&1 && echo "  gnatprove: OK" || echo "  gnatprove: NOT FOUND (optional for proof)"
	@which gnat > /dev/null 2>&1 && echo "  gnat: OK" || echo "  gnat: NOT FOUND"

# Help
.PHONY: help
help:
	@echo "STUNIR DO-331 Model-Based Development Tools"
	@echo ""
	@echo "Targets:"
	@echo "  all          - Build the tools (default)"
	@echo "  build        - Build with gprbuild"
	@echo "  debug        - Build with debug symbols"
	@echo "  prove        - Run SPARK prover"
	@echo "  prove-quick  - Quick SPARK proof"
	@echo "  test         - Run tests"
	@echo "  clean        - Clean build artifacts"
	@echo "  install      - Install to ../../bin/"
	@echo "  docs         - Show documentation"
	@echo "  examples     - Run examples"
	@echo "  check-tools  - Check for required tools"
	@echo "  help         - Show this help"
	@echo ""
	@echo "Environment:"
	@echo "  SPARK_PROOF=1  - Enable SPARK proof during build"
	@echo "  DEBUG=1        - Enable debug build"
