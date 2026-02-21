# STUNIR DO-332 Build Makefile
# Copyright (C) 2026 STUNIR Project
# SPDX-License-Identifier: Apache-2.0

.PHONY: all build debug release prove clean test install help

# Configuration
GPRBUILD ?= gprbuild
GNATPROVE ?= gnatprove
GPRCLEAN ?= gprclean

PROJECT = do332.gpr
BIN_DIR = bin
OBJ_DIR = obj
PROOF_DIR = proof

# Default target
all: build

# Build in debug mode
build: debug

debug:
	@echo "Building DO-332 tools in debug mode..."
	$(GPRBUILD) -P $(PROJECT) -XMODE=debug
	@echo "Build complete: $(BIN_DIR)/do332_analyzer"

# Build in release mode
release:
	@echo "Building DO-332 tools in release mode..."
	$(GPRBUILD) -P $(PROJECT) -XMODE=release
	@echo "Build complete: $(BIN_DIR)/do332_analyzer"

# Run SPARK proofs
prove:
	@echo "Running SPARK proofs..."
	@mkdir -p $(PROOF_DIR)
	$(GNATPROVE) -P $(PROJECT) -XMODE=prove --output-dir=$(PROOF_DIR) -j0
	@echo "Proof results in $(PROOF_DIR)/"

# Run proofs with report
prove-report:
	@echo "Running SPARK proofs with report..."
	@mkdir -p $(PROOF_DIR)
	$(GNATPROVE) -P $(PROJECT) -XMODE=prove --output-dir=$(PROOF_DIR) -j0 --report=all
	@echo "Proof report in $(PROOF_DIR)/gnatprove.out"

# Run tests
test: debug
	@echo "Running DO-332 tests..."
	@if [ -f $(BIN_DIR)/do332_analyzer ]; then \
		$(BIN_DIR)/do332_analyzer --help; \
		echo "Basic test passed."; \
	else \
		echo "Error: do332_analyzer not found"; \
		exit 1; \
	fi

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	$(GPRCLEAN) -P $(PROJECT) -XMODE=debug 2>/dev/null || true
	$(GPRCLEAN) -P $(PROJECT) -XMODE=release 2>/dev/null || true
	@rm -rf $(OBJ_DIR)/* $(BIN_DIR)/* $(PROOF_DIR)/*
	@echo "Clean complete."

# Install to parent bin directory
install: release
	@echo "Installing do332_analyzer..."
	@mkdir -p ../../bin
	@cp $(BIN_DIR)/do332_analyzer ../../bin/
	@echo "Installed to ../../bin/do332_analyzer"

# Show help
help:
	@echo "STUNIR DO-332 Build System"
	@echo ""
	@echo "Targets:"
	@echo "  build/debug  - Build in debug mode (default)"
	@echo "  release      - Build in release mode"
	@echo "  prove        - Run SPARK proofs"
	@echo "  prove-report - Run SPARK proofs with detailed report"
	@echo "  test         - Run tests"
	@echo "  clean        - Remove build artifacts"
	@echo "  install      - Install to ../../bin/"
	@echo "  help         - Show this help"
	@echo ""
	@echo "Requirements:"
	@echo "  - GNAT compiler (gprbuild)"
	@echo "  - SPARK prover (gnatprove) for 'prove' target"
