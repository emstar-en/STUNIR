# STUNIR DO-333 Formal Methods Support - Build Automation
# Copyright (C) 2026 STUNIR Project
# SPDX-License-Identifier: Apache-2.0

.PHONY: all build debug release prove clean test run help

GPRBUILD := gprbuild
GNATPROVE := gnatprove
GPRCLEAN := gprclean
PROJECT := do333.gpr

# Default target
all: build

# Build targets
build: debug

debug:
	@echo "Building DO-333 (debug mode)..."
	$(GPRBUILD) -P $(PROJECT) -XMODE=debug -j0

release:
	@echo "Building DO-333 (release mode)..."
	$(GPRBUILD) -P $(PROJECT) -XMODE=release -j0

prove:
	@echo "Running SPARK proofs..."
	$(GNATPROVE) -P $(PROJECT) -XMODE=prove --level=2 -j0

prove-all:
	@echo "Running SPARK proofs (maximum effort)..."
	$(GNATPROVE) -P $(PROJECT) -XMODE=prove --level=4 -j0

flow:
	@echo "Running SPARK flow analysis..."
	$(GNATPROVE) -P $(PROJECT) -XMODE=prove --mode=flow -j0

# Test targets
test: debug
	@echo "Running tests..."
	./bin/test_formal_spec
	./bin/test_po_manager
	./bin/test_vc_tracker
	./bin/test_spark_integration
	./bin/test_evidence_generator
	./bin/test_integration
	@echo "All tests passed!"

# Run main program
run: debug
	./bin/do333_analyzer --help

# Clean targets
clean:
	@echo "Cleaning..."
	$(GPRCLEAN) -P $(PROJECT) -XMODE=debug 2>/dev/null || true
	$(GPRCLEAN) -P $(PROJECT) -XMODE=release 2>/dev/null || true
	rm -rf obj/* bin/* proof/sessions

# Help
help:
	@echo "DO-333 Formal Methods Support - Build Targets"
	@echo ""
	@echo "  make build     - Build in debug mode (default)"
	@echo "  make debug     - Build in debug mode"
	@echo "  make release   - Build in release mode"
	@echo "  make prove     - Run SPARK proofs (level 2)"
	@echo "  make prove-all - Run SPARK proofs (level 4, max effort)"
	@echo "  make flow      - Run flow analysis only"
	@echo "  make test      - Run all tests"
	@echo "  make run       - Run main analyzer"
	@echo "  make clean     - Clean build artifacts"
	@echo "  make help      - Show this help"
