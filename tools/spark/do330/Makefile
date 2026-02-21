# STUNIR DO-330 Tool Qualification Framework
# Build Automation
# Copyright (C) 2026 STUNIR Project
# SPDX-License-Identifier: Apache-2.0

.PHONY: all build release debug prove test clean help

# Default target
all: build test

# Build targets
build:
	@echo "Building DO-330 framework (debug mode)..."
	gprbuild -P do330.gpr -XMODE=debug -j0
	@echo "Build complete."

release:
	@echo "Building DO-330 framework (release mode)..."
	gprbuild -P do330.gpr -XMODE=release -j0
	@echo "Release build complete."

debug:
	@echo "Building DO-330 framework (debug mode)..."
	gprbuild -P do330.gpr -XMODE=debug -j0
	@echo "Debug build complete."

prove:
	@echo "Running SPARK proofs..."
	gnatprove -P do330.gpr --level=2 --warnings=continue
	@echo "Proofs complete."

# Test targets
test: build
	@echo "Running DO-330 tests..."
	@echo "=== Template Tests ==="
	./bin/test_templates
	@echo ""
	@echo "=== Collector Tests ==="
	./bin/test_collector
	@echo ""
	@echo "=== Generator Tests ==="
	./bin/test_generator
	@echo ""
	@echo "All tests passed!"

test-templates: build
	@echo "Running template tests..."
	./bin/test_templates

test-collector: build
	@echo "Running collector tests..."
	./bin/test_collector

test-generator: build
	@echo "Running generator tests..."
	./bin/test_generator

# Example generation
example: build
	@echo "Generating example certification package..."
	mkdir -p examples/sample_package
	./bin/do330_generator --tool=verify_build --tql=4 --output=examples/sample_package
	@echo "Example package generated in examples/sample_package/"

# Clean targets
clean:
	@echo "Cleaning build artifacts..."
	gprclean -P do330.gpr 2>/dev/null || true
	rm -rf obj/* bin/* proof/sessions
	@echo "Clean complete."

clean-all: clean
	rm -rf examples/sample_package/*

# Help
help:
	@echo "STUNIR DO-330 Tool Qualification Framework"
	@echo ""
	@echo "Targets:"
	@echo "  all       - Build and run tests (default)"
	@echo "  build     - Build debug version"
	@echo "  release   - Build release version"
	@echo "  debug     - Build debug version"
	@echo "  prove     - Run SPARK proofs"
	@echo "  test      - Run all tests"
	@echo "  example   - Generate example package"
	@echo "  clean     - Clean build artifacts"
	@echo "  help      - Show this help"
	@echo ""
	@echo "Usage:"
	@echo "  make build    # Build the framework"
	@echo "  make test     # Run all tests"
	@echo "  make prove    # Run SPARK proofs"
	@echo "  make example  # Generate example package"
