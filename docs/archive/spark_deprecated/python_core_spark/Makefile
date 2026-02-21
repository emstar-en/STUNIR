# STUNIR Core SPARK Library Makefile
# Part of Phase 1 SPARK Migration

.PHONY: all build prove test clean

# Directories
OBJ_DIR := obj
LIB_DIR := lib
TEST_DIR := tests

# GNAT tools
GPRBUILD := gprbuild
GNATPROVE := gnatprove
GNATCLEAN := gnatclean

# Project file
PROJECT := stunir_core.gpr

# Default target
all: build

# Create output directories
$(OBJ_DIR):
	@mkdir -p $(OBJ_DIR)

$(LIB_DIR):
	@mkdir -p $(LIB_DIR)

# Build the library
build: $(OBJ_DIR) $(LIB_DIR)
	@echo "Building STUNIR Core SPARK library..."
	$(GPRBUILD) -P $(PROJECT) -j0
	@echo "Build complete."

# Run SPARK proofs
prove:
	@echo "Running SPARK proofs..."
	$(GNATPROVE) -P $(PROJECT) -j0 --level=2 --report=all
	@echo "Proofs complete."

# Run unit tests
test: build
	@echo "Running tests..."
	@if [ -f $(TEST_DIR)/run_tests.sh ]; then \
		bash $(TEST_DIR)/run_tests.sh; \
	else \
		echo "No tests configured yet."; \
	fi

# Run proofs with higher effort
prove-hard:
	@echo "Running SPARK proofs with maximum effort..."
	$(GNATPROVE) -P $(PROJECT) -j0 --level=4 --timeout=120 --report=all
	@echo "Proofs complete."

# Generate SPARK report
report:
	@echo "Generating SPARK report..."
	$(GNATPROVE) -P $(PROJECT) --report=statistics

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	$(GNATCLEAN) -P $(PROJECT)
	@rm -rf $(OBJ_DIR) $(LIB_DIR)
	@rm -rf gnatprove
	@echo "Clean complete."

# Show help
help:
	@echo "STUNIR Core SPARK Library"
	@echo ""
	@echo "Targets:"
	@echo "  all        - Build the library (default)"
	@echo "  build      - Build the library"
	@echo "  prove      - Run SPARK proofs (level 2)"
	@echo "  prove-hard - Run SPARK proofs (level 4, higher timeout)"
	@echo "  test       - Run unit tests"
	@echo "  report     - Generate SPARK statistics report"
	@echo "  clean      - Remove build artifacts"
	@echo "  help       - Show this help message"
