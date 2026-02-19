#!/usr/bin/env bash
# STUNIR WSL Bootstrap Script
# Purpose: Set up STUNIR development environment in WSL

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== STUNIR WSL Bootstrap ==="
echo "Setting up STUNIR development environment in WSL..."

# Check if running in WSL
if ! grep -qEi "(Microsoft|WSL)" /proc/version &> /dev/null; then
    echo "WARNING: This script is optimized for WSL but can run on any Linux system."
fi

# Function to check and install package
check_and_install() {
    local package=$1
    local binary=${2:-$1}
    
    if ! command -v "$binary" &> /dev/null; then
        echo "Installing $package..."
        sudo apt-get update -qq
        sudo apt-get install -y "$package"
    else
        echo "✓ $package is already installed"
    fi
}

# Update package list
echo ""
echo "Updating package list..."
sudo apt-get update -qq

# Install essential tools
echo ""
echo "Checking required dependencies..."
check_and_install "python3" "python3"
check_and_install "python3-pip" "pip3"
check_and_install "git" "git"
check_and_install "build-essential" "gcc"
check_and_install "curl" "curl"
check_and_install "wget" "wget"

# Check for optional but recommended tools
echo ""
echo "Checking optional dependencies..."

# Rust toolchain (optional)
if ! command -v rustc &> /dev/null; then
    echo "Rust not found. To install Rust, run:"
    echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
else
    echo "✓ Rust is installed: $(rustc --version)"
fi

# Haskell toolchain (optional)
if ! command -v ghc &> /dev/null; then
    echo "Haskell (GHC) not found. To install, run:"
    echo "  sudo apt-get install ghc cabal-install"
else
    echo "✓ GHC is installed: $(ghc --version)"
fi

# GNAT/Ada toolchain (optional)
if ! command -v gnatmake &> /dev/null; then
    echo "GNAT (Ada) not found. To install, run:"
    echo "  sudo apt-get install gnat gprbuild"
else
    echo "✓ GNAT is installed"
fi

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
if [ -f "pyproject.toml" ]; then
    pip3 install -e . --user
    echo "✓ Python package installed in development mode"
else
    echo "WARNING: pyproject.toml not found, skipping pip install"
fi

# Create necessary directories
echo ""
echo "Creating build directories..."
mkdir -p build bin obj test_output receipts

# Run toolchain discovery
echo ""
echo "Discovering local toolchain..."
if [ -f "scripts/discover_toolchain.sh" ]; then
    bash scripts/discover_toolchain.sh
    echo "✓ Toolchain lockfile generated"
else
    echo "WARNING: scripts/discover_toolchain.sh not found"
fi

# Make all shell scripts executable
echo ""
echo "Making shell scripts executable..."
find scripts -name "*.sh" -type f -exec chmod +x {} \;
find . -maxdepth 1 -name "*.sh" -type f -exec chmod +x {} \;
echo "✓ Shell scripts are now executable"

# Display precompiled binaries info
echo ""
echo "=== Precompiled Binaries ==="
if [ -d "precompiled/linux-x86_64" ]; then
    echo "✓ Linux precompiled binaries found in precompiled/linux-x86_64/"
    echo "  You can use these directly without building from source"
else
    echo "No precompiled binaries found for Linux x86_64"
fi

echo ""
echo "=== WSL Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Run tests: bash test_pipeline.sh"
echo "  2. Build tools: bash scripts/build.sh"
echo "  3. Explore examples: ls examples/"
echo ""
echo "For more information, see README.md"
