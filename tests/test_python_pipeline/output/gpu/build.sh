#!/bin/bash
# STUNIR CUDA Build Script
set -e

if command -v nvcc &> /dev/null; then
    echo "Compiling CUDA kernels..."
    nvcc -o module module.cu module_host.cpp
    echo "Generated: module"
else
    echo "Error: nvcc not found. Install CUDA toolkit."
    exit 1
fi
