#!/bin/bash
set -e

main() {
    echo "Compiling from Rust!"
    x = $((10 + 32))
    echo "$x"
    for ((i=0; i<3; i++)); do
        echo "Echo..."
    done
}

main
