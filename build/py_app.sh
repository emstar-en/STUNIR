#!/bin/bash
set -e

main() {
    echo "Hello World"
    a=10
    b=20
    c=$(( a + b ))
    echo "$c"
    for ((i=0; i<3; i++)); do
        echo "Looping..."
    done
}
main
