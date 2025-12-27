#!/bin/bash
set -e

main() {
    echo "hello"
    x=$(( 2 + 3 ))
    echo "$x"
    for ((i=0; i<2; i++)); do
        echo "loop"
    done
}
main
