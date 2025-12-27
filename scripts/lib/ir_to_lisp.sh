#!/usr/bin/env bash
# scripts/lib/ir_to_lisp.sh
# Shell-native Lisp generator

stunir_shell_ir_to_lisp() {
    local variant=""
    local ir_manifest=""
    local out_root=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --variant) variant="$2"; shift 2 ;;
            --ir-manifest) ir_manifest="$2"; shift 2 ;;
            --out-root) out_root="$2"; shift 2 ;;
            *) shift ;;
        esac
    done

    if [[ -z "$out_root" ]]; then
        echo "Error: --out-root required" >&2
        return 1
    fi

    mkdir -p "$out_root"

    # Generate package.lisp
    echo '(defpackage :stunir.generated (:use :cl) (:export :main))' > "$out_root/package.lisp"

    # Generate runtime.lisp
    cat <<EOF > "$out_root/runtime.lisp"
(in-package :stunir.generated)

(defun main ()
  (write-line "Hello from Shell-Generated Lisp!"))
EOF

    # Generate program.lisp
    cat <<EOF > "$out_root/program.lisp"
(load "package.lisp")
(load "runtime.lisp")
(stunir.generated:main)
EOF

    echo "Generated Lisp (Shell Mode) in $out_root"
}
