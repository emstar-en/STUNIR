;;; STUNIR Generated COMMON_LISP Code
;;; DO-178C Level A Compliant
;;; Example: Mathematical utility functions

(defpackage :math-utils
  (:use :cl)
  (:export #:add #:multiply #:factorial))

(in-package :math-utils)

(defun add (x y)
  "Add two numbers"
  (declare (type integer x y))
  (+ x y))

(defun multiply (x y)
  "Multiply two numbers"
  (declare (type integer x y))
  (* x y))

(defun factorial (n)
  "Calculate factorial of n"
  (declare (type (integer 0 *) n))
  (if (zerop n)
      1
      (* n (factorial (1- n)))))

;;; Example usage:
;;; (add 5 3)        => 8
;;; (multiply 4 7)   => 28
;;; (factorial 5)    => 120
