;;; math-utils.el --- Mathematical utility functions  -*- lexical-binding: t; -*-

;;; STUNIR Generated EMACS_LISP Code
;;; DO-178C Level A Compliant

;;; Commentary:
;; Mathematical utility functions for Emacs

;;; Code:

(defun math-utils-add (x y)
  "Add two numbers X and Y."
  (+ x y))

(defun math-utils-multiply (x y)
  "Multiply two numbers X and Y."
  (* x y))

(defun math-utils-factorial (n)
  "Calculate factorial of N."
  (if (zerop n)
      1
    (* n (math-utils-factorial (1- n)))))

(provide 'math-utils)

;;; Example usage:
;;; (math-utils-add 5 3)        => 8
;;; (math-utils-factorial 5)    => 120

;;; math-utils.el ends here
