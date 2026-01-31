;; STUNIR Generated SCHEME Code (R7RS)
;; DO-178C Level A Compliant
;; Example: Mathematical utility functions

(define-library (math-utils)
  (export add multiply factorial)
  (import (scheme base))
  (begin
    
    (define (add x y)
      "Add two numbers"
      (+ x y))
    
    (define (multiply x y)
      "Multiply two numbers"
      (* x y))
    
    (define (factorial n)
      "Calculate factorial of n"
      (if (zero? n)
          1
          (* n (factorial (- n 1)))))
    
    )) ; end define-library

;;; Example usage:
;;; (import (math-utils))
;;; (add 5 3)        => 8
;;; (factorial 5)    => 120
