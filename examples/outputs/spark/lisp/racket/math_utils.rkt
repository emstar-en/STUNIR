#lang racket/base
;;; STUNIR Generated RACKET Code
;;; DO-178C Level A Compliant
;;; Example: Mathematical utility functions

(require racket/contract)

(provide/contract
 [add (-> integer? integer? integer?)]
 [multiply (-> integer? integer? integer?)]
 [factorial (-> (integer-in 0 1000) integer?)])

(define (add x y)
  (+ x y))

(define (multiply x y)
  (* x y))

(define (factorial n)
  (if (zero? n)
      1
      (* n (factorial (sub1 n)))))

;;; Example usage:
;;; (add 5 3)        => 8
;;; (factorial 5)    => 120
