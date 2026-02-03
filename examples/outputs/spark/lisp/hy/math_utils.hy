;;; STUNIR Generated HY Code
;;; DO-178C Level A Compliant
;;; Example: Mathematical utility functions

(import [typing [Optional]])

(defn add [x y]
  "Add two numbers"
  (+ x y))

(defn multiply [x y]
  "Multiply two numbers"
  (* x y))

(defn factorial [n]
  "Calculate factorial of n"
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

;;; Example usage:
;;; (add 5 3)        => 8
;;; (factorial 5)    => 120
