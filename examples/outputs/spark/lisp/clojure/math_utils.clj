;;; STUNIR Generated CLOJURE Code
;;; DO-178C Level A Compliant
;;; Example: Mathematical utility functions

(ns math-utils
  (:gen-class))

(defn add
  "Add two numbers"
  [^Integer x ^Integer y]
  (+ x y))

(defn multiply
  "Multiply two numbers"
  [^Integer x ^Integer y]
  (* x y))

(defn factorial
  "Calculate factorial of n"
  [^Integer n]
  (if (zero? n)
    1
    (* n (factorial (dec n)))))

(defrecord Point [x y])

(defn distance
  "Calculate distance between two points"
  [^Point p1 ^Point p2]
  (Math/sqrt (+ (Math/pow (- (:x p2) (:x p1)) 2)
                (Math/pow (- (:y p2) (:y p1)) 2))))

;;; Example usage:
;;; (add 5 3)                                    => 8
;;; (factorial 5)                                => 120
;;; (distance (->Point 0 0) (->Point 3 4))       => 5.0
