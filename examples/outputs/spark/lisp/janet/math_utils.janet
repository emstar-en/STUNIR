# STUNIR Generated JANET Code
# DO-178C Level A Compliant
# Example: Mathematical utility functions

(defn add
  "Add two numbers"
  [x y]
  (+ x y))

(defn multiply
  "Multiply two numbers"
  [x y]
  (* x y))

(defn factorial
  "Calculate factorial of n"
  [n]
  (if (= n 0)
    1
    (* n (factorial (- n 1)))))

# Example usage:
# (add 5 3)        # => 8
# (factorial 5)    # => 120
