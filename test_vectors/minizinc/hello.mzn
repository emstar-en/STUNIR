% Determinism probe for MiniZinc.
% A trivial model with a unique solution.
var 1..1: x;
solve satisfy;
% In json output-mode, MiniZinc emits a JSON-compatible representation.
output ["x=", show(x), "
"]; 
