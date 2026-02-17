# Determinism probe: write deterministic bytes to a requested path.
out = length(ARGS) >= 1 ? ARGS[1] : "out.txt"
open(out, "w") do io
    write(io, "hello
")
end
