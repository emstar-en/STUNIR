# Determinism probe: write deterministic bytes to a requested path.
args <- commandArgs(trailingOnly = TRUE)
out <- if (length(args) >= 1) args[[1]] else 'out.txt'
cat('hello
', file=out)
