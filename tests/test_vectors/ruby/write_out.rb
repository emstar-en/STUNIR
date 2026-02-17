# Determinism probe: write deterministic bytes to a requested path.
out = ARGV[0] || 'out.txt'
File.open(out, 'wb') { |f| f.write("hello
") }
