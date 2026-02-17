-- Determinism probe: write deterministic bytes to a requested path.
local out = arg[1] or 'out.txt'
local f = assert(io.open(out, 'wb'))
f:write('hello
')
f:close()
