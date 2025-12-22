<?php
// Determinism probe: write deterministic bytes to a requested path.
$out = $argv[1] ?? 'out.txt';
file_put_contents($out, "hello
");
?>
