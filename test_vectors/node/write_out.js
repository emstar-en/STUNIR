#!/usr/bin/env node
const fs = require('fs');
const out = process.argv[2] || 'out.txt';
fs.writeFileSync(out, 'hello
', { encoding: 'utf8' });
