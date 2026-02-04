#!/usr/bin/env node
/**
 * STUNIR Test Vector - Node.js File Writer
 *
 * Simple utility to write test output files for Node.js compatibility testing.
 * Part of the STUNIR test vectors for verifying Node.js toolchain integration.
 *
 * @module write_out
 * @example
 *   node write_out.js output.txt
 */

const fs = require('fs');

/**
 * Output file path from command line arguments.
 * Defaults to 'out.txt' if not provided.
 * @type {string}
 */
const out = process.argv[2] || 'out.txt';

/**
 * Write test content to the specified file.
 *
 * @param {string} filepath - Path to the output file
 * @returns {void}
 */
function writeOutput(filepath) {
    fs.writeFileSync(filepath, 'hello\n', { encoding: 'utf8' });
}

// Execute the write operation
writeOutput(out);
