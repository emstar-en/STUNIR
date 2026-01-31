# WebAssembly Examples

## WASM Module Example

**Target:** WebAssembly MVP + SIMD  
**Runtime:** Browser, Node.js, WASI

### Compilation

```bash
# Compile to WASM
clang --target=wasm32-wasi \
  -O2 -flto \
  -Wl,--export-all \
  -Wl,--no-entry \
  module.c -o module.wasm

# Optimize with wasm-opt
wasm-opt -O3 --enable-simd module.wasm -o module.opt.wasm
```

### Running

```javascript
// In browser
WebAssembly.instantiateStreaming(fetch('module.wasm'))
  .then(obj => {
    const result = obj.instance.exports.add(5, 7);
    console.log(result);  // 12
  });

// In Node.js with WASI
const fs = require('fs');
const { WASI } = require('wasi');
const wasi = new WASI();

const wasm = fs.readFileSync('module.wasm');
WebAssembly.instantiate(wasm, wasi.getImportObject())
  .then(obj => {
    wasi.start(obj.instance);
  });
```

### Features

- Portable across platforms
- Near-native performance
- Memory-safe by design
- SIMD acceleration
