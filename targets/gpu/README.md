# STUNIR GPU Target

GPU compute kernel emitter for STUNIR (CUDA/OpenCL).

## Overview

This emitter converts STUNIR IR to GPU compute kernels supporting both
CUDA and OpenCL backends.

## Usage

```bash
python emitter.py <ir.json> --output=<output_dir> [--backend=cuda|opencl]
```

## Output Files

- `<module>.cu` / `<module>.cl` - Kernel source
- `<module>_host.cpp` - Host wrapper
- `build.sh` - Compilation script
- `manifest.json` - Deterministic file manifest
- `README.md` - Generated documentation

## Features

- CUDA kernel generation with thread indexing
- OpenCL kernel generation with work-item handling
- Host-side wrapper code
- Grid/block dimension templates

## Dependencies

- CUDA Toolkit (for CUDA backend)
- OpenCL SDK (for OpenCL backend)

## Schema

`stunir.gpu.cuda.v1` / `stunir.gpu.opencl.v1`
