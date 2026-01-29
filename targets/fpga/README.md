# STUNIR FPGA Target

FPGA hardware description emitter for STUNIR (Verilog/VHDL).

## Overview

This emitter converts STUNIR IR to hardware description languages for
FPGA synthesis and implementation.

## Usage

```bash
python emitter.py <ir.json> --output=<output_dir> [--lang=verilog|vhdl]
```

## Output Files

- `<function>.v` / `.vhd` - Module/Entity per function
- `<module>_top.v` / `.vhd` - Top-level wrapper
- `<module>_tb.v` / `.vhd` - Testbench
- `<module>.xdc` - Timing constraints
- `manifest.json` - Deterministic file manifest
- `README.md` - Generated documentation

## Features

- Verilog module generation with state machines
- VHDL entity/architecture generation
- Timing constraints (XDC format)
- Testbench with VCD output

## Dependencies

- Synthesis tools (Vivado, Quartus, etc.)
- Simulation: Icarus Verilog, ModelSim, etc.

## Schema

`stunir.fpga.verilog.v1` / `stunir.fpga.vhdl.v1`
