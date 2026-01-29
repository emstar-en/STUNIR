#!/usr/bin/env python3
"""STUNIR FPGA Emitter - Emit Verilog/VHDL hardware description.

This tool is part of the targets → fpga pipeline stage.
It converts STUNIR IR to hardware description languages.

Usage:
    emitter.py <ir.json> --output=<dir> [--lang=verilog|vhdl]
    emitter.py --help
"""

import json
import hashlib
import time
import sys
from pathlib import Path


def canonical_json(data):
    """Generate RFC 8785 / JCS subset canonical JSON."""
    return json.dumps(data, sort_keys=True, separators=(',', ':'))


def compute_sha256(content):
    """Compute SHA256 hash of content."""
    if isinstance(content, str):
        content = content.encode('utf-8')
    return hashlib.sha256(content).hexdigest()


class FpgaEmitter:
    """Emitter for FPGA hardware description (Verilog/VHDL)."""
    
    # Bit widths for types
    TYPE_WIDTHS = {
        'i32': 32, 'i64': 64, 'f32': 32, 'f64': 64,
        'int': 32, 'long': 64, 'float': 32, 'double': 64,
        'bool': 1, 'byte': 8, 'void': 0
    }
    
    def __init__(self, ir_data, out_dir, options=None):
        """Initialize FPGA emitter."""
        self.ir_data = ir_data
        self.out_dir = Path(out_dir)
        self.options = options or {}
        self.lang = options.get('lang', 'verilog') if options else 'verilog'
        self.generated_files = []
        self.epoch = int(time.time())
    
    def _write_file(self, path, content):
        """Write content to file."""
        full_path = self.out_dir / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding='utf-8', newline='\n')
        self.generated_files.append({
            'path': str(path),
            'sha256': compute_sha256(content),
            'size': len(content.encode('utf-8'))
        })
        return full_path
    
    def _get_width(self, ir_type):
        """Get bit width for IR type."""
        return self.TYPE_WIDTHS.get(ir_type, 32)
    
    def _emit_module_verilog(self, func):
        """Emit Verilog module for function."""
        name = func.get('name', 'module0')
        params = func.get('params', [])
        returns = func.get('returns', 'i32')
        body = func.get('body', [])
        
        lines = [
            f'// STUNIR Generated Verilog Module: {name}',
            f'// Epoch: {self.epoch}',
            '',
            f'module {name} (',
            '    input wire clk,',
            '    input wire rst_n,',
            '    input wire start,',
            '    output reg done,'
        ]
        
        # Input ports
        for i, p in enumerate(params):
            p_name = p.get('name', f'arg{i}') if isinstance(p, dict) else f'arg{i}'
            p_type = p.get('type', 'i32') if isinstance(p, dict) else 'i32'
            width = self._get_width(p_type)
            lines.append(f'    input wire [{width-1}:0] {p_name},')
        
        # Output port
        ret_width = self._get_width(returns)
        lines.append(f'    output reg [{ret_width-1}:0] result')
        lines.append(');')
        lines.append('')
        
        # State machine states
        lines.append('    // State machine')
        lines.append('    localparam IDLE = 2\'b00;')
        lines.append('    localparam EXEC = 2\'b01;')
        lines.append('    localparam DONE_STATE = 2\'b10;')
        lines.append('    reg [1:0] state;')
        lines.append('')
        
        # Local variables
        var_decls = [s for s in body if isinstance(s, dict) and s.get('type') == 'var_decl']
        for v in var_decls:
            v_name = v.get('var_name', 'v0')
            v_type = v.get('var_type', 'i32')
            width = self._get_width(v_type)
            lines.append(f'    reg [{width-1}:0] {v_name};')
        
        lines.append('')
        lines.append('    // Sequential logic')
        lines.append('    always @(posedge clk or negedge rst_n) begin')
        lines.append('        if (!rst_n) begin')
        lines.append('            state <= IDLE;')
        lines.append('            done <= 1\'b0;')
        lines.append(f'            result <= {ret_width}\'d0;')
        lines.append('        end else begin')
        lines.append('            case (state)')
        lines.append('                IDLE: begin')
        lines.append('                    if (start) begin')
        lines.append('                        state <= EXEC;')
        lines.append('                        done <= 1\'b0;')
        lines.append('                    end')
        lines.append('                end')
        lines.append('                EXEC: begin')
        lines.append('                    // Computation')
        
        # Emit computation logic
        for stmt in body:
            if isinstance(stmt, dict):
                stmt_type = stmt.get('type', '')
                if stmt_type == 'return':
                    value = stmt.get('value', '0')
                    lines.append(f'                    result <= {value};')
                elif stmt_type == 'assign':
                    lines.append(f'                    {stmt.get("target")} <= {stmt.get("value")};')
                elif stmt_type == 'add':
                    lines.append(f'                    {stmt.get("dest")} <= {stmt.get("left")} + {stmt.get("right")};')
                elif stmt_type == 'sub':
                    lines.append(f'                    {stmt.get("dest")} <= {stmt.get("left")} - {stmt.get("right")};')
                elif stmt_type == 'mul':
                    lines.append(f'                    {stmt.get("dest")} <= {stmt.get("left")} * {stmt.get("right")};')
        
        lines.append('                    state <= DONE_STATE;')
        lines.append('                end')
        lines.append('                DONE_STATE: begin')
        lines.append('                    done <= 1\'b1;')
        lines.append('                    state <= IDLE;')
        lines.append('                end')
        lines.append('            endcase')
        lines.append('        end')
        lines.append('    end')
        lines.append('')
        lines.append('endmodule')
        
        return '\n'.join(lines)
    
    def _emit_entity_vhdl(self, func):
        """Emit VHDL entity for function."""
        name = func.get('name', 'module0')
        params = func.get('params', [])
        returns = func.get('returns', 'i32')
        
        ret_width = self._get_width(returns)
        
        lines = [
            f'-- STUNIR Generated VHDL Entity: {name}',
            f'-- Epoch: {self.epoch}',
            '',
            'library IEEE;',
            'use IEEE.STD_LOGIC_1164.ALL;',
            'use IEEE.NUMERIC_STD.ALL;',
            '',
            f'entity {name} is',
            '    port (',
            '        clk     : in  std_logic;',
            '        rst_n   : in  std_logic;',
            '        start   : in  std_logic;',
            '        done    : out std_logic;'
        ]
        
        for i, p in enumerate(params):
            p_name = p.get('name', f'arg{i}') if isinstance(p, dict) else f'arg{i}'
            p_type = p.get('type', 'i32') if isinstance(p, dict) else 'i32'
            width = self._get_width(p_type)
            lines.append(f'        {p_name} : in  std_logic_vector({width-1} downto 0);')
        
        lines.append(f'        result  : out std_logic_vector({ret_width-1} downto 0)')
        lines.append('    );')
        lines.append(f'end {name};')
        lines.append('')
        lines.append(f'architecture behavioral of {name} is')
        lines.append('    type state_type is (IDLE, EXEC, DONE_STATE);')
        lines.append('    signal state : state_type;')
        lines.append('begin')
        lines.append('    process(clk, rst_n)')
        lines.append('    begin')
        lines.append('        if rst_n = \'0\' then')
        lines.append('            state <= IDLE;')
        lines.append('            done <= \'0\';')
        lines.append(f'            result <= (others => \'0\');')
        lines.append('        elsif rising_edge(clk) then')
        lines.append('            case state is')
        lines.append('                when IDLE =>')
        lines.append('                    if start = \'1\' then')
        lines.append('                        state <= EXEC;')
        lines.append('                    end if;')
        lines.append('                when EXEC =>')
        lines.append('                    -- Computation placeholder')
        lines.append('                    state <= DONE_STATE;')
        lines.append('                when DONE_STATE =>')
        lines.append('                    done <= \'1\';')
        lines.append('                    state <= IDLE;')
        lines.append('            end case;')
        lines.append('        end if;')
        lines.append('    end process;')
        lines.append(f'end behavioral;')
        
        return '\n'.join(lines)
    
    def emit(self):
        """Emit FPGA files."""
        module_name = self.ir_data.get('ir_module', self.ir_data.get('module', 'module'))
        functions = self.ir_data.get('ir_functions', self.ir_data.get('functions', []))
        
        if self.lang == 'verilog':
            ext = 'v'
            emit_func = self._emit_module_verilog
        else:
            ext = 'vhd'
            emit_func = self._emit_entity_vhdl
        
        # Emit each function as a module
        for func in functions:
            name = func.get('name', 'module0')
            content = emit_func(func)
            self._write_file(f'{name}.{ext}', content)
        
        # Top-level wrapper
        if functions:
            top_content = self._emit_top_module(module_name, functions)
            self._write_file(f'{module_name}_top.{ext}', top_content)
        
        # Constraints file
        constraints = self._emit_constraints(module_name)
        self._write_file(f'{module_name}.xdc', constraints)
        
        # Testbench
        tb_content = self._emit_testbench(module_name, functions)
        self._write_file(f'{module_name}_tb.{ext}', tb_content)
        
        # README
        self._write_file('README.md', self._emit_readme(module_name, len(functions)))
        
        return f"Generated {len(functions)} FPGA modules"
    
    def _emit_top_module(self, module_name, functions):
        """Emit top-level wrapper module."""
        if self.lang == 'verilog':
            return f"""// STUNIR FPGA Top Module: {module_name}
// Epoch: {self.epoch}

module {module_name}_top (
    input wire clk,
    input wire rst_n,
    input wire start,
    output wire done,
    output wire [31:0] result
);

    // Instantiate first function module
    {functions[0].get('name', 'func0')} u_{functions[0].get('name', 'func0')} (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .done(done),
        .result(result)
    );

endmodule
"""
        return "-- VHDL top module placeholder"
    
    def _emit_constraints(self, module_name):
        """Emit timing constraints file."""
        return f"""# STUNIR FPGA Constraints: {module_name}
# Epoch: {self.epoch}

# Clock constraint (100 MHz)
create_clock -period 10.000 -name clk [get_ports clk]

# Input delay constraints
set_input_delay -clock clk -max 2.0 [get_ports start]
set_input_delay -clock clk -min 0.5 [get_ports start]

# Output delay constraints
set_output_delay -clock clk -max 2.0 [get_ports done]
set_output_delay -clock clk -max 2.0 [get_ports result*]
"""
    
    def _emit_testbench(self, module_name, functions):
        """Emit testbench."""
        if self.lang == 'verilog':
            return f"""// STUNIR FPGA Testbench: {module_name}
`timescale 1ns/1ps

module {module_name}_tb;
    reg clk = 0;
    reg rst_n = 0;
    reg start = 0;
    wire done;
    wire [31:0] result;
    
    {module_name}_top uut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .done(done),
        .result(result)
    );
    
    always #5 clk = ~clk;  // 100 MHz
    
    initial begin
        $dumpfile("{module_name}.vcd");
        $dumpvars(0, {module_name}_tb);
        
        #20 rst_n = 1;
        #10 start = 1;
        #10 start = 0;
        
        wait(done);
        $display("Result: %d", result);
        #20 $finish;
    end
endmodule
"""
        return "-- VHDL testbench placeholder"
    
    def _emit_readme(self, module_name, func_count):
        """Generate README."""
        return f"""# {module_name} (FPGA)

Generated by STUNIR FPGA Emitter.

## Language

{self.lang.upper()}

## Files

- `*.{'v' if self.lang == 'verilog' else 'vhd'}` - HDL modules
- `{module_name}_top.{'v' if self.lang == 'verilog' else 'vhd'}` - Top module
- `{module_name}_tb.{'v' if self.lang == 'verilog' else 'vhd'}` - Testbench
- `{module_name}.xdc` - Timing constraints

## Simulation

```bash
# Icarus Verilog
iverilog -o {module_name}_tb.vvp {module_name}_tb.v {module_name}_top.v *.v
vvp {module_name}_tb.vvp
```

## Statistics

- Modules: {func_count}
- Clock: 100 MHz target
- Epoch: {self.epoch}

## Schema

stunir.fpga.{self.lang}.v1
"""
    
    def emit_manifest(self):
        """Generate target manifest."""
        return {
            'schema': f'stunir.target.fpga.{self.lang}.manifest.v1',
            'epoch': self.epoch,
            'language': self.lang,
            'files': sorted(self.generated_files, key=lambda f: f['path']),
            'file_count': len(self.generated_files)
        }
    
    def emit_receipt(self):
        """Generate target receipt."""
        manifest = self.emit_manifest()
        manifest_json = canonical_json(manifest)
        return {
            'schema': f'stunir.target.fpga.{self.lang}.receipt.v1',
            'epoch': self.epoch,
            'manifest_sha256': compute_sha256(manifest_json),
            'file_count': len(self.generated_files)
        }


def main():
    args = {'output': None, 'input': None, 'lang': 'verilog'}
    for arg in sys.argv[1:]:
        if arg.startswith('--output='):
            args['output'] = arg.split('=', 1)[1]
        elif arg.startswith('--lang='):
            args['lang'] = arg.split('=', 1)[1]
        elif arg == '--help':
            print(__doc__)
            sys.exit(0)
        elif not arg.startswith('--'):
            args['input'] = arg
    
    if not args['input']:
        print(f"Usage: {sys.argv[0]} <ir.json> --output=<dir>", file=sys.stderr)
        sys.exit(1)
    
    out_dir = args['output'] or 'fpga_output'
    
    try:
        with open(args['input'], 'r') as f:
            ir_data = json.load(f)
        
        emitter = FpgaEmitter(ir_data, out_dir, {'lang': args['lang']})
        emitter.emit()
        
        manifest = emitter.emit_manifest()
        manifest_path = Path(out_dir) / 'manifest.json'
        manifest_path.write_text(canonical_json(manifest), encoding='utf-8')
        
        print(f"FPGA modules emitted to {out_dir}/", file=sys.stderr)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
