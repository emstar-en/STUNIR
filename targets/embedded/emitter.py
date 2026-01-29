#!/usr/bin/env python3
"""STUNIR Embedded Emitter - Emit bare-metal C for embedded systems.

This tool is part of the targets → embedded pipeline stage.
It converts STUNIR IR to C89-compliant code for embedded systems.

Usage:
    emitter.py <ir.json> --output=<dir> [--arch=arm|avr|mips]
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


class EmbeddedEmitter:
    """Emitter for embedded C code (bare-metal, no stdlib)."""
    
    # Embedded-safe type mappings (fixed-width)
    TYPE_MAP = {
        'i32': 'int32_t', 'i64': 'int64_t', 'f32': 'float', 'f64': 'double',
        'int': 'int32_t', 'long': 'int64_t', 'float': 'float', 'double': 'double',
        'void': 'void', 'bool': 'uint8_t', 'byte': 'uint8_t', 'char': 'char',
        'u8': 'uint8_t', 'u16': 'uint16_t', 'u32': 'uint32_t', 'u64': 'uint64_t'
    }
    
    # Architecture-specific settings
    ARCH_CONFIGS = {
        'arm': {'word_size': 32, 'endian': 'little', 'align': 4},
        'avr': {'word_size': 8, 'endian': 'little', 'align': 1},
        'mips': {'word_size': 32, 'endian': 'big', 'align': 4},
        'riscv': {'word_size': 32, 'endian': 'little', 'align': 4},
    }
    
    def __init__(self, ir_data, out_dir, options=None):
        """Initialize embedded emitter."""
        self.ir_data = ir_data
        self.out_dir = Path(out_dir)
        self.options = options or {}
        self.arch = options.get('arch', 'arm') if options else 'arm'
        self.arch_config = self.ARCH_CONFIGS.get(self.arch, self.ARCH_CONFIGS['arm'])
        self.generated_files = []
        self.epoch = int(time.time())
        self.stack_size = options.get('stack_size', 1024) if options else 1024
        self.heap_size = options.get('heap_size', 0) if options else 0  # No heap by default
    
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
    
    def _map_type(self, ir_type):
        """Map IR type to embedded C type."""
        return self.TYPE_MAP.get(ir_type, 'int32_t')
    
    def _emit_statement(self, stmt, indent='    '):
        """Convert IR statement to C code."""
        if isinstance(stmt, dict):
            stmt_type = stmt.get('type', 'nop')
            if stmt_type == 'var_decl':
                c_type = self._map_type(stmt.get('var_type', 'i32'))
                var_name = stmt.get('var_name', 'v0')
                init = stmt.get('init', '0')
                return f'{indent}{c_type} {var_name} = {init};'
            elif stmt_type == 'return':
                value = stmt.get('value', '0')
                return f'{indent}return {value};'
            elif stmt_type == 'assign':
                return f'{indent}{stmt.get("target", "v0")} = {stmt.get("value", "0")};'
            elif stmt_type in ('add', 'sub', 'mul', 'div'):
                ops = {'add': '+', 'sub': '-', 'mul': '*', 'div': '/'}
                op = ops.get(stmt_type, '+')
                return f'{indent}{stmt.get("dest", "v0")} = {stmt.get("left", "0")} {op} {stmt.get("right", "0")};'
            elif stmt_type == 'call':
                func = stmt.get('func', 'noop')
                args = ', '.join(stmt.get('args', []))
                return f'{indent}{func}({args});'
            else:
                return f'{indent}/* {stmt_type}: not implemented */'
        return f'{indent}/* nop */'
    
    def _emit_function(self, func):
        """Emit a C function."""
        name = func.get('name', 'func0')
        params = func.get('params', [])
        returns = func.get('returns', 'void')
        body = func.get('body', [])
        
        ret_type = self._map_type(returns)
        param_str = ', '.join([
            f"{self._map_type(p.get('type', 'i32'))} {p.get('name', f'arg{i}')}"
            if isinstance(p, dict) else f'int32_t arg{i}'
            for i, p in enumerate(params)
        ]) or 'void'
        
        lines = [
            f'/* Function: {name} */',
            f'{ret_type} {name}({param_str}) {{'
        ]
        
        for stmt in body:
            lines.append(self._emit_statement(stmt))
        
        if returns == 'void' or not any(isinstance(s, dict) and s.get('type') == 'return' for s in body):
            if returns != 'void':
                lines.append('    return 0;')
        
        lines.append('}')
        return '\n'.join(lines)
    
    def emit(self):
        """Emit embedded C files."""
        module_name = self.ir_data.get('ir_module', self.ir_data.get('module', 'module'))
        functions = self.ir_data.get('ir_functions', self.ir_data.get('functions', []))
        
        # Header file
        header = self._emit_header(module_name, functions)
        self._write_file(f'{module_name}.h', header)
        
        # Source file
        source = self._emit_source(module_name, functions)
        self._write_file(f'{module_name}.c', source)
        
        # Startup code
        startup = self._emit_startup(module_name)
        self._write_file('startup.c', startup)
        
        # Linker script
        linker = self._emit_linker_script(module_name)
        self._write_file(f'{module_name}.ld', linker)
        
        # Makefile
        makefile = self._emit_makefile(module_name)
        self._write_file('Makefile', makefile)
        
        # Config header
        config = self._emit_config_header(module_name)
        self._write_file('config.h', config)
        
        # README
        self._write_file('README.md', self._emit_readme(module_name, len(functions)))
        
        return source
    
    def _emit_header(self, module_name, functions):
        """Generate header file."""
        guard = f'{module_name.upper()}_H'
        lines = [
            f'/* STUNIR Embedded Module: {module_name} */',
            f'/* Architecture: {self.arch} */',
            f'/* Epoch: {self.epoch} */',
            '',
            f'#ifndef {guard}',
            f'#define {guard}',
            '',
            '#include <stdint.h>',
            '',
            '/* Function prototypes */',
        ]
        
        for func in functions:
            name = func.get('name', 'func0')
            params = func.get('params', [])
            returns = func.get('returns', 'void')
            ret_type = self._map_type(returns)
            param_str = ', '.join([
                f"{self._map_type(p.get('type', 'i32'))} {p.get('name', f'arg{i}')}"
                if isinstance(p, dict) else f'int32_t arg{i}'
                for i, p in enumerate(params)
            ]) or 'void'
            lines.append(f'{ret_type} {name}({param_str});')
        
        lines.extend([
            '',
            f'#endif /* {guard} */'
        ])
        return '\n'.join(lines)
    
    def _emit_source(self, module_name, functions):
        """Generate source file."""
        lines = [
            f'/* STUNIR Embedded Module: {module_name} */',
            f'/* Schema: stunir.embedded.{self.arch}.v1 */',
            f'/* Epoch: {self.epoch} */',
            '',
            f'#include "{module_name}.h"',
            '#include "config.h"',
            '',
            '/* No dynamic memory allocation */',
            '/* All variables are stack-allocated or static */',
            '',
        ]
        
        for func in functions:
            lines.append(self._emit_function(func))
            lines.append('')
        
        return '\n'.join(lines)
    
    def _emit_startup(self, module_name):
        """Generate startup code."""
        return f"""/* STUNIR Embedded Startup: {module_name} */
/* Architecture: {self.arch} */

#include <stdint.h>
#include "{module_name}.h"

/* Stack and heap configuration */
extern uint32_t _estack;
extern uint32_t _sdata, _edata, _sidata;
extern uint32_t _sbss, _ebss;

/* Reset handler */
void Reset_Handler(void) {{
    uint32_t *src, *dst;
    
    /* Copy .data section from Flash to RAM */
    src = &_sidata;
    dst = &_sdata;
    while (dst < &_edata) {{
        *dst++ = *src++;
    }}
    
    /* Zero .bss section */
    dst = &_sbss;
    while (dst < &_ebss) {{
        *dst++ = 0;
    }}
    
    /* Call main (or first function) */
    main();
    
    /* Infinite loop if main returns */
    while (1) {{}}
}}

/* Default interrupt handler */
void Default_Handler(void) {{
    while (1) {{}}
}}

/* Weak aliases for interrupt handlers */
void NMI_Handler(void) __attribute__((weak, alias("Default_Handler")));
void HardFault_Handler(void) __attribute__((weak, alias("Default_Handler")));
"""
    
    def _emit_linker_script(self, module_name):
        """Generate linker script."""
        return f"""/* STUNIR Embedded Linker Script: {module_name} */
/* Architecture: {self.arch} */
/* Stack: {self.stack_size} bytes */

MEMORY
{{
    FLASH (rx)  : ORIGIN = 0x08000000, LENGTH = 64K
    RAM (rwx)   : ORIGIN = 0x20000000, LENGTH = 16K
}}

_estack = ORIGIN(RAM) + LENGTH(RAM);

SECTIONS
{{
    .text :
    {{
        . = ALIGN(4);
        *(.isr_vector)
        *(.text)
        *(.text.*)
        *(.rodata)
        *(.rodata.*)
        . = ALIGN(4);
        _etext = .;
    }} > FLASH

    _sidata = LOADADDR(.data);

    .data :
    {{
        . = ALIGN(4);
        _sdata = .;
        *(.data)
        *(.data.*)
        . = ALIGN(4);
        _edata = .;
    }} > RAM AT > FLASH

    .bss :
    {{
        . = ALIGN(4);
        _sbss = .;
        *(.bss)
        *(.bss.*)
        *(COMMON)
        . = ALIGN(4);
        _ebss = .;
    }} > RAM

    ._stack :
    {{
        . = ALIGN(8);
        . = . + {self.stack_size};
        . = ALIGN(8);
    }} > RAM
}}
"""
    
    def _emit_makefile(self, module_name):
        """Generate Makefile."""
        toolchain = {
            'arm': 'arm-none-eabi-',
            'avr': 'avr-',
            'mips': 'mips-elf-',
            'riscv': 'riscv32-unknown-elf-'
        }.get(self.arch, 'arm-none-eabi-')
        
        return f"""# STUNIR Embedded Makefile: {module_name}
# Architecture: {self.arch}

PREFIX = {toolchain}
CC = $(PREFIX)gcc
OBJCOPY = $(PREFIX)objcopy
SIZE = $(PREFIX)size

CFLAGS = -mcpu=cortex-m3 -mthumb -Os -Wall -fno-common
CFLAGS += -ffunction-sections -fdata-sections
CFLAGS += -DSTUNIR_EMBEDDED -I.

LDFLAGS = -T{module_name}.ld -nostartfiles -Wl,--gc-sections

SRCS = {module_name}.c startup.c
OBJS = $(SRCS:.c=.o)
TARGET = {module_name}

.PHONY: all clean size

all: $(TARGET).elf $(TARGET).bin $(TARGET).hex size

$(TARGET).elf: $(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

$(TARGET).bin: $(TARGET).elf
	$(OBJCOPY) -O binary $< $@

$(TARGET).hex: $(TARGET).elf
	$(OBJCOPY) -O ihex $< $@

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

size: $(TARGET).elf
	$(SIZE) $<

clean:
	rm -f $(OBJS) $(TARGET).elf $(TARGET).bin $(TARGET).hex
"""
    
    def _emit_config_header(self, module_name):
        """Generate configuration header."""
        return f"""/* STUNIR Embedded Configuration: {module_name} */
/* Generated by STUNIR Embedded Emitter */

#ifndef CONFIG_H
#define CONFIG_H

/* Architecture */
#define STUNIR_ARCH_{self.arch.upper()} 1
#define STUNIR_WORD_SIZE {self.arch_config['word_size']}
#define STUNIR_ENDIAN_{self.arch_config['endian'].upper()} 1

/* Memory configuration */
#define STUNIR_STACK_SIZE {self.stack_size}
#define STUNIR_HEAP_SIZE {self.heap_size}

/* Feature flags */
#define STUNIR_NO_MALLOC 1
#define STUNIR_NO_STDIO 1
#define STUNIR_NO_FLOAT {'0' if self.arch_config['word_size'] >= 32 else '1'}

/* Epoch */
#define STUNIR_BUILD_EPOCH {self.epoch}

#endif /* CONFIG_H */
"""
    
    def _emit_readme(self, module_name, func_count):
        """Generate README."""
        return f"""# {module_name} (Embedded)

Generated by STUNIR Embedded Emitter.

## Architecture

{self.arch.upper()}

## Files

- `{module_name}.h` - Header file
- `{module_name}.c` - Implementation
- `startup.c` - Startup/reset handler
- `{module_name}.ld` - Linker script
- `Makefile` - Build system
- `config.h` - Configuration

## Build

```bash
make
```

Requires `{{'arm': 'arm-none-eabi', 'avr': 'avr', 'mips': 'mips-elf'}.get(self.arch, 'arm-none-eabi')}` toolchain.

## Features

- No dynamic memory allocation
- C89 compliant (with stdint.h)
- Bare-metal execution
- Configurable stack size: {self.stack_size} bytes

## Statistics

- Functions: {func_count}
- Stack: {self.stack_size} bytes
- Heap: {self.heap_size} bytes
- Epoch: {self.epoch}

## Schema

stunir.embedded.{self.arch}.v1
"""
    
    def emit_manifest(self):
        """Generate target manifest."""
        return {
            'schema': f'stunir.target.embedded.{self.arch}.manifest.v1',
            'epoch': self.epoch,
            'arch': self.arch,
            'files': sorted(self.generated_files, key=lambda f: f['path']),
            'file_count': len(self.generated_files)
        }
    
    def emit_receipt(self):
        """Generate target receipt."""
        manifest = self.emit_manifest()
        manifest_json = canonical_json(manifest)
        return {
            'schema': f'stunir.target.embedded.{self.arch}.receipt.v1',
            'epoch': self.epoch,
            'manifest_sha256': compute_sha256(manifest_json),
            'file_count': len(self.generated_files)
        }


def main():
    args = {'output': None, 'input': None, 'arch': 'arm'}
    for arg in sys.argv[1:]:
        if arg.startswith('--output='):
            args['output'] = arg.split('=', 1)[1]
        elif arg.startswith('--arch='):
            args['arch'] = arg.split('=', 1)[1]
        elif arg == '--help':
            print(__doc__)
            sys.exit(0)
        elif not arg.startswith('--'):
            args['input'] = arg
    
    if not args['input']:
        print(f"Usage: {sys.argv[0]} <ir.json> --output=<dir>", file=sys.stderr)
        sys.exit(1)
    
    out_dir = args['output'] or 'embedded_output'
    
    try:
        with open(args['input'], 'r') as f:
            ir_data = json.load(f)
        
        emitter = EmbeddedEmitter(ir_data, out_dir, {'arch': args['arch']})
        emitter.emit()
        
        manifest = emitter.emit_manifest()
        manifest_path = Path(out_dir) / 'manifest.json'
        manifest_path.write_text(canonical_json(manifest), encoding='utf-8')
        
        print(f"Embedded code emitted to {out_dir}/", file=sys.stderr)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
