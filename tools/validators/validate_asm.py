#!/usr/bin/env python3
"""STUNIR ASM Validator

Validates assembly code for STUNIR pipeline.
Supports x86 (Intel/AT&T), ARM, and WASM text format.

Issue: asm/validators/1083
"""

import os
import re
import sys
import json
from typing import Any, Dict, List, Optional, Tuple

try:
    from base import BaseValidator, ValidationResult, ValidationError, canonical_json, get_epoch
except ImportError:
    from .base import BaseValidator, ValidationResult, ValidationError, canonical_json, get_epoch


class ASMValidator(BaseValidator):
    """Validator for assembly code."""
    
    SCHEMA = "stunir.validator.asm.v1"
    VALIDATOR_TYPE = "asm"
    
    # Register patterns for different architectures
    X86_REGISTERS = r'\b(eax|ebx|ecx|edx|esi|edi|ebp|esp|rax|rbx|rcx|rdx|rsi|rdi|rbp|rsp|r8|r9|r10|r11|r12|r13|r14|r15|al|ah|bl|bh|cl|ch|dl|dh|ax|bx|cx|dx)\b'
    ARM_REGISTERS = r'\b(r[0-9]|r1[0-5]|sp|lr|pc|w[0-9]|w[12][0-9]|w30|x[0-9]|x[12][0-9]|x30)\b'
    
    # Instruction patterns
    X86_INSTRUCTIONS = r'\b(mov|push|pop|add|sub|mul|div|inc|dec|and|or|xor|not|shl|shr|cmp|test|jmp|je|jne|jg|jl|call|ret|nop|lea)\b'
    ARM_INSTRUCTIONS = r'\b(mov|add|sub|mul|ldr|str|push|pop|bl|bx|cmp|beq|bne|nop|ret|svc)\b'
    WASM_INSTRUCTIONS = r'\b(i32|i64|f32|f64|local|global|func|param|result|call|return|block|loop|br|if|else|end)\b'
    
    def __init__(self, arch: str = "x86", syntax: str = "intel", strict: bool = False):
        super().__init__(strict)
        self.arch = arch.lower()
        self.syntax = syntax.lower()
    
    def _detect_architecture(self, content: str) -> str:
        """Auto-detect assembly architecture from content."""
        # Check for WASM
        if re.search(r'\(module|\(func|\(param|i32\.|i64\.', content):
            return "wasm"
        
        # Check for ARM
        if re.search(self.ARM_REGISTERS, content, re.IGNORECASE):
            arm_count = len(re.findall(self.ARM_REGISTERS, content, re.IGNORECASE))
            x86_count = len(re.findall(self.X86_REGISTERS, content, re.IGNORECASE))
            if arm_count > x86_count:
                return "arm"
        
        return "x86"
    
    def _validate_x86(self, content: str) -> List[Tuple[str, str, int]]:
        """Validate x86 assembly."""
        issues = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith(';') or stripped.startswith('#'):
                continue
            
            # Check for valid section directives
            if stripped.startswith('.'):
                if not re.match(r'\.(text|data|bss|section|global|extern|align)', stripped):
                    issues.append(("UNKNOWN_DIRECTIVE", f"Unknown directive: {stripped[:30]}", i))
            
            # Check for balanced brackets
            if stripped.count('[') != stripped.count(']'):
                issues.append(("UNBALANCED_BRACKETS", "Unbalanced square brackets", i))
            
            # Check for potentially unsafe instructions
            if self.strict and re.search(r'\b(int\s+0x80|syscall|sysenter)\b', stripped):
                issues.append(("UNSAFE_SYSCALL", "Direct syscall detected", i))
        
        return issues
    
    def _validate_arm(self, content: str) -> List[Tuple[str, str, int]]:
        """Validate ARM assembly."""
        issues = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith(';') or stripped.startswith('@') or stripped.startswith('//'):
                continue
            
            # Check for stack alignment (ARM64 requires 16-byte)
            if re.search(r'\bsp\b.*#[0-9]+', stripped):
                match = re.search(r'#([0-9]+)', stripped)
                if match:
                    val = int(match.group(1))
                    if val % 16 != 0 and 'sub' in stripped.lower():
                        issues.append(("STACK_ALIGNMENT", f"Stack may not be 16-byte aligned: {val}", i))
        
        return issues
    
    def _validate_wasm(self, content: str) -> List[Tuple[str, str, int]]:
        """Validate WASM text format."""
        issues = []
        lines = content.split('\n')
        paren_depth = 0
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith(';;'):
                continue
            
            paren_depth += stripped.count('(') - stripped.count(')')
            
            # Check for negative depth (extra closing parens)
            if paren_depth < 0:
                issues.append(("UNBALANCED_PARENS", "Unbalanced parentheses (extra closing)", i))
                paren_depth = 0
        
        if paren_depth != 0:
            issues.append(("UNBALANCED_PARENS", f"Unbalanced parentheses at end (depth: {paren_depth})", len(lines)))
        
        return issues
    
    def validate(self, content: str, filepath: str = None) -> ValidationResult:
        """Validate assembly content."""
        self.errors = []
        self.warnings = []
        
        if not content.strip():
            self.add_warning("EMPTY_FILE", "Empty assembly file")
            return ValidationResult(True, [], self.warnings)
        
        # Auto-detect architecture if not specified
        arch = self._detect_architecture(content)
        
        # Validate based on architecture
        if arch == "wasm":
            issues = self._validate_wasm(content)
        elif arch == "arm":
            issues = self._validate_arm(content)
        else:
            issues = self._validate_x86(content)
        
        # Convert issues to errors/warnings
        for code, msg, line in issues:
            if self.strict:
                self.add_error(code, msg, line)
            else:
                self.add_warning(code, msg, line)
        
        valid = len(self.errors) == 0
        return ValidationResult(valid, self.errors, self.warnings)


def main():
    """CLI entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="STUNIR ASM Validator")
    parser.add_argument("file", nargs="?", help="Assembly file to validate")
    parser.add_argument("--arch", choices=["x86", "arm", "wasm", "auto"], default="auto",
                        help="Target architecture")
    parser.add_argument("--syntax", choices=["intel", "att"], default="intel",
                        help="Assembly syntax (x86 only)")
    parser.add_argument("--strict", action="store_true", help="Enable strict mode")
    parser.add_argument("--json", action="store_true", help="Output JSON receipt")
    parser.add_argument("--self-test", action="store_true", help="Run self-test")
    args = parser.parse_args()
    
    if args.self_test:
        # Self-test with sample assembly
        test_x86 = """section .text\nglobal _start\n_start:\n    mov eax, 1\n    ret"""
        test_arm = """    .text\n    .global main\nmain:\n    mov r0, #0\n    bx lr"""
        test_wasm = """(module\n  (func $add (param i32 i32) (result i32)\n    local.get 0\n    local.get 1\n    i32.add))"""
        
        validator = ASMValidator(strict=False)
        
        for name, code in [("x86", test_x86), ("arm", test_arm), ("wasm", test_wasm)]:
            result = validator.validate(code)
            print(f"[{name}] Valid: {result.valid}, Errors: {len(result.errors)}, Warnings: {len(result.warnings)}")
        
        print("Self-test passed!")
        return 0
    
    if not args.file:
        parser.print_help()
        return 1
    
    validator = ASMValidator(arch=args.arch, syntax=args.syntax, strict=args.strict)
    result = validator.validate_file(args.file)
    
    if args.json:
        receipt = validator.generate_receipt(args.file, result)
        print(canonical_json(receipt))
    else:
        status = "✅ VALID" if result.valid else "❌ INVALID"
        print(f"{status}: {args.file}")
        for err in result.errors:
            loc = f":{err.line}" if err.line else ""
            print(f"  ERROR [{err.code}] {err.message}{loc}")
        for warn in result.warnings:
            loc = f":{warn.line}" if warn.line else ""
            print(f"  WARN  [{warn.code}] {warn.message}{loc}")
    
    return 0 if result.valid else 1


if __name__ == "__main__":
    sys.exit(main())
