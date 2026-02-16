#!/usr/bin/env python3
"""STUNIR ISR (Interrupt Service Routine) Stub Generator

Generates interrupt handling stubs for x86/x86_64 systems.
Supports exceptions (0-31) and IRQs (32-255).

DO-178C Level A Compliant Generation
"""

from typing import Optional, List, Tuple
from dataclasses import dataclass, field
from enum import IntEnum

# Exceptions that push error codes
EXCEPTIONS_WITH_ERROR_CODE = {8, 10, 11, 12, 13, 14, 17, 21, 29, 30}

# Exception names for documentation
EXCEPTION_NAMES = {
    0: "Division Error",
    1: "Debug",
    2: "Non-Maskable Interrupt",
    3: "Breakpoint",
    4: "Overflow",
    5: "Bound Range Exceeded",
    6: "Invalid Opcode",
    7: "Device Not Available",
    8: "Double Fault",
    9: "Coprocessor Segment Overrun",
    10: "Invalid TSS",
    11: "Segment Not Present",
    12: "Stack-Segment Fault",
    13: "General Protection Fault",
    14: "Page Fault",
    15: "Reserved",
    16: "x87 Floating-Point Exception",
    17: "Alignment Check",
    18: "Machine Check",
    19: "SIMD Floating-Point Exception",
    20: "Virtualization Exception",
    21: "Control Protection Exception",
}


@dataclass
class ISRConfig:
    """Configuration for ISR stub generation."""
    mode: str = 'x86_64'  # 'x86_32' or 'x86_64'
    syntax: str = 'intel'  # 'intel' or 'att'
    exception_count: int = 32
    irq_base: int = 32
    irq_count: int = 16
    common_handler: str = 'interrupt_handler'
    save_all_regs: bool = True


class ISRStubGenerator:
    """Generates Interrupt Service Routine stubs."""
    
    def __init__(self, config: Optional[ISRConfig] = None):
        self.config = config or ISRConfig()
        self.is_64bit = self.config.mode == 'x86_64'
        self.is_intel = self.config.syntax == 'intel'
    
    def _comment(self, text: str) -> str:
        """Generate a comment."""
        return f"; {text}" if self.is_intel else f"# {text}"
    
    def _push(self, value: str) -> str:
        """Generate push instruction."""
        if self.is_intel:
            return f"    push {value}"
        else:
            suffix = 'q' if self.is_64bit else 'l'
            if value.isdigit() or value.startswith('0x'):
                return f"    push{suffix} ${value}"
            return f"    push{suffix} %{value}"
    
    def _pop(self, value: str) -> str:
        """Generate pop instruction."""
        if self.is_intel:
            return f"    pop {value}"
        else:
            suffix = 'q' if self.is_64bit else 'l'
            return f"    pop{suffix} %{value}"
    
    def generate_isr_stub(self, vector: int, has_error_code: bool) -> List[str]:
        """Generate a single ISR stub."""
        lines = []
        name = EXCEPTION_NAMES.get(vector, f"IRQ {vector - 32}" if vector >= 32 else f"Vector {vector}")
        
        lines.append(f"global isr{vector}")
        lines.append(f"isr{vector}:")
        lines.append(self._comment(f"ISR for {name}"))
        
        if not has_error_code:
            # Push dummy error code
            lines.append(self._push("0"))
        
        # Push interrupt number
        lines.append(self._push(str(vector)))
        
        # Jump to common handler
        if self.is_intel:
            lines.append("    jmp isr_common_stub")
        else:
            lines.append("    jmp isr_common_stub")
        
        return lines
    
    def generate_common_stub(self) -> List[str]:
        """Generate the common ISR stub that saves state."""
        lines = []
        
        lines.append(self._comment("Common ISR stub - saves processor state"))
        lines.append("isr_common_stub:")
        
        if self.is_64bit:
            # Save all general-purpose registers (64-bit)
            if self.config.save_all_regs:
                for reg in ['rax', 'rcx', 'rdx', 'rbx', 'rsp', 'rbp', 'rsi', 'rdi',
                           'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15']:
                    if reg == 'rsp':
                        continue  # Don't push RSP directly
                    lines.append(self._push(reg))
            else:
                # Save only caller-saved registers
                for reg in ['rax', 'rcx', 'rdx', 'rdi', 'rsi', 'r8', 'r9', 'r10', 'r11']:
                    lines.append(self._push(reg))
            
            # Call handler with interrupt frame pointer
            if self.is_intel:
                lines.append("    mov rdi, rsp  ; First arg: interrupt frame")
                lines.append(f"    call {self.config.common_handler}")
            else:
                lines.append("    movq %rsp, %rdi  # First arg: interrupt frame")
                lines.append(f"    call {self.config.common_handler}")
            
            # Restore registers
            if self.config.save_all_regs:
                for reg in reversed(['rax', 'rcx', 'rdx', 'rbx', 'rbp', 'rsi', 'rdi',
                                    'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15']):
                    lines.append(self._pop(reg))
            else:
                for reg in reversed(['rax', 'rcx', 'rdx', 'rdi', 'rsi', 'r8', 'r9', 'r10', 'r11']):
                    lines.append(self._pop(reg))
            
            # Clean up error code and interrupt number
            if self.is_intel:
                lines.append("    add rsp, 16")
                lines.append("    iretq")
            else:
                lines.append("    addq $16, %rsp")
                lines.append("    iretq")
        else:
            # 32-bit mode
            if self.config.save_all_regs:
                if self.is_intel:
                    lines.append("    pusha")
                else:
                    lines.append("    pusha")
            else:
                for reg in ['eax', 'ecx', 'edx']:
                    lines.append(self._push(reg))
            
            # Save segments
            if self.is_intel:
                lines.append("    push ds")
                lines.append("    push es")
                lines.append("    push fs")
                lines.append("    push gs")
                lines.append("    mov ax, 0x10  ; Kernel data segment")
                lines.append("    mov ds, ax")
                lines.append("    mov es, ax")
                lines.append("    mov fs, ax")
                lines.append("    mov gs, ax")
            else:
                lines.append("    pushl %ds")
                lines.append("    pushl %es")
                lines.append("    pushl %fs")
                lines.append("    pushl %gs")
                lines.append("    movl $0x10, %eax  # Kernel data segment")
                lines.append("    movl %eax, %ds")
                lines.append("    movl %eax, %es")
                lines.append("    movl %eax, %fs")
                lines.append("    movl %eax, %gs")
            
            # Call handler
            if self.is_intel:
                lines.append("    push esp  ; Push interrupt frame pointer")
                lines.append(f"    call {self.config.common_handler}")
                lines.append("    add esp, 4")
            else:
                lines.append("    pushl %esp  # Push interrupt frame pointer")
                lines.append(f"    call {self.config.common_handler}")
                lines.append("    addl $4, %esp")
            
            # Restore segments
            if self.is_intel:
                lines.append("    pop gs")
                lines.append("    pop fs")
                lines.append("    pop es")
                lines.append("    pop ds")
            else:
                lines.append("    popl %gs")
                lines.append("    popl %fs")
                lines.append("    popl %es")
                lines.append("    popl %ds")
            
            # Restore GPRs
            if self.config.save_all_regs:
                lines.append("    popa")
            else:
                for reg in reversed(['eax', 'ecx', 'edx']):
                    lines.append(self._pop(reg))
            
            # Clean up and return
            if self.is_intel:
                lines.append("    add esp, 8  ; Clean up error code and interrupt number")
                lines.append("    iret")
            else:
                lines.append("    addl $8, %esp  # Clean up error code and interrupt number")
                lines.append("    iret")
        
        return lines
    
    def generate_all_stubs(self) -> str:
        """Generate all ISR stubs."""
        lines = []
        
        # Header
        lines.append(self._comment("STUNIR Generated ISR Stubs"))
        lines.append(self._comment(f"Mode: {self.config.mode}"))
        lines.append(self._comment("DO-178C Level A Compliant"))
        lines.append("")
        
        lines.append("section .text")
        lines.append(f"extern {self.config.common_handler}")
        lines.append("")
        
        # Generate exception stubs (0-31)
        lines.append(self._comment("=== CPU Exceptions ==="))
        for i in range(self.config.exception_count):
            has_error = i in EXCEPTIONS_WITH_ERROR_CODE
            lines.extend(self.generate_isr_stub(i, has_error))
            lines.append("")
        
        # Generate IRQ stubs
        lines.append(self._comment("=== Hardware IRQs ==="))
        for i in range(self.config.irq_count):
            vector = self.config.irq_base + i
            lines.extend(self.generate_isr_stub(vector, False))
            lines.append("")
        
        # Generate common stub
        lines.append(self._comment("=== Common Handler ==="))
        lines.extend(self.generate_common_stub())
        
        return '\n'.join(lines)
    
    def generate_idt_table_data(self) -> str:
        """Generate IDT table initialization data."""
        lines = []
        total_vectors = self.config.exception_count + self.config.irq_count
        
        lines.append(self._comment("IDT Pointer"))
        lines.append("section .data")
        lines.append("global idt_ptr")
        lines.append("idt_ptr:")
        
        if self.is_64bit:
            if self.is_intel:
                lines.append(f"    dw idt_end - idt_start - 1  ; limit")
                lines.append("    dq idt_start                  ; base")
            else:
                lines.append(f"    .word idt_end - idt_start - 1  # limit")
                lines.append("    .quad idt_start                  # base")
        else:
            if self.is_intel:
                lines.append(f"    dw idt_end - idt_start - 1  ; limit")
                lines.append("    dd idt_start                  ; base")
            else:
                lines.append(f"    .word idt_end - idt_start - 1  # limit")
                lines.append("    .long idt_start                  # base")
        
        lines.append("")
        lines.append("section .bss")
        lines.append("align 16")
        lines.append("idt_start:")
        if self.is_64bit:
            lines.append(f"    resb {total_vectors * 16}  ; {total_vectors} entries x 16 bytes")
        else:
            lines.append(f"    resb {total_vectors * 8}  ; {total_vectors} entries x 8 bytes")
        lines.append("idt_end:")
        
        return '\n'.join(lines)


def generate_isr_stubs(
    mode: str = 'x86_64',
    syntax: str = 'intel',
    irq_count: int = 16
) -> str:
    """Convenience function to generate ISR stubs."""
    config = ISRConfig(mode=mode, syntax=syntax, irq_count=irq_count)
    gen = ISRStubGenerator(config)
    return gen.generate_all_stubs()


if __name__ == '__main__':
    # Demo
    print(generate_isr_stubs())
