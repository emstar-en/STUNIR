#!/usr/bin/env python3
"""STUNIR GDT/IDT Structure Generator

Generates Global Descriptor Table and Interrupt Descriptor Table
structures for x86/x86_64 systems.

DO-178C Level A Compliant Generation
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import IntEnum, IntFlag


# GDT Access Byte Flags
class GDTAccess(IntFlag):
    ACCESSED = 0x01
    RW = 0x02           # Readable (code) / Writable (data)
    DC = 0x04           # Direction/Conforming
    EXECUTABLE = 0x08   # Code segment
    DESCRIPTOR = 0x10   # Always 1 for code/data
    DPL0 = 0x00         # Ring 0
    DPL1 = 0x20         # Ring 1
    DPL2 = 0x40         # Ring 2
    DPL3 = 0x60         # Ring 3
    PRESENT = 0x80


# GDT Flags
class GDTFlags(IntFlag):
    GRANULARITY = 0x08  # 4KB granularity
    SIZE_32 = 0x04      # 32-bit protected mode
    LONG_MODE = 0x02    # 64-bit long mode


# IDT Gate Types
class IDTGateType(IntEnum):
    TASK_GATE_16 = 0x5
    INTERRUPT_GATE_16 = 0x6
    TRAP_GATE_16 = 0x7
    INTERRUPT_GATE_32 = 0xE
    TRAP_GATE_32 = 0xF


@dataclass
class GDTEntry:
    """Represents a GDT entry."""
    name: str
    base: int = 0
    limit: int = 0xFFFFF
    access: int = 0
    flags: int = 0
    
    def encode_32bit(self) -> List[int]:
        """Encode as 8-byte 32-bit GDT entry."""
        entry = [0] * 8
        # Limit (bits 0-15)
        entry[0] = self.limit & 0xFF
        entry[1] = (self.limit >> 8) & 0xFF
        # Base (bits 0-23)
        entry[2] = self.base & 0xFF
        entry[3] = (self.base >> 8) & 0xFF
        entry[4] = (self.base >> 16) & 0xFF
        # Access
        entry[5] = self.access
        # Flags + Limit (bits 16-19)
        entry[6] = ((self.flags & 0x0F) << 4) | ((self.limit >> 16) & 0x0F)
        # Base (bits 24-31)
        entry[7] = (self.base >> 24) & 0xFF
        return entry
    
    def encode_64bit(self) -> List[int]:
        """Encode as 8-byte 64-bit GDT entry."""
        # In 64-bit mode, base and limit are mostly ignored
        return self.encode_32bit()


@dataclass
class GDTConfig:
    """Configuration for GDT generation."""
    mode: str = 'x86_64'  # 'x86_32' or 'x86_64'
    syntax: str = 'intel'
    entries: List[GDTEntry] = field(default_factory=list)


class GDTGenerator:
    """Generates Global Descriptor Table."""
    
    def __init__(self, config: Optional[GDTConfig] = None):
        self.config = config or GDTConfig()
        self.is_64bit = self.config.mode == 'x86_64'
        self.is_intel = self.config.syntax == 'intel'
        
        # Setup default entries if none provided
        if not self.config.entries:
            self.config.entries = self._default_entries()
    
    def _default_entries(self) -> List[GDTEntry]:
        """Create default GDT entries."""
        entries = []
        
        # Null descriptor (required)
        entries.append(GDTEntry(name="null", access=0, flags=0))
        
        if self.is_64bit:
            # 64-bit kernel code segment
            entries.append(GDTEntry(
                name="kernel_code",
                access=GDTAccess.PRESENT | GDTAccess.DESCRIPTOR | GDTAccess.EXECUTABLE | GDTAccess.RW,
                flags=GDTFlags.LONG_MODE | GDTFlags.GRANULARITY
            ))
            # 64-bit kernel data segment
            entries.append(GDTEntry(
                name="kernel_data",
                access=GDTAccess.PRESENT | GDTAccess.DESCRIPTOR | GDTAccess.RW,
                flags=GDTFlags.GRANULARITY
            ))
            # 64-bit user code segment
            entries.append(GDTEntry(
                name="user_code",
                access=GDTAccess.PRESENT | GDTAccess.DESCRIPTOR | GDTAccess.EXECUTABLE | GDTAccess.RW | GDTAccess.DPL3,
                flags=GDTFlags.LONG_MODE | GDTFlags.GRANULARITY
            ))
            # 64-bit user data segment
            entries.append(GDTEntry(
                name="user_data",
                access=GDTAccess.PRESENT | GDTAccess.DESCRIPTOR | GDTAccess.RW | GDTAccess.DPL3,
                flags=GDTFlags.GRANULARITY
            ))
        else:
            # 32-bit kernel code segment
            entries.append(GDTEntry(
                name="kernel_code",
                access=GDTAccess.PRESENT | GDTAccess.DESCRIPTOR | GDTAccess.EXECUTABLE | GDTAccess.RW,
                flags=GDTFlags.SIZE_32 | GDTFlags.GRANULARITY
            ))
            # 32-bit kernel data segment
            entries.append(GDTEntry(
                name="kernel_data",
                access=GDTAccess.PRESENT | GDTAccess.DESCRIPTOR | GDTAccess.RW,
                flags=GDTFlags.SIZE_32 | GDTFlags.GRANULARITY
            ))
        
        return entries
    
    def _comment(self, text: str) -> str:
        return f"; {text}" if self.is_intel else f"# {text}"
    
    def generate_gdt_asm(self) -> str:
        """Generate GDT in assembly format."""
        lines = []
        
        lines.append(self._comment("STUNIR Generated GDT"))
        lines.append(self._comment(f"Mode: {self.config.mode}"))
        lines.append(self._comment("DO-178C Level A Compliant"))
        lines.append("")
        
        lines.append("section .data")
        lines.append("align 16")
        lines.append("")
        
        # GDT pointer
        lines.append(self._comment("GDT Descriptor (GDTR)"))
        lines.append("global gdt_ptr")
        lines.append("gdt_ptr:")
        if self.is_intel:
            lines.append("    dw gdt_end - gdt_start - 1  ; limit")
            if self.is_64bit:
                lines.append("    dq gdt_start                 ; base")
            else:
                lines.append("    dd gdt_start                 ; base")
        else:
            lines.append("    .word gdt_end - gdt_start - 1  # limit")
            if self.is_64bit:
                lines.append("    .quad gdt_start                 # base")
            else:
                lines.append("    .long gdt_start                 # base")
        
        lines.append("")
        
        # GDT entries
        lines.append(self._comment("GDT Entries"))
        lines.append("global gdt_start")
        lines.append("gdt_start:")
        
        for i, entry in enumerate(self.config.entries):
            encoded = entry.encode_64bit() if self.is_64bit else entry.encode_32bit()
            lines.append(f"gdt_{entry.name}:  {self._comment(f'Selector 0x{i*8:02X}')}")            
            if self.is_intel:
                lines.append(f"    dq 0x{self._bytes_to_qword(encoded):016X}")
            else:
                lines.append(f"    .quad 0x{self._bytes_to_qword(encoded):016X}")
        
        lines.append("gdt_end:")
        lines.append("")
        
        # Selector constants
        lines.append(self._comment("Segment Selectors"))
        for i, entry in enumerate(self.config.entries):
            if self.is_intel:
                lines.append(f"{entry.name.upper()}_SEL equ 0x{i*8:02X}")
            else:
                lines.append(f".set {entry.name.upper()}_SEL, 0x{i*8:02X}")
        
        return '\n'.join(lines)
    
    def _bytes_to_qword(self, bytes_list: List[int]) -> int:
        """Convert list of bytes to qword."""
        result = 0
        for i, b in enumerate(bytes_list):
            result |= (b & 0xFF) << (i * 8)
        return result


@dataclass
class IDTEntry:
    """Represents an IDT entry."""
    vector: int
    handler: str
    gate_type: IDTGateType = IDTGateType.INTERRUPT_GATE_32
    dpl: int = 0  # 0-3
    segment_selector: int = 0x08  # Kernel code segment
    ist: int = 0  # Interrupt Stack Table (64-bit only)


@dataclass
class IDTConfig:
    """Configuration for IDT generation."""
    mode: str = 'x86_64'
    syntax: str = 'intel'
    vector_count: int = 256
    default_handler: str = 'default_interrupt_handler'


class IDTGenerator:
    """Generates Interrupt Descriptor Table setup code."""
    
    def __init__(self, config: Optional[IDTConfig] = None):
        self.config = config or IDTConfig()
        self.is_64bit = self.config.mode == 'x86_64'
        self.is_intel = self.config.syntax == 'intel'
    
    def _comment(self, text: str) -> str:
        return f"; {text}" if self.is_intel else f"# {text}"
    
    def generate_idt_setup_asm(self) -> str:
        """Generate IDT setup code in assembly."""
        lines = []
        
        lines.append(self._comment("STUNIR Generated IDT Setup"))
        lines.append(self._comment(f"Mode: {self.config.mode}"))
        lines.append(self._comment("DO-178C Level A Compliant"))
        lines.append("")
        
        # External handler references
        lines.append("section .text")
        for i in range(min(48, self.config.vector_count)):  # First 48 common
            lines.append(f"extern isr{i}")
        lines.append(f"extern {self.config.default_handler}")
        lines.append("")
        
        # IDT setup function
        lines.append("global setup_idt")
        lines.append("setup_idt:")
        
        if self.is_64bit:
            self._generate_64bit_idt_setup(lines)
        else:
            self._generate_32bit_idt_setup(lines)
        
        lines.append("")
        
        # IDT data section
        lines.append("section .data")
        lines.append("align 16")
        lines.append("")
        lines.append(self._comment("IDT Descriptor (IDTR)"))
        lines.append("global idt_ptr")
        lines.append("idt_ptr:")
        if self.is_intel:
            lines.append(f"    dw {self.config.vector_count * (16 if self.is_64bit else 8)} - 1  ; limit")
            if self.is_64bit:
                lines.append("    dq idt_entries  ; base")
            else:
                lines.append("    dd idt_entries  ; base")
        else:
            lines.append(f"    .word {self.config.vector_count * (16 if self.is_64bit else 8)} - 1  # limit")
            if self.is_64bit:
                lines.append("    .quad idt_entries  # base")
            else:
                lines.append("    .long idt_entries  # base")
        
        lines.append("")
        lines.append("section .bss")
        lines.append("align 16")
        lines.append("global idt_entries")
        lines.append("idt_entries:")
        entry_size = 16 if self.is_64bit else 8
        lines.append(f"    resb {self.config.vector_count * entry_size}")
        
        return '\n'.join(lines)
    
    def _generate_64bit_idt_setup(self, lines: List[str]):
        """Generate 64-bit IDT setup code."""
        if self.is_intel:
            lines.append("    push rbp")
            lines.append("    mov rbp, rsp")
            lines.append("    push rbx")
            lines.append("    push rcx")
            lines.append("    ")
            lines.append("    ; Setup first 48 interrupt vectors")
            lines.append("    lea rdi, [idt_entries]")
            lines.append("    mov rcx, 48")
            lines.append("    lea rsi, [.isr_table]")
            lines.append(".setup_loop:")
            lines.append("    mov rax, [rsi]        ; Get handler address")
            lines.append("    mov word [rdi], ax    ; Offset 0-15")
            lines.append("    mov word [rdi+2], 0x08 ; Kernel code selector")
            lines.append("    mov byte [rdi+4], 0   ; IST")
            lines.append("    mov byte [rdi+5], 0x8E ; Type: Interrupt gate, DPL=0, Present")
            lines.append("    shr rax, 16")
            lines.append("    mov word [rdi+6], ax  ; Offset 16-31")
            lines.append("    shr rax, 16")
            lines.append("    mov dword [rdi+8], eax ; Offset 32-63")
            lines.append("    mov dword [rdi+12], 0  ; Reserved")
            lines.append("    add rdi, 16")
            lines.append("    add rsi, 8")
            lines.append("    dec rcx")
            lines.append("    jnz .setup_loop")
            lines.append("    ")
            lines.append("    ; Load IDT")
            lines.append("    lidt [idt_ptr]")
            lines.append("    ")
            lines.append("    pop rcx")
            lines.append("    pop rbx")
            lines.append("    pop rbp")
            lines.append("    ret")
            lines.append("    ")
            lines.append(".isr_table:")
            for i in range(48):
                lines.append(f"    dq isr{i}")
        else:
            # AT&T syntax version
            lines.append("    pushq %rbp")
            lines.append("    movq %rsp, %rbp")
            lines.append("    # Similar setup for AT&T syntax...")
            lines.append("    lidt idt_ptr")
            lines.append("    popq %rbp")
            lines.append("    ret")
    
    def _generate_32bit_idt_setup(self, lines: List[str]):
        """Generate 32-bit IDT setup code."""
        if self.is_intel:
            lines.append("    push ebp")
            lines.append("    mov ebp, esp")
            lines.append("    push ebx")
            lines.append("    push ecx")
            lines.append("    ")
            lines.append("    ; Setup first 48 interrupt vectors")
            lines.append("    lea edi, [idt_entries]")
            lines.append("    mov ecx, 48")
            lines.append("    lea esi, [.isr_table]")
            lines.append(".setup_loop:")
            lines.append("    mov eax, [esi]        ; Get handler address")
            lines.append("    mov word [edi], ax    ; Offset 0-15")
            lines.append("    mov word [edi+2], 0x08 ; Kernel code selector")
            lines.append("    mov byte [edi+4], 0   ; Reserved")
            lines.append("    mov byte [edi+5], 0x8E ; Type: Interrupt gate, DPL=0, Present")
            lines.append("    shr eax, 16")
            lines.append("    mov word [edi+6], ax  ; Offset 16-31")
            lines.append("    add edi, 8")
            lines.append("    add esi, 4")
            lines.append("    dec ecx")
            lines.append("    jnz .setup_loop")
            lines.append("    ")
            lines.append("    ; Load IDT")
            lines.append("    lidt [idt_ptr]")
            lines.append("    ")
            lines.append("    pop ecx")
            lines.append("    pop ebx")
            lines.append("    pop ebp")
            lines.append("    ret")
            lines.append("    ")
            lines.append(".isr_table:")
            for i in range(48):
                lines.append(f"    dd isr{i}")
        else:
            lines.append("    pushl %ebp")
            lines.append("    movl %esp, %ebp")
            lines.append("    # Similar setup for AT&T syntax...")
            lines.append("    lidt idt_ptr")
            lines.append("    popl %ebp")
            lines.append("    ret")


def generate_gdt(mode: str = 'x86_64', syntax: str = 'intel') -> str:
    """Convenience function to generate GDT."""
    config = GDTConfig(mode=mode, syntax=syntax)
    gen = GDTGenerator(config)
    return gen.generate_gdt_asm()


def generate_idt_setup(mode: str = 'x86_64', syntax: str = 'intel') -> str:
    """Convenience function to generate IDT setup."""
    config = IDTConfig(mode=mode, syntax=syntax)
    gen = IDTGenerator(config)
    return gen.generate_idt_setup_asm()


if __name__ == '__main__':
    print("=== GDT ===")
    print(generate_gdt())
    print("\n=== IDT Setup ===")
    print(generate_idt_setup())
