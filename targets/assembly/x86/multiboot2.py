#!/usr/bin/env python3
"""STUNIR Multiboot2 Header Generator

Generates Multiboot2 compliant headers for OS bootloaders.
Supports both 32-bit and 64-bit modes.

Multiboot2 Specification: https://www.gnu.org/software/grub/manual/multiboot2/
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import IntEnum

# Multiboot2 Magic Numbers
MULTIBOOT2_HEADER_MAGIC = 0xE85250D6
MULTIBOOT2_BOOTLOADER_MAGIC = 0x36D76289

# Architecture Types
MULTIBOOT2_ARCH_I386 = 0
MULTIBOOT2_ARCH_MIPS32 = 4

# Tag Types
class MB2TagType(IntEnum):
    END = 0
    INFO_REQUEST = 1
    ADDRESS = 2
    ENTRY_ADDRESS = 3
    CONSOLE_FLAGS = 4
    FRAMEBUFFER = 5
    MODULE_ALIGN = 6
    EFI_BS = 7
    ENTRY_ADDRESS_EFI32 = 8
    ENTRY_ADDRESS_EFI64 = 9
    RELOCATABLE = 10


@dataclass
class Multiboot2Config:
    """Configuration for Multiboot2 header generation."""
    arch: str = 'i386'  # 'i386' or 'x86_64'
    entry_point: str = '_start'
    request_memory_map: bool = True
    request_framebuffer: bool = False
    fb_width: int = 800
    fb_height: int = 600
    fb_depth: int = 32
    console_flags: int = 0
    syntax: str = 'intel'  # 'intel' or 'att'


class Multiboot2HeaderGenerator:
    """Generates Multiboot2 compliant bootloader headers."""
    
    def __init__(self, config: Optional[Multiboot2Config] = None):
        self.config = config or Multiboot2Config()
    
    def generate_header_asm(self) -> str:
        """Generate Multiboot2 header in assembly format."""
        is_intel = self.config.syntax == 'intel'
        lines = []
        
        # Header comment
        cmt = ';' if is_intel else '#'
        lines.append(f"{cmt} STUNIR Generated Multiboot2 Header")
        lines.append(f"{cmt} Architecture: {self.config.arch}")
        lines.append(f"{cmt} DO-178C Level A Compliant")
        lines.append("")
        
        # Section directive
        lines.append("section .multiboot2")
        lines.append("align 8")
        lines.append("")
        
        lines.append("multiboot2_header_start:")
        
        # Magic number
        if is_intel:
            lines.append(f"    dd 0x{MULTIBOOT2_HEADER_MAGIC:08X}  ; magic")
        else:
            lines.append(f"    .long 0x{MULTIBOOT2_HEADER_MAGIC:08X}  # magic")
        
        # Architecture
        arch_val = MULTIBOOT2_ARCH_I386 if self.config.arch == 'i386' else MULTIBOOT2_ARCH_I386
        if is_intel:
            lines.append(f"    dd {arch_val}                     ; architecture (i386 protected mode)")
        else:
            lines.append(f"    .long {arch_val}                     # architecture (i386 protected mode)")
        
        # Header length placeholder (will be calculated)
        if is_intel:
            lines.append("    dd multiboot2_header_end - multiboot2_header_start  ; header length")
        else:
            lines.append("    .long multiboot2_header_end - multiboot2_header_start  # header length")
        
        # Checksum
        if is_intel:
            lines.append(f"    dd -(0x{MULTIBOOT2_HEADER_MAGIC:08X} + {arch_val} + (multiboot2_header_end - multiboot2_header_start))  ; checksum")
        else:
            lines.append(f"    .long -(0x{MULTIBOOT2_HEADER_MAGIC:08X} + {arch_val} + (multiboot2_header_end - multiboot2_header_start))  # checksum")
        
        lines.append("")
        
        # Optional: Info Request Tag
        if self.config.request_memory_map:
            lines.append(f"{cmt} Information Request Tag (memory map)")
            lines.append("align 8")
            lines.append("info_request_tag:")
            if is_intel:
                lines.append(f"    dw {MB2TagType.INFO_REQUEST}  ; type")
                lines.append("    dw 0                           ; flags")
                lines.append("    dd info_request_tag_end - info_request_tag  ; size")
                lines.append("    dd 6   ; MULTIBOOT_TAG_TYPE_MMAP")
            else:
                lines.append(f"    .word {MB2TagType.INFO_REQUEST}  # type")
                lines.append("    .word 0                           # flags")
                lines.append("    .long info_request_tag_end - info_request_tag  # size")
                lines.append("    .long 6   # MULTIBOOT_TAG_TYPE_MMAP")
            lines.append("info_request_tag_end:")
            lines.append("")
        
        # Optional: Framebuffer Tag
        if self.config.request_framebuffer:
            lines.append(f"{cmt} Framebuffer Tag")
            lines.append("align 8")
            lines.append("framebuffer_tag:")
            if is_intel:
                lines.append(f"    dw {MB2TagType.FRAMEBUFFER}  ; type")
                lines.append("    dw 0                          ; flags")
                lines.append("    dd 20                         ; size")
                lines.append(f"    dd {self.config.fb_width}    ; width")
                lines.append(f"    dd {self.config.fb_height}   ; height")
                lines.append(f"    dd {self.config.fb_depth}    ; depth")
            else:
                lines.append(f"    .word {MB2TagType.FRAMEBUFFER}  # type")
                lines.append("    .word 0                          # flags")
                lines.append("    .long 20                         # size")
                lines.append(f"    .long {self.config.fb_width}    # width")
                lines.append(f"    .long {self.config.fb_height}   # height")
                lines.append(f"    .long {self.config.fb_depth}    # depth")
            lines.append("framebuffer_tag_end:")
            lines.append("")
        
        # End Tag (required)
        lines.append(f"{cmt} End Tag")
        lines.append("align 8")
        lines.append("end_tag:")
        if is_intel:
            lines.append(f"    dw {MB2TagType.END}  ; type")
            lines.append("    dw 0                  ; flags")
            lines.append("    dd 8                  ; size")
        else:
            lines.append(f"    .word {MB2TagType.END}  # type")
            lines.append("    .word 0                  # flags")
            lines.append("    .long 8                  # size")
        lines.append("")
        lines.append("multiboot2_header_end:")
        
        return '\n'.join(lines)
    
    def generate_boot_entry(self) -> str:
        """Generate basic boot entry point assembly."""
        is_intel = self.config.syntax == 'intel'
        cmt = ';' if is_intel else '#'
        lines = []
        
        lines.append(f"{cmt} STUNIR Generated Boot Entry")
        lines.append("")
        
        if self.config.arch == 'x86_64':
            # 64-bit entry
            lines.append("section .text")
            lines.append("global _start")
            lines.append("extern kernel_main")
            lines.append("")
            lines.append("_start:")
            if is_intel:
                lines.append("    ; Save multiboot info pointer")
                lines.append("    mov rdi, rbx")
                lines.append("    ; Setup stack")
                lines.append("    mov rsp, stack_top")
                lines.append("    ; Call kernel")
                lines.append("    call kernel_main")
                lines.append("    ; Halt")
                lines.append(".halt:")
                lines.append("    cli")
                lines.append("    hlt")
                lines.append("    jmp .halt")
            else:
                lines.append("    # Save multiboot info pointer")
                lines.append("    movq %rbx, %rdi")
                lines.append("    # Setup stack")
                lines.append("    movq $stack_top, %rsp")
                lines.append("    # Call kernel")
                lines.append("    call kernel_main")
                lines.append("    # Halt")
                lines.append(".halt:")
                lines.append("    cli")
                lines.append("    hlt")
                lines.append("    jmp .halt")
        else:
            # 32-bit entry
            lines.append("section .text")
            lines.append("global _start")
            lines.append("extern kernel_main")
            lines.append("")
            lines.append("_start:")
            if is_intel:
                lines.append("    ; Disable interrupts")
                lines.append("    cli")
                lines.append("    ; Setup stack")
                lines.append("    mov esp, stack_top")
                lines.append("    ; Push multiboot info pointer")
                lines.append("    push ebx")
                lines.append("    ; Push multiboot magic")
                lines.append("    push eax")
                lines.append("    ; Call kernel")
                lines.append("    call kernel_main")
                lines.append("    ; Halt")
                lines.append(".halt:")
                lines.append("    cli")
                lines.append("    hlt")
                lines.append("    jmp .halt")
            else:
                lines.append("    # Disable interrupts")
                lines.append("    cli")
                lines.append("    # Setup stack")
                lines.append("    movl $stack_top, %esp")
                lines.append("    # Push multiboot info pointer")
                lines.append("    pushl %ebx")
                lines.append("    # Push multiboot magic")
                lines.append("    pushl %eax")
                lines.append("    # Call kernel")
                lines.append("    call kernel_main")
                lines.append("    # Halt")
                lines.append(".halt:")
                lines.append("    cli")
                lines.append("    hlt")
                lines.append("    jmp .halt")
        
        # Stack section
        lines.append("")
        lines.append("section .bss")
        lines.append("align 16")
        lines.append("stack_bottom:")
        lines.append("    resb 16384  ; 16KB stack")
        lines.append("stack_top:")
        
        return '\n'.join(lines)


def generate_multiboot2_header(
    arch: str = 'i386',
    syntax: str = 'intel',
    request_memory_map: bool = True,
    request_framebuffer: bool = False
) -> str:
    """Convenience function to generate Multiboot2 header."""
    config = Multiboot2Config(
        arch=arch,
        syntax=syntax,
        request_memory_map=request_memory_map,
        request_framebuffer=request_framebuffer
    )
    gen = Multiboot2HeaderGenerator(config)
    return gen.generate_header_asm()


if __name__ == '__main__':
    # Demo
    print(generate_multiboot2_header())
