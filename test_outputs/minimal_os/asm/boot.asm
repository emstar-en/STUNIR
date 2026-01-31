; STUNIR Generated Code - Multiboot2 Bootloader
; Module: multiboot_boot
; Architecture: x86_64 (starts in 32-bit protected mode)
; DO-178C Level A Compliance

[BITS 32]

; Multiboot2 constants
MULTIBOOT2_MAGIC        equ 0xE85250D6
MULTIBOOT2_ARCHITECTURE equ 0           ; i386 protected mode
MULTIBOOT2_HEADER_LENGTH equ (multiboot_header_end - multiboot_header_start)
MULTIBOOT2_CHECKSUM     equ -(MULTIBOOT2_MAGIC + MULTIBOOT2_ARCHITECTURE + MULTIBOOT2_HEADER_LENGTH)

KERNEL_STACK_SIZE       equ 16384

; Page table constants
CR0_PG                  equ 0x80000000
CR4_PAE                 equ 0x00000020
EFER_MSR                equ 0xC0000080
EFER_LME                equ 0x00000100

section .multiboot
align 8
multiboot_header_start:
    dd MULTIBOOT2_MAGIC
    dd MULTIBOOT2_ARCHITECTURE
    dd MULTIBOOT2_HEADER_LENGTH
    dd MULTIBOOT2_CHECKSUM
    ; End tag
    dw 0    ; type
    dw 0    ; flags
    dd 8    ; size
multiboot_header_end:

section .rodata
align 16
gdt64:
    ; Null descriptor
    dq 0
    ; Code segment (64-bit)
gdt64_code: equ $ - gdt64
    dw 0xFFFF       ; Limit low
    dw 0            ; Base low
    db 0            ; Base middle
    db 0x9A         ; Access (present, ring 0, code, readable)
    db 0xAF         ; Flags (64-bit, 4KB granularity) + Limit high
    db 0            ; Base high
    ; Data segment (64-bit)
gdt64_data: equ $ - gdt64
    dw 0xFFFF       ; Limit low
    dw 0            ; Base low
    db 0            ; Base middle
    db 0x92         ; Access (present, ring 0, data, writable)
    db 0xCF         ; Flags (32-bit, 4KB granularity) + Limit high
    db 0            ; Base high
gdt64_end:

gdt64_ptr:
    dw gdt64_end - gdt64 - 1  ; Limit (16-bit)
    dq gdt64                   ; Base (64-bit)

section .bss
align 4096
pml4:
    resb 4096
pdpt:
    resb 4096
pd:
    resb 4096

align 16
stack_bottom:
    resb KERNEL_STACK_SIZE
stack_top:

section .text
global _start
extern kernel_main

_start:
    ; Save multiboot info
    mov edi, eax    ; Multiboot magic
    mov esi, ebx    ; Multiboot info pointer
    
    ; Set up the stack
    mov esp, stack_top
    
    ; Set up page tables
    call setup_page_tables
    
    ; Enable PAE
    mov eax, cr4
    or eax, CR4_PAE
    mov cr4, eax
    
    ; Enable Long Mode in EFER MSR
    mov ecx, EFER_MSR
    rdmsr
    or eax, EFER_LME
    wrmsr
    
    ; Enable paging
    mov eax, cr0
    or eax, CR0_PG
    mov cr0, eax
    
    ; Load 64-bit GDT
    lgdt [gdt64_ptr]
    
    ; Far jump to 64-bit code
    jmp gdt64_code:long_mode_start

setup_page_tables:
    ; Clear page tables
    mov edi, pml4
    xor eax, eax
    mov ecx, 3072       ; 3 pages * 1024 dwords
    rep stosd
    
    ; Load PML4 into CR3
    mov eax, pml4
    mov cr3, eax
    
    ; Set up PML4[0] -> PDPT
    mov eax, pdpt
    or eax, 0x03        ; Present + Writable
    mov [pml4], eax
    
    ; Set up PDPT[0] -> PD
    mov eax, pd
    or eax, 0x03
    mov [pdpt], eax
    
    ; Set up PD with 2MB pages (identity map first 16MB)
    mov dword [pd + 0],  0x00000083  ; 0-2MB
    mov dword [pd + 8],  0x00200083  ; 2-4MB
    mov dword [pd + 16], 0x00400083  ; 4-6MB
    mov dword [pd + 24], 0x00600083  ; 6-8MB
    mov dword [pd + 32], 0x00800083  ; 8-10MB
    mov dword [pd + 40], 0x00A00083  ; 10-12MB
    mov dword [pd + 48], 0x00C00083  ; 12-14MB
    mov dword [pd + 56], 0x00E00083  ; 14-16MB
    
    ret

[BITS 64]
long_mode_start:
    ; Set up segment registers
    mov ax, gdt64_data
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    mov ss, ax
    
    ; Set up 64-bit stack
    mov rsp, stack_top
    
    ; Clear direction flag
    cld
    
    ; Pass saved multiboot info to kernel_main
    ; edi already has magic, esi has mboot_info from earlier
    ; But we need to zero-extend them to 64-bit
    mov eax, edi
    mov edi, eax        ; magic in rdi (first arg)
    mov eax, esi
    mov rsi, rax        ; mboot_info in rsi (second arg)
    
    ; Call kernel main
    call kernel_main
    
    ; Hang if kernel_main returns
.hang:
    cli
    hlt
    jmp .hang
