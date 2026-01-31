; STUNIR Generated Code - Interrupt Handlers
; Module: interrupts
; Architecture: x86_64
; DO-178C Level A Compliance

[BITS 64]

section .text

; External C handlers
extern irq_handler
extern exception_handler

; Macro to save all registers
%macro PUSH_ALL 0
    push rax
    push rbx
    push rcx
    push rdx
    push rsi
    push rdi
    push rbp
    push r8
    push r9
    push r10
    push r11
    push r12
    push r13
    push r14
    push r15
%endmacro

; Macro to restore all registers
%macro POP_ALL 0
    pop r15
    pop r14
    pop r13
    pop r12
    pop r11
    pop r10
    pop r9
    pop r8
    pop rbp
    pop rdi
    pop rsi
    pop rdx
    pop rcx
    pop rbx
    pop rax
%endmacro

; ISR stubs for CPU exceptions (0-31)
%macro ISR_NOERRCODE 1
global isr%1
isr%1:
    cli
    push 0              ; Dummy error code
    push %1             ; Interrupt number
    jmp isr_common
%endmacro

%macro ISR_ERRCODE 1
global isr%1
isr%1:
    cli
    push %1             ; Interrupt number
    jmp isr_common
%endmacro

; IRQ stubs (32-47)
%macro IRQ 2
global irq%1
irq%1:
    cli
    push 0              ; Dummy error code
    push %2             ; Interrupt number
    jmp irq_common
%endmacro

; CPU Exceptions
ISR_NOERRCODE 0     ; Divide by zero
ISR_NOERRCODE 1     ; Debug
ISR_NOERRCODE 2     ; NMI
ISR_NOERRCODE 3     ; Breakpoint
ISR_NOERRCODE 4     ; Overflow
ISR_NOERRCODE 5     ; Bound range exceeded
ISR_NOERRCODE 6     ; Invalid opcode
ISR_NOERRCODE 7     ; Device not available
ISR_ERRCODE   8     ; Double fault
ISR_NOERRCODE 9     ; Coprocessor segment overrun
ISR_ERRCODE   10    ; Invalid TSS
ISR_ERRCODE   11    ; Segment not present
ISR_ERRCODE   12    ; Stack fault
ISR_ERRCODE   13    ; General protection fault
ISR_ERRCODE   14    ; Page fault
ISR_NOERRCODE 15    ; Reserved
ISR_NOERRCODE 16    ; x87 FP exception
ISR_ERRCODE   17    ; Alignment check
ISR_NOERRCODE 18    ; Machine check
ISR_NOERRCODE 19    ; SIMD FP exception

; IRQs
IRQ 0, 32           ; Timer
IRQ 1, 33           ; Keyboard
IRQ 2, 34           ; Cascade
IRQ 3, 35           ; COM2
IRQ 4, 36           ; COM1
IRQ 5, 37           ; LPT2
IRQ 6, 38           ; Floppy
IRQ 7, 39           ; LPT1
IRQ 8, 40           ; CMOS RTC
IRQ 9, 41           ; Free
IRQ 10, 42          ; Free
IRQ 11, 43          ; Free
IRQ 12, 44          ; PS/2 Mouse
IRQ 13, 45          ; FPU
IRQ 14, 46          ; Primary ATA
IRQ 15, 47          ; Secondary ATA

; Common ISR handler
isr_common:
    PUSH_ALL
    
    ; Get interrupt number
    mov rdi, [rsp + 120]    ; int_no is at offset 120 (15 regs * 8)
    
    ; Call C exception handler
    call exception_handler
    
    POP_ALL
    add rsp, 16             ; Remove error code and int_no
    iretq

; Common IRQ handler
irq_common:
    PUSH_ALL
    
    ; Get interrupt number and convert to IRQ number
    mov rdi, [rsp + 120]
    sub rdi, 32             ; Convert to IRQ number (0-15)
    
    ; Call C IRQ handler
    call irq_handler
    
    POP_ALL
    add rsp, 16
    iretq

; Syscall entry point
global syscall_entry
extern syscall_handler

syscall_entry:
    PUSH_ALL
    
    ; Args: rax=syscall, rdi=arg1, rsi=arg2, rdx=arg3
    ; Rearrange for C calling convention
    mov rcx, rdx            ; arg3
    mov rdx, rsi            ; arg2
    mov rsi, rdi            ; arg1
    mov rdi, rax            ; syscall_num
    
    call syscall_handler
    
    ; Return value is in rax, store it
    mov [rsp + 112], rax    ; Store return value at rax position
    
    POP_ALL
    iretq

; Context switch (called from C)
; void context_switch(task_context* old, task_context* new)
global context_switch
context_switch:
    ; Save old context
    mov [rdi + 0], rsp
    mov [rdi + 16], rbp
    mov [rdi + 24], rbx
    mov [rdi + 32], r12
    mov [rdi + 40], r13
    mov [rdi + 48], r14
    mov [rdi + 56], r15
    
    ; Load new context
    mov rsp, [rsi + 0]
    mov rbp, [rsi + 16]
    mov rbx, [rsi + 24]
    mov r12, [rsi + 32]
    mov r13, [rsi + 40]
    mov r14, [rsi + 48]
    mov r15, [rsi + 56]
    
    ret
