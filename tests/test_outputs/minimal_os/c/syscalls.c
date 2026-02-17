/* STUNIR Generated Code - System Call Interface
 * Module: syscalls
 * Architecture: x86_64
 * DO-178C Level A Compliance
 */

#include "types.h"
#include "serial.h"
#include "timer.h"
#include "scheduler.h"
#include "interrupts.h"
#include "syscalls.h"

#define SYS_PRINT   0
#define SYS_YIELD   1
#define SYS_EXIT    2
#define SYS_GETPID  3
#define SYS_SLEEP   4

#define SYSCALL_VECTOR  0x80

/* External syscall entry point from assembly */
extern void syscall_entry(void);

/* Main syscall dispatch handler */
i64 syscall_handler(u64 syscall_num, u64 arg1, u64 arg2, u64 arg3) {
    (void)arg2;  /* Unused for now */
    (void)arg3;  /* Unused for now */
    
    switch (syscall_num) {
        case SYS_PRINT:
            serial_write_string((const char*)arg1);
            return 0;
            
        case SYS_YIELD:
            yield();
            return 0;
            
        case SYS_EXIT:
            task_exit();
            return 0;  /* Never reached */
            
        case SYS_GETPID:
            return get_current_task_id();
            
        case SYS_SLEEP:
            timer_sleep((u32)arg1);
            return 0;
            
        default:
            return -1;
    }
}

/* Install syscall handler */
void syscall_init(void) {
    idt_set_gate(SYSCALL_VECTOR, (u64)syscall_entry, 0x08, 0xEE);
}

/* Userspace syscall wrappers */
i32 sys_print(const char* str) {
    i64 result;
    __asm__ volatile (
        "movq $0, %%rax\n"
        "movq %1, %%rdi\n"
        "int $0x80\n"
        "movq %%rax, %0\n"
        : "=r"(result)
        : "r"((u64)str)
        : "rax", "rdi"
    );
    return (i32)result;
}

void sys_yield(void) {
    __asm__ volatile (
        "movq $1, %%rax\n"
        "int $0x80\n"
        : : : "rax"
    );
}

void sys_exit(i32 code) {
    (void)code;
    __asm__ volatile (
        "movq $2, %%rax\n"
        "int $0x80\n"
        : : : "rax"
    );
    while (1);
}

u32 sys_getpid(void) {
    i64 result;
    __asm__ volatile (
        "movq $3, %%rax\n"
        "int $0x80\n"
        "movq %%rax, %0\n"
        : "=r"(result)
        : : "rax"
    );
    return (u32)result;
}
