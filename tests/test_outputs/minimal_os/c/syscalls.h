/* STUNIR Generated Code - System Call Interface Header
 * Module: syscalls
 * Architecture: x86_64
 * DO-178C Level A Compliance
 */

#ifndef STUNIR_SYSCALLS_H
#define STUNIR_SYSCALLS_H

#include "types.h"

i64 syscall_handler(u64 syscall_num, u64 arg1, u64 arg2, u64 arg3);
void syscall_init(void);

/* Userspace syscall wrappers */
i32 sys_print(const char* str);
void sys_yield(void);
void sys_exit(i32 code);
u32 sys_getpid(void);

#endif /* STUNIR_SYSCALLS_H */
