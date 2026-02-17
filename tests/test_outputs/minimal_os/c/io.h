/* STUNIR Generated Code - I/O Port Access
 * Module: io
 * Architecture: x86_64
 * DO-178C Level A Compliance
 */

#ifndef STUNIR_IO_H
#define STUNIR_IO_H

#include "types.h"

/* Write byte to I/O port */
static inline void outb(u16 port, u8 value) {
    __asm__ volatile ("outb %1, %0" : : "dN"(port), "a"(value));
}

/* Read byte from I/O port */
static inline u8 inb(u16 port) {
    u8 result;
    __asm__ volatile ("inb %1, %0" : "=a"(result) : "dN"(port));
    return result;
}

/* I/O wait (short delay) */
static inline void io_wait(void) {
    outb(0x80, 0);
}

#endif /* STUNIR_IO_H */
