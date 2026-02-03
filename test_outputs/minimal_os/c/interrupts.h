/* STUNIR Generated Code - Interrupt Handling Header
 * Module: interrupts
 * Architecture: x86_64
 * DO-178C Level A Compliance
 */

#ifndef STUNIR_INTERRUPTS_H
#define STUNIR_INTERRUPTS_H

#include "types.h"

void idt_init(void);
void idt_set_gate(u8 num, u64 handler, u16 selector, u8 flags);
void irq_handler(u32 irq_num);
void exception_handler(u32 int_no);

#endif /* STUNIR_INTERRUPTS_H */
