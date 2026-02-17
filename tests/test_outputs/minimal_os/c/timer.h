/* STUNIR Generated Code - PIT Timer Driver Header
 * Module: timer_driver
 * Architecture: x86_64
 * DO-178C Level A Compliance
 */

#ifndef STUNIR_TIMER_H
#define STUNIR_TIMER_H

#include "types.h"

void timer_init(u32 freq);
void timer_handler(void);
u64 timer_get_ticks(void);
void timer_sleep(u32 ms);
u32 timer_get_seconds(void);

#endif /* STUNIR_TIMER_H */
