/* STUNIR Generated Code - PIT Timer Driver
 * Module: timer_driver
 * Architecture: x86_64
 * DO-178C Level A Compliance
 */

#include "types.h"
#include "io.h"
#include "timer.h"

#define PIT_CHANNEL0    0x40
#define PIT_CHANNEL1    0x41
#define PIT_CHANNEL2    0x42
#define PIT_COMMAND     0x43

#define PIT_FREQUENCY   1193182

/* Global timer state */
static volatile u64 timer_ticks = 0;
static u32 timer_frequency = 100;

/* Initialize PIT to fire at specified frequency */
void timer_init(u32 freq) {
    u32 divisor;
    
    if (freq == 0) freq = 100;
    divisor = PIT_FREQUENCY / freq;
    
    /* Channel 0, lobyte/hibyte, rate generator */
    outb(PIT_COMMAND, 0x36);
    
    /* Send divisor */
    outb(PIT_CHANNEL0, divisor & 0xFF);
    outb(PIT_CHANNEL0, (divisor >> 8) & 0xFF);
    
    timer_frequency = freq;
    timer_ticks = 0;
}

/* Timer interrupt handler - called from ISR */
void timer_handler(void) {
    timer_ticks++;
}

/* Get current tick count */
u64 timer_get_ticks(void) {
    return timer_ticks;
}

/* Sleep for specified number of milliseconds */
void timer_sleep(u32 ms) {
    u64 target_ticks = timer_ticks + (ms * timer_frequency / 1000);
    if (target_ticks == timer_ticks) target_ticks++;
    
    while (timer_ticks < target_ticks) {
        __asm__ volatile ("hlt");
    }
}

/* Get elapsed seconds since boot */
u32 timer_get_seconds(void) {
    return timer_ticks / timer_frequency;
}
