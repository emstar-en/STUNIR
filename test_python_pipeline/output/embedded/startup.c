/* STUNIR Embedded Startup: module */
/* Architecture: arm */

#include <stdint.h>
#include "module.h"

/* Stack and heap configuration */
extern uint32_t _estack;
extern uint32_t _sdata, _edata, _sidata;
extern uint32_t _sbss, _ebss;

/* Reset handler */
void Reset_Handler(void) {
    uint32_t *src, *dst;
    
    /* Copy .data section from Flash to RAM */
    src = &_sidata;
    dst = &_sdata;
    while (dst < &_edata) {
        *dst++ = *src++;
    }
    
    /* Zero .bss section */
    dst = &_sbss;
    while (dst < &_ebss) {
        *dst++ = 0;
    }
    
    /* Call main (or first function) */
    main();
    
    /* Infinite loop if main returns */
    while (1) {}
}

/* Default interrupt handler */
void Default_Handler(void) {
    while (1) {}
}

/* Weak aliases for interrupt handlers */
void NMI_Handler(void) __attribute__((weak, alias("Default_Handler")));
void HardFault_Handler(void) __attribute__((weak, alias("Default_Handler")));
