/* STUNIR Embedded Startup Code */
/* Architecture: ARM */

extern void main(void);
extern unsigned long _estack;

void Reset_Handler(void) {
    /* Initialize data and bss */
    /* Call main */
    main();
    while(1);
}

