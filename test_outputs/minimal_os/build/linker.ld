/* STUNIR Generated Linker Script
 * Target: x86_64 bare-metal
 * Entry: _start (Multiboot2)
 */

ENTRY(_start)

SECTIONS
{
    /* Load at 1MB (standard for Multiboot) */
    . = 1M;

    /* Multiboot header must be in first 32KB */
    .multiboot ALIGN(8) :
    {
        *(.multiboot)
    }

    /* Text section */
    .text ALIGN(4K) :
    {
        *(.text)
        *(.text.*)
    }

    /* Read-only data */
    .rodata ALIGN(4K) :
    {
        *(.rodata)
        *(.rodata.*)
    }

    /* Read-write data */
    .data ALIGN(4K) :
    {
        *(.data)
        *(.data.*)
    }

    /* BSS (uninitialized data) */
    .bss ALIGN(4K) :
    {
        __bss_start = .;
        *(.bss)
        *(.bss.*)
        *(COMMON)
        __bss_end = .;
    }

    /* Kernel end marker */
    __kernel_end = .;
}
