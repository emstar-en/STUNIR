/* STUNIR Embedded Linker Script: module */
/* Architecture: arm */
/* Stack: 1024 bytes */

MEMORY
{
    FLASH (rx)  : ORIGIN = 0x08000000, LENGTH = 64K
    RAM (rwx)   : ORIGIN = 0x20000000, LENGTH = 16K
}

_estack = ORIGIN(RAM) + LENGTH(RAM);

SECTIONS
{
    .text :
    {
        . = ALIGN(4);
        *(.isr_vector)
        *(.text)
        *(.text.*)
        *(.rodata)
        *(.rodata.*)
        . = ALIGN(4);
        _etext = .;
    } > FLASH

    _sidata = LOADADDR(.data);

    .data :
    {
        . = ALIGN(4);
        _sdata = .;
        *(.data)
        *(.data.*)
        . = ALIGN(4);
        _edata = .;
    } > RAM AT > FLASH

    .bss :
    {
        . = ALIGN(4);
        _sbss = .;
        *(.bss)
        *(.bss.*)
        *(COMMON)
        . = ALIGN(4);
        _ebss = .;
    } > RAM

    ._stack :
    {
        . = ALIGN(8);
        . = . + 1024;
        . = ALIGN(8);
    } > RAM
}
