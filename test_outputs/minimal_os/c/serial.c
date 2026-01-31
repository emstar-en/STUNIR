/* STUNIR Generated Code - Serial Driver
 * Module: serial_driver
 * Architecture: x86_64
 * DO-178C Level A Compliance
 */

#include "types.h"
#include "io.h"
#include "serial.h"

/* COM1 port base address */
#define COM1_PORT               0x3F8

/* Register offsets */
#define SERIAL_DATA             0
#define SERIAL_INTERRUPT_ENABLE 1
#define SERIAL_FIFO_CONTROL     2
#define SERIAL_LINE_CONTROL     3
#define SERIAL_MODEM_CONTROL    4
#define SERIAL_LINE_STATUS      5
#define SERIAL_BAUD_DIVISOR_LOW  0
#define SERIAL_BAUD_DIVISOR_HIGH 1

#define BAUD_115200_DIVISOR     1

/* Initialize COM1 serial port at 115200 baud, 8N1 */
i32 serial_init(void) {
    /* Disable interrupts */
    outb(COM1_PORT + SERIAL_INTERRUPT_ENABLE, 0x00);
    
    /* Enable DLAB to set baud rate */
    outb(COM1_PORT + SERIAL_LINE_CONTROL, 0x80);
    
    /* Set baud rate divisor (115200) */
    outb(COM1_PORT + SERIAL_BAUD_DIVISOR_LOW, BAUD_115200_DIVISOR);
    outb(COM1_PORT + SERIAL_BAUD_DIVISOR_HIGH, 0x00);
    
    /* 8 bits, no parity, 1 stop bit */
    outb(COM1_PORT + SERIAL_LINE_CONTROL, 0x03);
    
    /* Enable FIFO, clear, 14-byte threshold */
    outb(COM1_PORT + SERIAL_FIFO_CONTROL, 0xC7);
    
    /* Enable IRQs, RTS/DSR set */
    outb(COM1_PORT + SERIAL_MODEM_CONTROL, 0x0B);
    
    /* Test serial chip (set loopback mode) */
    outb(COM1_PORT + SERIAL_MODEM_CONTROL, 0x1E);
    outb(COM1_PORT + SERIAL_DATA, 0xAE);
    
    /* Check if we get same byte back */
    u8 test_byte = inb(COM1_PORT + SERIAL_DATA);
    if (test_byte != 0xAE) {
        return -1;
    }
    
    /* Serial is working, disable loopback and enable normal mode */
    outb(COM1_PORT + SERIAL_MODEM_CONTROL, 0x0F);
    return 0;
}

/* Check if transmit buffer is empty */
static i32 serial_is_transmit_empty(void) {
    u8 status = inb(COM1_PORT + SERIAL_LINE_STATUS);
    return (status & 0x20) != 0;
}

/* Write single character to serial port (blocking) */
void serial_write_char(char c) {
    while (!serial_is_transmit_empty());
    outb(COM1_PORT + SERIAL_DATA, c);
}

/* Write null-terminated string to serial port */
void serial_write_string(const char* str) {
    while (*str) {
        serial_write_char(*str);
        str++;
    }
}

/* Write 32-bit value as hexadecimal */
void serial_write_hex(u32 value) {
    const char* hex_chars = "0123456789ABCDEF";
    i32 i;
    
    serial_write_string("0x");
    for (i = 28; i >= 0; i -= 4) {
        u8 nibble = (value >> i) & 0xF;
        serial_write_char(hex_chars[nibble]);
    }
}

/* Write 64-bit value as hexadecimal */
void serial_write_hex64(u64 value) {
    const char* hex_chars = "0123456789ABCDEF";
    i32 i;
    
    serial_write_string("0x");
    for (i = 60; i >= 0; i -= 4) {
        u8 nibble = (value >> i) & 0xF;
        serial_write_char(hex_chars[nibble]);
    }
}
