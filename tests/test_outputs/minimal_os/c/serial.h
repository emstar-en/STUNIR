/* STUNIR Generated Code - Serial Driver Header
 * Module: serial_driver
 * Architecture: x86_64
 * DO-178C Level A Compliance
 */

#ifndef STUNIR_SERIAL_H
#define STUNIR_SERIAL_H

#include "types.h"

i32 serial_init(void);
void serial_write_char(char c);
void serial_write_string(const char* str);
void serial_write_hex(u32 value);
void serial_write_hex64(u64 value);

#endif /* STUNIR_SERIAL_H */
