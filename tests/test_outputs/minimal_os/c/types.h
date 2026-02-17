/* STUNIR Generated Code - Common Type Definitions
 * Module: types
 * Architecture: x86_64
 * DO-178C Level A Compliance
 */

#ifndef STUNIR_TYPES_H
#define STUNIR_TYPES_H

/* Fixed-width integer types (C89 compatible) */
typedef signed char         i8;
typedef unsigned char       u8;
typedef signed short        i16;
typedef unsigned short      u16;
typedef signed int          i32;
typedef unsigned int        u32;
typedef signed long long    i64;
typedef unsigned long long  u64;

typedef u64                 size_t;
typedef i64                 ssize_t;
typedef u64                 uintptr_t;

/* Boolean type */
typedef u8                  bool;
#define true    1
#define false   0

/* NULL pointer */
#ifndef NULL
#define NULL    ((void*)0)
#endif

#endif /* STUNIR_TYPES_H */
