#ifndef STUNIR_RUNTIME_H
#define STUNIR_RUNTIME_H

#include <stddef.h>

/* Minimal deterministic runtime helpers. */
void stunir_println(const char* s);

/* Generated program entrypoint (to be filled by codegen). */
int stunir_program(void);

#endif
