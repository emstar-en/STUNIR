#include "stunir_runtime.h"

#include <stdio.h>

void stunir_println(const char* s) {
  if (s == NULL) {
    fputs("\n", stdout);
    return;
  }
  fputs(s, stdout);
  fputs("\n", stdout);
}

int stunir_program(void) {
  /* STUB: codegen should replace this body. */
  stunir_println("STUNIR C skeleton: program stub");
  return 0;
}
