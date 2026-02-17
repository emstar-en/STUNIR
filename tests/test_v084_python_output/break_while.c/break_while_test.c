/* STUNIR: C emission (raw target) */
/* module: break_while_test */
/* Test break statement in while loop - v0.9.0 */


#include <stdint.h>
#include <stdbool.h>

/* NOTE: This is a minimal, deterministic stub emitter.
 * It preserves IR ordering and emits placeholder bodies.
 */

/* fn: find_first_divisible */
int32_t find_first_divisible(int32_t max, int32_t divisor) {
  uint8_t i = 1;
  int32_t result = -1;
  while (i < max) {
    if (i % divisor == 0) {
      int32_t result = i;
      break;
    }
    int32_t i = i + 1;
  }
  return result;
}

