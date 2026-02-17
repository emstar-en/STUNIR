/* STUNIR: C emission (raw target) */
/* module: switch_simple_test */
/* Test simple switch/case statement - v0.9.0 */


#include <stdint.h>
#include <stdbool.h>

/* NOTE: This is a minimal, deterministic stub emitter.
 * It preserves IR ordering and emits placeholder bodies.
 */

/* fn: get_day_type */
int32_t get_day_type(int32_t day) {
  uint8_t result = 0;
  switch (day) {
    case 1:
      uint8_t result = 1;
      break;
    case 2:
      uint8_t result = 1;
      break;
    case 6:
      uint8_t result = 2;
      break;
    case 7:
      uint8_t result = 2;
      break;
    default:
      uint8_t result = 1;
  }
  return result;
}

