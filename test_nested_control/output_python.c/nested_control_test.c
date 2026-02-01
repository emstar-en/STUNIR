/* STUNIR: C emission (raw target) */
/* module: nested_control_test */
/* Test nested control flow */


#include <stdint.h>
#include <stdbool.h>

/* NOTE: This is a minimal, deterministic stub emitter.
 * It preserves IR ordering and emits placeholder bodies.
 */

/* fn: nested_if_test */
int32_t nested_if_test(int32_t x, int32_t y) {
  if (x > 0) {
    if (y > 0) {
      return x + y;
    } else {
      return x - y;
    }
  } else {
    return 0;
  }
}

