/* STUNIR: C emission (raw target) */
/* module: break_nested_test */
/* Test break in nested loops - v0.9.0 */


#include <stdint.h>
#include <stdbool.h>

/* NOTE: This is a minimal, deterministic stub emitter.
 * It preserves IR ordering and emits placeholder bodies.
 */

/* fn: find_pair_sum */
int32_t find_pair_sum(int32_t target) {
  uint8_t found = 0;
  for (i = 0; i < 10; i = i + 1) {
    for (j = 0; j < 10; j = j + 1) {
      if (i + j == target) {
        uint8_t found = 1;
        break;
      }
    }
    if (found == 1) {
      break;
    }
  }
  return found;
}

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

/* fn: complex_control_flow */
int32_t complex_control_flow(int32_t n) {
  uint8_t result = 0;
  for (i = 0; i < n; i = i + 1) {
    if (i > 10) {
      break;
    }
    switch (i % 3) {
      case 0:
        int32_t result = result + 1;
        break;
      case 1:
        continue;
      default:
        int32_t result = result + 2;
    }
  }
  return result;
}

/* fn: sum_odd_numbers */
int32_t sum_odd_numbers(int32_t max) {
  uint8_t sum = 0;
  for (i = 0; i < max; i = i + 1) {
    if (i % 2 == 0) {
      continue;
    }
    int32_t sum = sum + i;
  }
  return sum;
}

/* fn: handle_command */
int32_t handle_command(int32_t cmd) {
  uint8_t status = 0;
  switch (cmd) {
    case 1:
      int32_t status = status + 1;
    case 2:
      int32_t status = status + 10;
      break;
    case 3:
      uint8_t status = 100;
      break;
    default:
      int32_t status = -1;
  }
  return status;
}

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

