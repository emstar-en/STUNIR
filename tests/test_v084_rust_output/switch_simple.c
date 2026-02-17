/*
 * STUNIR Generated Code
 * Language: C99
 * Module: switch_simple_test
 * Generator: Rust Pipeline
 */

#include <stdint.h>
#include <stdbool.h>

int32_t
get_day_type(int32_t day)
{
    uint8_t result = 0;
    switch (day) {
      case 1:
            result = 1;
            break;
      case 2:
            result = 1;
            break;
      case 6:
            result = 2;
            break;
      case 7:
            result = 2;
            break;
      default:
            result = 1;
    }
    return result;
}

