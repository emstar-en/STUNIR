/*
 * STUNIR Generated Code
 * Language: C99
 * Module: switch_fallthrough_test
 * Generator: Rust Pipeline
 */

#include <stdint.h>
#include <stdbool.h>

int32_t
handle_command(int32_t cmd)
{
    uint8_t status = 0;
    switch (cmd) {
      case 1:
            status = status + 1;
      case 2:
            status = status + 10;
            break;
      case 3:
            status = 100;
            break;
      default:
            status = -1;
    }
    return status;
}

