/*
 * STUNIR Generated Code
 * Language: C99
 * Module: combined_features_test
 * Generator: Rust Pipeline
 */

#include <stdint.h>
#include <stdbool.h>

int32_t
complex_control_flow(int32_t n)
{
    uint8_t result = 0;
    uint8_t i;
    for (i = 0; i < n; i = i + 1) {
        if (i > 10) {
            break;
        }
        switch (i % 3) {
          case 0:
                result = result + 1;
                break;
          case 1:
                continue;
          default:
                result = result + 2;
        }
    }
    return result;
}

