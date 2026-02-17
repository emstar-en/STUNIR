/*
 * STUNIR Generated Code
 * Language: C99
 * Module: break_while_test
 * Generator: Rust Pipeline
 */

#include <stdint.h>
#include <stdbool.h>

int32_t
find_first_divisible(int32_t max, int32_t divisor)
{
    uint8_t i = 1;
    int32_t result = -1;
    while (i < max) {
        if (i % divisor == 0) {
            result = i;
            break;
        }
        i = i + 1;
    }
    return result;
}

