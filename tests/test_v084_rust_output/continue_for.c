/*
 * STUNIR Generated Code
 * Language: C99
 * Module: continue_for_test
 * Generator: Rust Pipeline
 */

#include <stdint.h>
#include <stdbool.h>

int32_t
sum_odd_numbers(int32_t max)
{
    uint8_t sum = 0;
    uint8_t i;
    for (i = 0; i < max; i = i + 1) {
        if (i % 2 == 0) {
            continue;
        }
        sum = sum + i;
    }
    return sum;
}

