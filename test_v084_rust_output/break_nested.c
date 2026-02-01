/*
 * STUNIR Generated Code
 * Language: C99
 * Module: break_nested_test
 * Generator: Rust Pipeline
 */

#include <stdint.h>
#include <stdbool.h>

int32_t
find_pair_sum(int32_t target)
{
    uint8_t found = 0;
    uint8_t i;
    for (i = 0; i < 10; i = i + 1) {
        uint8_t j;
        for (j = 0; j < 10; j = j + 1) {
            if (i + j == target) {
                found = 1;
                break;
            }
        }
        if (found == 1) {
            break;
        }
    }
    return found;
}

