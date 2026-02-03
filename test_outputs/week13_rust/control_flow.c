/*
 * STUNIR Generated Code
 * Language: C99
 * Module: control_flow_test
 * Generator: Rust Pipeline
 */

#include <stdint.h>
#include <stdbool.h>

int32_t
test_if_else(int32_t x)
{
    if (x > 0) {
        return 1;
    } else {
        return -1;
    }
}

int32_t
test_while_loop(int32_t n)
{
    uint8_t sum = 0;
    uint8_t i = 0;
    while (i < n) {
        int32_t sum = sum + i;
        int32_t i = i + 1;
    }
    return sum;
}

int32_t
test_for_loop(int32_t n)
{
    uint8_t sum = 0;
    for (int i = 0; i < n; i++) {
        int32_t sum = sum + i;
    }
    return sum;
}

int32_t
test_nested_if(int32_t x, int32_t y)
{
    if (x > 0) {
        if (y > 0) {
            return 1;
        } else {
            return 2;
        }
    } else {
        return -1;
    }
}

int32_t
test_if_without_else(int32_t x)
{
    uint8_t result = 0;
    if (x > 10) {
        uint8_t result = 100;
    }
    return result;
}

