/*
 * STUNIR Generated Code
 * Language: C99
 * Module: call_operations_test
 * Generator: Rust Pipeline
 */

#include <stdint.h>
#include <stdbool.h>

int32_t
add(int32_t a, int32_t b)
{
    return a + b;
}

int32_t
multiply(int32_t x, int32_t y)
{
    return x * y;
}

uint8_t
get_buffer_value(const uint8_t* buffer, int32_t index)
{
    return buffer[index];
}

int32_t
get_message_id(void msg)
{
    return msg->id;
}

int32_t
test_call_operations(const uint8_t* buffer, void msg)
{
    /* nop */
    /* nop */
    int32_t sum = add(10, 20);
    /* nop */
    /* nop */
    int32_t result = multiply(sum, 2);
    /* nop */
    /* nop */
    int32_t byte_val = get_buffer_value(buffer, 0);
    /* nop */
    /* nop */
    int32_t msg_id = get_message_id(msg);
    /* nop */
    int32_t calc = result + msg_id * 2;
    /* nop */
    int32_t is_equal = sum == 30;
    /* nop */
    add(1, 2);
    return calc;
}

int32_t
test_complex_expressions(const uint8_t* data, int32_t size)
{
    /* nop */
    int32_t first = data[0];
    int32_t last = data[size - 1];
    /* nop */
    int32_t average = (first + last) / 2;
    /* nop */
    int32_t masked = first & 0xFF;
    /* nop */
    int32_t check = average > 10 && average < 100;
    return average;
}

