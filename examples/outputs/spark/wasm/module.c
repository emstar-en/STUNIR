/* STUNIR Generated WASM Code */
/* WASI Enabled */

#include <stdint.h>
#include <wasi/api.h>

#define WASM_EXPORT __attribute__((visibility("default")))

typedef struct {
    int32_t x;
    int32_t y;
} Point;

WASM_EXPORT int32_t add(int32_t a, int32_t b) {
    /* WASM Function Body */
    return a + b;
}

WASM_EXPORT void process_point(Point* p) {
    /* WASM Function Body */
}

