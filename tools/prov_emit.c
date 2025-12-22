#include <stdio.h>
#include <stdint.h>
#include "provenance.h"

#ifndef _STUNIR_BUILD_EPOCH
#define _STUNIR_BUILD_EPOCH 0
#endif

int main(void) {
    printf("build_epoch=%lld\n", (long long)STUNIR_PROV_BUILD_EPOCH);
    printf("selected_epoch=%lld\n", (long long)_STUNIR_BUILD_EPOCH);
    printf("spec_digest=%s\n", STUNIR_PROV_SPEC_DIGEST);
    printf("asm_digest=%s\n", STUNIR_PROV_ASM_DIGEST);
    return 0;
}
