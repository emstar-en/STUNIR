#include <stdio.h>

void main_func() {
    printf("Compiling from Rust!\n");
    int x = 10 + 32;
    printf("%d\n", x);
    for (int i = 0; i < 3; i++) {
        printf("Echo...\n");
    }
}

int main() { main_func(); return 0; }
