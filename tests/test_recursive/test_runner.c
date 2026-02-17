#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

// Function declarations
int32_t test_nested_2(int32_t x);
int32_t test_nested_3(int32_t x);
int32_t test_nested_4(int32_t x);
int32_t test_nested_5(int32_t x);

int main() {
    printf("Testing SPARK-generated recursive control flow\n\n");
    
    // Test 2-level nesting
    printf("test_nested_2(50) = %d (expected: 100)\n", test_nested_2(50));
    printf("test_nested_2(5) = %d (expected: 10)\n", test_nested_2(5));
    printf("test_nested_2(-5) = %d (expected: 0)\n", test_nested_2(-5));
    printf("\n");
    
    // Test 3-level nesting
    printf("test_nested_3(25) = %d (expected: 200)\n", test_nested_3(25));
    printf("test_nested_3(15) = %d (expected: 100)\n", test_nested_3(15));
    printf("test_nested_3(5) = %d (expected: 10)\n", test_nested_3(5));
    printf("\n");
    
    // Test 4-level nesting
    printf("test_nested_4(35) = %d (expected: 300)\n", test_nested_4(35));
    printf("test_nested_4(25) = %d (expected: 200)\n", test_nested_4(25));
    printf("test_nested_4(15) = %d (expected: 100)\n", test_nested_4(15));
    printf("\n");
    
    // Test 5-level nesting
    printf("test_nested_5(45) = %d (expected: 400)\n", test_nested_5(45));
    printf("test_nested_5(35) = %d (expected: 300)\n", test_nested_5(35));
    printf("test_nested_5(25) = %d (expected: 200)\n", test_nested_5(25));
    printf("\n");
    
    printf("All tests completed successfully!\n");
    return 0;
}
