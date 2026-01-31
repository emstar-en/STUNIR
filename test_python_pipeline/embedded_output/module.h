/* STUNIR Embedded Module: module */
/* Architecture: arm */
/* Epoch: 1769856144 */

#ifndef MODULE_H
#define MODULE_H

#include <stdint.h>

/* Function prototypes */
void connect(void);
void execute_query(void);
void close(void);
void map(void);
void filter(void);
void reduce(void);
void vector_add_kernel(void);
void matrix_mul_kernel(void);
void matrix_multiply(void);
void vector_dot_product(void);
void matrix_transpose(void);
void add(void);
void multiply(void);
void get_user(void);
void create_user(void);
void update_user(void);
void delete_user(void);

#endif /* MODULE_H */