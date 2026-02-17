/* STUNIR Generated Code - Physical Memory Manager Header
 * Module: pmm
 * Architecture: x86_64
 * DO-178C Level A Compliance
 */

#ifndef STUNIR_PMM_H
#define STUNIR_PMM_H

#include "types.h"

void pmm_init(u64 total_mem);
void pmm_mark_free(u64 address);
void pmm_mark_used(u64 address);
u64 pmm_alloc_page(void);
void pmm_free_page(u64 address);
u64 pmm_get_free_pages(void);
u64 pmm_get_total_memory(void);

#endif /* STUNIR_PMM_H */
