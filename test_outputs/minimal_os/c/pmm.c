/* STUNIR Generated Code - Physical Memory Manager
 * Module: pmm
 * Architecture: x86_64
 * DO-178C Level A Compliance
 */

#include "types.h"
#include "pmm.h"

#define PAGE_SIZE       4096
#define MAX_MEMORY_MB   256
#define MAX_PAGES       ((MAX_MEMORY_MB * 1024 * 1024) / PAGE_SIZE)
#define BITMAP_SIZE     (MAX_PAGES / 8)

/* Memory bitmap - each bit represents one page */
static u8 memory_bitmap[BITMAP_SIZE];

/* Statistics */
static u64 total_memory = 0;
static u64 free_pages = 0;
static u64 used_pages = 0;

/* Initialize physical memory manager */
void pmm_init(u64 total_mem) {
    u32 i;
    
    /* Clear bitmap - mark all as used initially */
    for (i = 0; i < BITMAP_SIZE; i++) {
        memory_bitmap[i] = 0xFF;
    }
    
    total_memory = total_mem;
    free_pages = 0;
    used_pages = total_mem / PAGE_SIZE;
    
    /* Mark usable memory as free (skip first 2MB for kernel) */
    for (i = 512; i < (total_mem / PAGE_SIZE) && i < MAX_PAGES; i++) {
        pmm_mark_free(i * PAGE_SIZE);
    }
}

/* Mark a page as free */
void pmm_mark_free(u64 address) {
    u32 page = address / PAGE_SIZE;
    u32 byte_idx = page / 8;
    u32 bit_idx = page % 8;
    
    if (byte_idx >= BITMAP_SIZE) return;
    
    memory_bitmap[byte_idx] &= ~(1 << bit_idx);
    free_pages++;
    if (used_pages > 0) used_pages--;
}

/* Mark a page as used */
void pmm_mark_used(u64 address) {
    u32 page = address / PAGE_SIZE;
    u32 byte_idx = page / 8;
    u32 bit_idx = page % 8;
    
    if (byte_idx >= BITMAP_SIZE) return;
    
    memory_bitmap[byte_idx] |= (1 << bit_idx);
    if (free_pages > 0) free_pages--;
    used_pages++;
}

/* Allocate a single physical page */
u64 pmm_alloc_page(void) {
    u32 i, j;
    
    for (i = 0; i < BITMAP_SIZE; i++) {
        if (memory_bitmap[i] != 0xFF) {
            for (j = 0; j < 8; j++) {
                if (!(memory_bitmap[i] & (1 << j))) {
                    u32 page = i * 8 + j;
                    pmm_mark_used(page * PAGE_SIZE);
                    return (u64)(page * PAGE_SIZE);
                }
            }
        }
    }
    
    return 0;  /* Out of memory */
}

/* Free a physical page */
void pmm_free_page(u64 address) {
    if (address == 0) return;
    pmm_mark_free(address);
}

/* Get number of free pages */
u64 pmm_get_free_pages(void) {
    return free_pages;
}

/* Get total memory */
u64 pmm_get_total_memory(void) {
    return total_memory;
}
