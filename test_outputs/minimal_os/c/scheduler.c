/* STUNIR Generated Code - Cooperative Scheduler
 * Module: scheduler
 * Architecture: x86_64
 * DO-178C Level A Compliance
 */

#include "types.h"
#include "pmm.h"
#include "serial.h"
#include "scheduler.h"

#define MAX_TASKS       16
#define TASK_STACK_SIZE 4096

#define TASK_STATE_READY        0
#define TASK_STATE_RUNNING      1
#define TASK_STATE_BLOCKED      2
#define TASK_STATE_TERMINATED   3

/* Task context (callee-saved registers) */
struct task_context {
    u64 rsp;
    u64 rip;
    u64 rbp;
    u64 rbx;
    u64 r12;
    u64 r13;
    u64 r14;
    u64 r15;
};

/* Task structure */
struct task {
    u32 id;
    u32 state;
    struct task_context context;
    u8* stack;
    void (*entry)(void);
    char name[32];
};

/* Global scheduler state */
static struct task tasks[MAX_TASKS];
static u32 task_count = 0;
static u32 current_task = 0;
u32 scheduler_running = 0;

/* External assembly function for context switch */
extern void context_switch(struct task_context* old, struct task_context* new);

/* Initialize the scheduler */
void scheduler_init(void) {
    u32 i;
    
    for (i = 0; i < MAX_TASKS; i++) {
        tasks[i].id = i;
        tasks[i].state = TASK_STATE_TERMINATED;
        tasks[i].stack = 0;
        tasks[i].entry = 0;
    }
    
    task_count = 0;
    current_task = 0;
    scheduler_running = 0;
}

/* Create a new task */
i32 task_create(void (*entry)(void), const char* name) {
    u32 i;
    
    if (task_count >= MAX_TASKS) {
        return -1;
    }
    
    /* Find free slot */
    for (i = 0; i < MAX_TASKS; i++) {
        if (tasks[i].state == TASK_STATE_TERMINATED) {
            /* Allocate stack */
            tasks[i].stack = (u8*)pmm_alloc_page();
            if (tasks[i].stack == 0) {
                return -1;
            }
            
            /* Set up initial context - just store entry point */
            tasks[i].context.rsp = (u64)(tasks[i].stack + TASK_STACK_SIZE - 8);
            tasks[i].context.rip = (u64)entry;
            tasks[i].context.rbp = 0;
            tasks[i].context.rbx = 0;
            tasks[i].context.r12 = 0;
            tasks[i].context.r13 = 0;
            tasks[i].context.r14 = 0;
            tasks[i].context.r15 = 0;
            
            tasks[i].entry = entry;
            tasks[i].state = TASK_STATE_READY;
            
            /* Copy name */
            if (name) {
                u32 j;
                for (j = 0; j < 31 && name[j]; j++) {
                    tasks[i].name[j] = name[j];
                }
                tasks[i].name[j] = '\0';
            }
            
            task_count++;
            return i;
        }
    }
    
    return -1;
}

/* Start the first task - called once from kernel_main */
void start_first_task(void) {
    u32 i;
    
    /* Find first ready task */
    for (i = 0; i < MAX_TASKS; i++) {
        if (tasks[i].state == TASK_STATE_READY) {
            tasks[i].state = TASK_STATE_RUNNING;
            current_task = i;
            
            /* Call the task entry directly */
            if (tasks[i].entry) {
                tasks[i].entry();
            }
            return;
        }
    }
}

/* Select and switch to next ready task */
void schedule(void) {
    u32 next, i, prev;
    
    if (task_count == 0) {
        return;
    }
    
    prev = current_task;
    next = (current_task + 1) % MAX_TASKS;
    
    /* Find next ready task */
    for (i = 0; i < MAX_TASKS; i++) {
        if (tasks[next].state == TASK_STATE_READY) {
            /* Found ready task */
            if (tasks[prev].state == TASK_STATE_RUNNING) {
                tasks[prev].state = TASK_STATE_READY;
            }
            
            tasks[next].state = TASK_STATE_RUNNING;
            current_task = next;
            
            context_switch(&tasks[prev].context, &tasks[next].context);
            return;
        }
        next = (next + 1) % MAX_TASKS;
    }
    
    /* If only current task, just return */
}

/* Yield CPU to next task */
void yield(void) {
    schedule();
}

/* Terminate current task */
void task_exit(void) {
    tasks[current_task].state = TASK_STATE_TERMINATED;
    
    /* Free stack if allocated */
    if (tasks[current_task].stack) {
        pmm_free_page((u64)tasks[current_task].stack);
        tasks[current_task].stack = 0;
    }
    
    task_count--;
    
    /* Schedule next task */
    schedule();
    
    /* Should never reach here */
    while (1) {
        __asm__ volatile ("hlt");
    }
}

/* Get ID of currently running task */
u32 get_current_task_id(void) {
    return current_task;
}
