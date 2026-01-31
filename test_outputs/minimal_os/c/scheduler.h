/* STUNIR Generated Code - Cooperative Scheduler Header
 * Module: scheduler
 * Architecture: x86_64
 * DO-178C Level A Compliance
 */

#ifndef STUNIR_SCHEDULER_H
#define STUNIR_SCHEDULER_H

#include "types.h"

extern u32 scheduler_running;

void scheduler_init(void);
i32 task_create(void (*entry)(void), const char* name);
void schedule(void);
void start_first_task(void);
void yield(void);
void task_exit(void);
u32 get_current_task_id(void);

#endif /* STUNIR_SCHEDULER_H */
