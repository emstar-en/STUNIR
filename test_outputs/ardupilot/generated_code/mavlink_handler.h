/* STUNIR Embedded Module: mavlink_handler */
/* Architecture: arm */
/* Epoch: 1769791517 */

#ifndef MAVLINK_HANDLER_H
#define MAVLINK_HANDLER_H

#include <stdint.h>

/* Function prototypes */
int32_t parse_heartbeat(int32_t buffer, uint8_t len);
int32_t send_heartbeat(uint8_t sys_id, uint8_t comp_id);

#endif /* MAVLINK_HANDLER_H */