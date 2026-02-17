/* STUNIR Embedded Module: module */
/* Architecture: arm */
/* Epoch: 1769938963 */

#ifndef MODULE_H
#define MODULE_H

#include <stdint.h>

/* Function prototypes */
void parse_heartbeat(void);
void send_heartbeat(void);
void init_mavlink(void);
void arm_vehicle(void);
void disarm_vehicle(void);
void set_mode(void);
void send_takeoff_cmd(void);
void send_land_cmd(void);
void get_position(void);
void get_battery_status(void);
void request_heartbeat(void);

#endif /* MODULE_H */