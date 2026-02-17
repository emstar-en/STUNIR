@ STUNIR Generated ARM Assembly
@ Module: module
@ Syntax: arm

section .data

section .text

@ Function: parse_heartbeat
global parse_heartbeat
parse_heartbeat:
    push {fp, lr}
    mov fp, sp
    mov sp, fp
    pop {fp, lr}
    bx lr

@ Function: send_heartbeat
global send_heartbeat
send_heartbeat:
    push {fp, lr}
    mov fp, sp
    mov sp, fp
    pop {fp, lr}
    bx lr

@ Function: init_mavlink
global init_mavlink
init_mavlink:
    push {fp, lr}
    mov fp, sp
    mov sp, fp
    pop {fp, lr}
    bx lr

@ Function: arm_vehicle
global arm_vehicle
arm_vehicle:
    push {fp, lr}
    mov fp, sp
    mov sp, fp
    pop {fp, lr}
    bx lr

@ Function: disarm_vehicle
global disarm_vehicle
disarm_vehicle:
    push {fp, lr}
    mov fp, sp
    mov sp, fp
    pop {fp, lr}
    bx lr

@ Function: set_mode
global set_mode
set_mode:
    push {fp, lr}
    mov fp, sp
    mov sp, fp
    pop {fp, lr}
    bx lr

@ Function: send_takeoff_cmd
global send_takeoff_cmd
send_takeoff_cmd:
    push {fp, lr}
    mov fp, sp
    mov sp, fp
    pop {fp, lr}
    bx lr

@ Function: send_land_cmd
global send_land_cmd
send_land_cmd:
    push {fp, lr}
    mov fp, sp
    mov sp, fp
    pop {fp, lr}
    bx lr

@ Function: get_position
global get_position
get_position:
    push {fp, lr}
    mov fp, sp
    mov sp, fp
    pop {fp, lr}
    bx lr

@ Function: get_battery_status
global get_battery_status
get_battery_status:
    push {fp, lr}
    mov fp, sp
    mov sp, fp
    pop {fp, lr}
    bx lr

@ Function: request_heartbeat
global request_heartbeat
request_heartbeat:
    push {fp, lr}
    mov fp, sp
    mov sp, fp
    pop {fp, lr}
    bx lr
