;; STUNIR WebAssembly Module
;; Module: module
;; Schema: stunir.wasm.v1
;; Epoch: 1769856341

(module
  ;; Memory: 1 pages (64KB each)
  (memory (export "memory") 1)

  (func $parse_heartbeat 
  )
  (func $send_heartbeat 
  )
  (func $init_mavlink 
  )
  (func $arm_vehicle 
  )
  (func $disarm_vehicle 
  )
  (func $set_mode 
  )
  (func $send_takeoff_cmd 
  )
  (func $send_land_cmd 
  )
  (func $get_position 
  )
  (func $get_battery_status 
  )
  (func $request_heartbeat 
  )
  (export "parse_heartbeat" (func $parse_heartbeat))
  (export "send_heartbeat" (func $send_heartbeat))
  (export "init_mavlink" (func $init_mavlink))
  (export "arm_vehicle" (func $arm_vehicle))
  (export "disarm_vehicle" (func $disarm_vehicle))
  (export "set_mode" (func $set_mode))
  (export "send_takeoff_cmd" (func $send_takeoff_cmd))
  (export "send_land_cmd" (func $send_land_cmd))
  (export "get_position" (func $get_position))
  (export "get_battery_status" (func $get_battery_status))
  (export "request_heartbeat" (func $request_heartbeat))
)