;; STUNIR: wasm emission (raw target) in WAT (.wat)
;; module: mavlink_handler

(module
  (memory (export "memory") 1)

  (func $parse_heartbeat (export "parse_heartbeat") (param $buffer i32) (param $len i32) (result i32)
    (i32.const 0)
  )

  (func $send_heartbeat (export "send_heartbeat") (param $sys_id i32) (param $comp_id i32) (result i32)
    (i32.const 0)
  )

  (func $init_mavlink (export "init_mavlink") (param $port i32) (result i32)
    (i32.const 0)
  )

  (func $arm_vehicle (export "arm_vehicle") (param $sysid i32) (param $compid i32) (result i32)
    (i32.const 0)
  )

  (func $disarm_vehicle (export "disarm_vehicle") (param $sysid i32) (param $compid i32) (result i32)
    (i32.const 0)
  )

  (func $set_mode (export "set_mode") (param $sysid i32) (param $mode i32) (result i32)
    (i32.const 0)
  )

  (func $send_takeoff_cmd (export "send_takeoff_cmd") (param $sysid i32) (param $altitude i32) (result i32)
    (i32.const 0)
  )

  (func $send_land_cmd (export "send_land_cmd") (param $sysid i32) (result i32)
    (i32.const 0)
  )

  (func $get_position (export "get_position") (param $sysid i32) (result i32)
    (i32.const 0)
  )

  (func $get_battery_status (export "get_battery_status") (param $sysid i32) (result i32)
    (i32.const 0)
  )

  (func $request_heartbeat (export "request_heartbeat") (param $sysid i32) (param $compid i32) (result i32)
    (i32.const 0)
  )

)
