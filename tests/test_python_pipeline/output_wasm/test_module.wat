;; STUNIR: wasm emission (raw target) in WAT (.wat)
;; module: test_module

(module
  (memory (export "memory") 1)

  (func $add (export "add") (param $a i32) (param $b i32) (result i32)
    (i32.const 0)
  )

  (func $greet (export "greet") (param $name i32) (result i32)
    (i32.const 0)
  )

)
