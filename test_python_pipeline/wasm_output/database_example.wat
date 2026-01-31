;; STUNIR: wasm emission (raw target) in WAT (.wat)
;; module: database_example

(module
  (memory (export "memory") 1)

  (func $connect (export "connect") (param $host i32) (param $port i32) (result i32)
    (i32.const 0)
  )

  (func $execute_query (export "execute_query") (param $conn_id i32) (param $query i32) (result i32)
    (i32.const 0)
  )

  (func $close (export "close") (param $conn_id i32) (result i32)
    (i32.const 0)
  )

  (func $map (export "map") (param $func i32) (param $list i32) (result i32)
    (i32.const 0)
  )

  (func $filter (export "filter") (param $predicate i32) (param $list i32) (result i32)
    (i32.const 0)
  )

  (func $reduce (export "reduce") (param $func i32) (param $list i32) (param $initial i32) (result i32)
    (i32.const 0)
  )

  (func $vector_add_kernel (export "vector_add_kernel") (param $a i32) (param $b i32) (param $c i32) (param $n i32)
  )

  (func $matrix_mul_kernel (export "matrix_mul_kernel") (param $a i32) (param $b i32) (param $c i32) (param $width i32)
  )

  (func $matrix_multiply (export "matrix_multiply") (param $a i32) (param $b i32) (param $n i32) (result i32)
    (i32.const 0)
  )

  (func $vector_dot_product (export "vector_dot_product") (param $v1 i32) (param $v2 i32) (param $len i32) (result i32)
    (i32.const 0)
  )

  (func $matrix_transpose (export "matrix_transpose") (param $matrix i32) (param $rows i32) (param $cols i32) (result i32)
    (i32.const 0)
  )

  (func $add (export "add") (param $a i32) (param $b i32) (result i32)
    (i32.const 0)
  )

  (func $multiply (export "multiply") (param $x i32) (param $y i32) (result i32)
    (i32.const 0)
  )

  (func $get_user (export "get_user") (param $user_id i32) (result i32)
    (i32.const 0)
  )

  (func $create_user (export "create_user") (param $name i32) (param $email i32) (result i32)
    (i32.const 0)
  )

  (func $update_user (export "update_user") (param $user_id i32) (param $data i32) (result i32)
    (i32.const 0)
  )

  (func $delete_user (export "delete_user") (param $user_id i32) (result i32)
    (i32.const 0)
  )

)
