--  Emit Target Lisp Family - Lisp Dialect Code Emission
--  Phase: 3 (Emit)
--  SPARK_Mode: On
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

with STUNIR_Types;
use STUNIR_Types;

package Emit_Target.Lisp_Family is

   --  Emit steps for Clojure/ClojureScript
   procedure Emit_Steps_Clojure (Func : IR_Function; Append_Line : not null access procedure (Text : String));

   --  Emit steps for Common Lisp
   procedure Emit_Steps_Common_Lisp (Func : IR_Function; Append_Line : not null access procedure (Text : String));

   --  Emit steps for Scheme/Racket/Guile
   procedure Emit_Steps_Scheme (Func : IR_Function; Append_Line : not null access procedure (Text : String));

   --  Check if step index is inside a nested block
   function Is_In_Nested_Block (Func : IR_Function; Idx : Step_Index) return Boolean;

end Emit_Target.Lisp_Family;
