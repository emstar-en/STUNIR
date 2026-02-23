--  Emit Target C Family - C/C++ Code Emission
--  Phase: 3 (Emit)
--  SPARK_Mode: On
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

with STUNIR_Types;
use STUNIR_Types;

package Emit_Target.C_Family is

   --  Emit steps for C/C++ targets
   procedure Emit_Steps_C (Func : IR_Function; Append_Line : not null access procedure (Text : String));

   --  Check if step index is inside a nested block
   function Is_In_Nested_Block (Func : IR_Function; Idx : Step_Index) return Boolean;

end Emit_Target.C_Family;
