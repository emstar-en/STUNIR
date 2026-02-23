--  Emit Target Functional Formal - Functional/Formal Language Code Emission
--  Phase: 3 (Emit)
--  SPARK_Mode: On
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

with STUNIR_Types;
use STUNIR_Types;

package Emit_Target.Functional_Formal is

   --  Emit steps for Futhark
   procedure Emit_Steps_Futhark
     (Func        : IR_Function;
      Append_Line : not null access procedure (Text : String));

   --  Emit steps for Lean4
   procedure Emit_Steps_Lean4
     (Func        : IR_Function;
      Append_Line : not null access procedure (Text : String));

   --  Emit steps for Haskell
   procedure Emit_Steps_Haskell
     (Func        : IR_Function;
      Append_Line : not null access procedure (Text : String));

end Emit_Target.Functional_Formal;