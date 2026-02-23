--  Emit Target Prolog Family - Prolog Dialect Code Emission
--  Phase: 3 (Emit)
--  SPARK_Mode: On
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

with STUNIR_Types;
use STUNIR_Types;

package Emit_Target.Prolog_Family is

   --  Emit steps for SWI-Prolog
   procedure Emit_Steps_SWI_Prolog
     (Func        : IR_Function;
      Append_Line : not null access procedure (Text : String));

   --  Emit steps for GNU Prolog
   procedure Emit_Steps_GNU_Prolog
     (Func        : IR_Function;
      Append_Line : not null access procedure (Text : String));

   --  Emit steps for Mercury
   procedure Emit_Steps_Mercury
     (Func        : IR_Function;
      Append_Line : not null access procedure (Text : String));

end Emit_Target.Prolog_Family;