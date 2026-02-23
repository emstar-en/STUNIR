--  Emit Target Mainstream - Mainstream Language Code Emission
--  Phase: 3 (Emit)
--  SPARK_Mode: On
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

with STUNIR_Types;
use STUNIR_Types;

package Emit_Target.Mainstream is

   --  Emit steps for Rust
   procedure Emit_Steps_Rust
     (Func        : IR_Function;
      Append_Line : not null access procedure (Text : String));

   --  Emit steps for Python
   procedure Emit_Steps_Python
     (Func        : IR_Function;
      Append_Line : not null access procedure (Text : String));

   --  Emit steps for Go
   procedure Emit_Steps_Go
     (Func        : IR_Function;
      Append_Line : not null access procedure (Text : String));

   --  Emit steps for Java
   procedure Emit_Steps_Java
     (Func        : IR_Function;
      Append_Line : not null access procedure (Text : String));

   --  Emit steps for JavaScript
   procedure Emit_Steps_JavaScript
     (Func        : IR_Function;
      Append_Line : not null access procedure (Text : String));

   --  Emit steps for C#
   procedure Emit_Steps_CSharp
     (Func        : IR_Function;
      Append_Line : not null access procedure (Text : String));

   --  Emit steps for Swift
   procedure Emit_Steps_Swift
     (Func        : IR_Function;
      Append_Line : not null access procedure (Text : String));

   --  Emit steps for Kotlin
   procedure Emit_Steps_Kotlin
     (Func        : IR_Function;
      Append_Line : not null access procedure (Text : String));

   --  Emit steps for SPARK/Ada
   procedure Emit_Steps_SPARK
     (Func        : IR_Function;
      Append_Line : not null access procedure (Text : String));

   --  Emit steps for Ada (non-SPARK)
   procedure Emit_Steps_Ada
     (Func        : IR_Function;
      Append_Line : not null access procedure (Text : String));

end Emit_Target.Mainstream;