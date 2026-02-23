--  Test: Edge Cases
--  Purpose: Test extraction with edge cases (empty, comments, malformed)
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

package Edge_Cases is

   --  Empty body marker
   procedure Do_Nothing;

   --  Comment-only section
   --  This is just a comment
   --  Another comment
   --  No actual code here

   --  Function with no params and simple return
   function Get_True return Boolean;

   --  Function with complex param names
   procedure Process
     (Input_Data_Buffer    : in  String;
      Output_Result_Buffer : out String;
      Processing_Status    : out Integer);

   --  Malformed-looking but valid signature
   function Weird_But_Valid
     (A : Integer;
      B : Float;
      C : String) return Boolean;

   --  Very long function name with underscores
   function This_Is_A_Very_Long_Function_Name_With_Many_Underscores
     (Param_With_Long_Name : Integer) return Integer;

   --  Procedure with mode indicators
   procedure All_Modes
     (In_Param    : in     Integer;
      Out_Param   :    out Integer;
      InOut_Param : in out Integer);

end Edge_Cases;
