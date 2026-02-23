--  Test: Multiline Signatures
--  Purpose: Test extraction of signatures split across multiple lines
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

package Multiline_Signatures is

   --  Function with params split across lines
   function Compute_Average
     (First_Value  : in Float;
      Second_Value : in Float;
      Third_Value  : in Float) return Float;

   --  Procedure with multiline params
   procedure Process_Data
     (Input_Buffer  : in  String;
      Output_Buffer : out String;
      Status        : out Integer);

   --  Function with long return type
   function Transform_Matrix
     (Matrix : in  Float_Array;
      Scale  : in  Float) return Float_Array;

   --  Generic-looking but not generic
   function Swap_Values
     (Left  : in out Integer;
      Right : in out Integer) return Boolean;

end Multiline_Signatures;
