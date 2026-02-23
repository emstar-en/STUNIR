--  Test: Nested Packages
--  Purpose: Test extraction of nested package declarations
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

package Nested_Packages is

   --  Top-level subprograms
   procedure Initialize;
   function Get_Status return Integer;

   --  Nested package
   package Inner is
      procedure Inner_Proc (X : in Integer);
      function Inner_Func return Boolean;
      
      --  Deeper nested package
      package Deeper is
         procedure Deep_Proc;
         function Deep_Func (V : Float) return Float;
      end Deeper;
   end Inner;

   --  Another nested package
   package Utils is
      function Clamp (Value : Integer; Min : Integer; Max : Integer) return Integer;
      procedure Swap (A : in out Integer; B : in out Integer);
   end Utils;

private
   --  Private nested package
   package Internal is
      procedure Internal_Proc;
      function Internal_Func return Integer;
   end Internal;

end Nested_Packages;
