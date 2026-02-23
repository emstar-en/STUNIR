--  Test: Implementation Body
--  Purpose: Test extraction from body files with implementation details
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (Off);  --  Body often has SPARK_Mode Off

package body Implementation_Body is

   --  Local procedure (not in spec)
   procedure Local_Helper (X : in out Integer) is
   begin
      X := X + 1;
   end Local_Helper;

   --  Public procedure implementation
   procedure Public_Proc (Value : in Integer; Result : out Integer) is
      Local_Var : Integer := 0;
   begin
      Local_Helper (Local_Var);
      Result := Value + Local_Var;
   end Public_Proc;

   --  Public function implementation
   function Public_Func (A : Float; B : Float) return Float is
   begin
      return A + B;
   end Public_Func;

   --  Function with local declarations
   function Complex_Func
     (Input : String;
      Len   : Integer) return Boolean
   is
      Temp : String (1 .. Len);
      Valid : Boolean := True;
   begin
      if Input'Length < Len then
         Valid := False;
      end if;
      return Valid;
   end Complex_Func;

end Implementation_Body;
