--  Test: Implementation Body Spec
--  Purpose: Spec for implementation_body.adb
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

package Implementation_Body is

   --  Public procedure
   procedure Public_Proc (Value : in Integer; Result : out Integer);

   --  Public function
   function Public_Func (A : Float; B : Float) return Float;

   --  Complex function with string params
   function Complex_Func
     (Input : String;
      Len   : Integer) return Boolean;

end Implementation_Body;
