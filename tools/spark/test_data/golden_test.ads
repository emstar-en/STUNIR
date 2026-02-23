--  Golden Test Spec for SPARK Extractor
--  This file contains simple, parseable SPARK/Ada signatures
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

package Golden_Test is

   --  Simple procedure with no parameters
   procedure Initialize;

   --  Simple function with no parameters
   function Get_Version return String;

   --  Procedure with parameters
   procedure Set_Value (Val : in Integer);

   --  Function with parameters
   function Add (A : Integer; B : Integer) return Integer;

   --  Function with multiple parameter types
   function Compute (X : Float; Y : Float; Mode : Integer) return Float;

end Golden_Test;
