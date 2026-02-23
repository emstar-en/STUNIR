--  Golden Test File for SPARK Extractor
--  This file contains simple, parseable SPARK/Ada signatures
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

package body Golden_Test is

   --  Simple procedure with no parameters
   procedure Initialize is
   begin
      null;
   end Initialize;

   --  Simple function with no parameters
   function Get_Version return String is
   begin
      return "1.0.0";
   end Get_Version;

   --  Procedure with parameters
   procedure Set_Value (Val : in Integer) is
   begin
      null;
   end Set_Value;

   --  Function with parameters
   function Add (A : Integer; B : Integer) return Integer is
   begin
      return A + B;
   end Add;

   --  Function with multiple parameter types
   function Compute (X : Float; Y : Float; Mode : Integer) return Float is
   begin
      return X + Y;
   end Compute;

end Golden_Test;
