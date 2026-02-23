--  Test: Overloaded Functions
--  Purpose: Test extraction of overloaded subprograms
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

package Overloaded_Functions is

   --  Overloaded Process - Integer version
   procedure Process (Value : in Integer);

   --  Overloaded Process - Float version
   procedure Process (Value : in Float);

   --  Overloaded Process - String version
   procedure Process (Value : in String);

   --  Overloaded Compute - no params
   function Compute return Integer;

   --  Overloaded Compute - one param
   function Compute (X : Integer) return Integer;

   --  Overloaded Compute - two params
   function Compute (X : Integer; Y : Integer) return Integer;

   --  Overloaded Convert - different return types
   function Convert (S : String) return Integer;
   function Convert (S : String) return Float;
   function Convert (S : String) return Boolean;

end Overloaded_Functions;
