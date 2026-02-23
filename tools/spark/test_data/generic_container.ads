--  Test: Generics
--  Purpose: Test extraction of generic declarations
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

generic
   type Element_Type is private;
   Initial_Size : Positive := 100;
package Generic_Container is

   --  Generic function
   function Get_Size return Natural;

   --  Generic procedure
   procedure Clear;

   --  Generic function with type parameter
   function Contains (Item : Element_Type) return Boolean;

   --  Generic procedure with multiple params
   procedure Add
     (Item     : in  Element_Type;
      Success  : out Boolean);

private
   --  Private declarations should not be extracted
   type Internal_Type is new Integer;

end Generic_Container;
