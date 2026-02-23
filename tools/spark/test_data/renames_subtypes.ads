--  Test: Renames and Subtypes
--  Purpose: Test extraction of renames and subtype declarations
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

package Renames_Subtypes is

   --  Subtype declarations
   subtype Index_Type is Integer range 1 .. 100;
   subtype Name_String is String (1 .. 64);
   subtype Count_Type is Natural;

   --  Renames declarations
   Max_Index : Index_Type renames Index_Type'Last;
   Min_Index : Index_Type renames Index_Type'First;

   --  Regular subprograms
   function Get_Max_Index return Index_Type;
   function Get_Min_Index return Index_Type;

   --  Procedure using subtypes
   procedure Set_Name
     (Name : out Name_String;
      Len  : out Count_Type);

   --  Function with subtype return
   function Create_Index (Value : Integer) return Index_Type;

   --  Generic renames (not supported, but test parsing)
   --  package Integer_IO renames Ada.Integer_IO;

end Renames_Subtypes;
