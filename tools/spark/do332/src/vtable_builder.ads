--  STUNIR DO-332 VTable Builder Specification
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0
--
--  This package implements virtual method table construction
--  for DO-332 OO.3 dynamic dispatch analysis.

pragma SPARK_Mode (On);

with OOP_Types; use OOP_Types;
with OOP_Analysis; use OOP_Analysis;

package VTable_Builder is

   --  ============================================================
   --  Constants
   --  ============================================================

   Max_VTable_Entries : constant := 1_000;

   --  ============================================================
   --  VTable Structure
   --  ============================================================

   type VTable is record
      Class_ID     : OOP_Types.Class_ID;
      Entry_Count  : Natural;
      Has_Abstract : Boolean;
   end record;

   Null_VTable : constant VTable := (
      Class_ID     => Null_Class_ID,
      Entry_Count  => 0,
      Has_Abstract => False
   );

   type VTable_Set is array (Positive range <>) of VTable;

   --  ============================================================
   --  VTable Entry (extended)
   --  ============================================================

   type Extended_VTable_Entry is record
      Slot_Index      : Natural;
      Method_ID       : OOP_Types.Method_ID;
      Method_Name     : Name_String;
      Name_Length     : Natural;
      Declaring_Class : Class_ID;
      Impl_Class      : Class_ID;
      Is_Abstract     : Boolean;
      Is_Final        : Boolean;
      Is_Override     : Boolean;
   end record;

   Null_Extended_Entry : constant Extended_VTable_Entry := (
      Slot_Index      => 0,
      Method_ID       => Null_Method_ID,
      Method_Name     => (others => ' '),
      Name_Length     => 0,
      Declaring_Class => Null_Class_ID,
      Impl_Class      => Null_Class_ID,
      Is_Abstract     => False,
      Is_Final        => False,
      Is_Override     => False
   );

   type Extended_VTable_Array is array (Positive range <>) of Extended_VTable_Entry;

   --  ============================================================
   --  Core Functions
   --  ============================================================

   --  Build vtable for a single class
   function Build_VTable (
      Class   : Class_Info;
      Classes : Class_Array;
      Methods : Method_Array;
      Links   : Inheritance_Array
   ) return VTable
     with Pre  => Is_Valid_Class_ID (Class.ID),
          Post => Build_VTable'Result.Class_ID = Class.ID;

   --  Build vtables for all classes
   procedure Build_All_VTables (
      Classes  : in     Class_Array;
      Methods  : in     Method_Array;
      Links    : in     Inheritance_Array;
      VTables  :    out VTable_Set;
      Success  :    out Boolean
   ) with Pre => VTables'Length >= Classes'Length;

   --  Get vtable entries for a class
   function Get_VTable_Entries (
      Class   : Class_Info;
      Classes : Class_Array;
      Methods : Method_Array;
      Links   : Inheritance_Array
   ) return Extended_VTable_Array;

   --  ============================================================
   --  VTable Analysis Functions
   --  ============================================================

   --  Count virtual methods in vtable
   function Count_Virtual_Slots (
      VT : VTable
   ) return Natural is (VT.Entry_Count);

   --  Check if vtable has unimplemented abstract methods
   function Has_Unimplemented (
      VT : VTable
   ) return Boolean is (VT.Has_Abstract);

   --  Find slot for a method in vtable
   function Find_Slot (
      Method_Name : String;
      Entries     : Extended_VTable_Array
   ) return Natural;

   --  ============================================================
   --  Utility Functions
   --  ============================================================

   --  Merge parent vtable with child overrides
   function Merge_VTable (
      Parent_Entries : Extended_VTable_Array;
      Child_Methods  : Method_Array;
      Child_Class    : Class_ID
   ) return Extended_VTable_Array;

   --  Get inherited virtual methods
   function Get_Inherited_Virtuals (
      Class   : Class_ID;
      Classes : Class_Array;
      Methods : Method_Array;
      Links   : Inheritance_Array
   ) return Method_ID_Array;

end VTable_Builder;
