-------------------------------------------------------------------------------
--  STUNIR Type Registry - Ada SPARK Specification
--  Part of Phase 1 SPARK Migration
--
--  This package provides a registry for managing named types.
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

with Stunir_Types; use Stunir_Types;

package Stunir_Type_Registry is

   --  Maximum number of types in the registry
   Max_Registry_Types : constant := 1024;

   --  Type ID for registry lookups
   type Type_Id is range 0 .. Max_Registry_Types - 1;
   No_Type_Id : constant Type_Id := 0;

   --  Registry entry
   type Registry_Entry is record
      Name    : Type_Name := Empty_Type_Name;
      T       : STUNIR_Type := Void_Type;
      Is_Used : Boolean := False;
   end record;

   --  Type Registry (non-limited for SPARK 'Old support)
   type Type_Registry is private;

   --  Initialize a new registry with built-in types
   procedure Initialize (Reg : out Type_Registry)
     with
       Post => Get_Count (Reg) >= 15;  --  At least built-in types

   --  Register a named type
   procedure Register (
      Reg  : in out Type_Registry;
      Name : Type_Name;
      T    : STUNIR_Type;
      Id   : out Type_Id)
     with
       Pre  => Name.Length > 0 and Get_Count (Reg) < Max_Registry_Types,
       Post => Id /= No_Type_Id or else Get_Count (Reg) = Get_Count (Reg'Old);

   --  Look up a type by name
   function Lookup (
      Reg  : Type_Registry;
      Name : Type_Name) return Type_Id;

   --  Get a type by ID
   function Get_Type (
      Reg : Type_Registry;
      Id  : Type_Id) return STUNIR_Type
     with
       Pre => Id /= No_Type_Id and Has_Type (Reg, Id);

   --  Check if a type ID exists
   function Has_Type (
      Reg : Type_Registry;
      Id  : Type_Id) return Boolean;

   --  Check if a name exists
   function Has_Name (
      Reg  : Type_Registry;
      Name : Type_Name) return Boolean;

   --  Get the number of registered types
   function Get_Count (Reg : Type_Registry) return Natural;

   --  Built-in type IDs (available after initialization)
   Void_Type_Id   : constant Type_Id := 1;
   Unit_Type_Id   : constant Type_Id := 2;
   Bool_Type_Id   : constant Type_Id := 3;
   I8_Type_Id     : constant Type_Id := 4;
   I16_Type_Id    : constant Type_Id := 5;
   I32_Type_Id    : constant Type_Id := 6;
   I64_Type_Id    : constant Type_Id := 7;
   U8_Type_Id     : constant Type_Id := 8;
   U16_Type_Id    : constant Type_Id := 9;
   U32_Type_Id    : constant Type_Id := 10;
   U64_Type_Id    : constant Type_Id := 11;
   F32_Type_Id    : constant Type_Id := 12;
   F64_Type_Id    : constant Type_Id := 13;
   Char_Type_Id   : constant Type_Id := 14;
   String_Type_Id : constant Type_Id := 15;

private

   type Entry_Array is array (Type_Id range 1 .. Max_Registry_Types - 1) of Registry_Entry;

   type Type_Registry is record
      Entries : Entry_Array;
      Count   : Natural := 0;
   end record;

end Stunir_Type_Registry;
