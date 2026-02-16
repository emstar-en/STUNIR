-------------------------------------------------------------------------------
--  STUNIR Toolchain Types - Ada SPARK Specification
--  Part of Phase 2 SPARK Migration
--
--  This package provides toolchain-related types for build configuration.
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

with Stunir_Strings; use Stunir_Strings;
with Stunir_Hashes;  use Stunir_Hashes;

package Toolchain_Types is

   --  Maximum number of tools in registry
   Max_Tools : constant := 32;

   --  Tool requirement type
   type Tool_Requirement is (Required, Optional);

   --  Tool resolution status
   type Tool_Status is (
      Status_Unknown,
      Status_Resolved,
      Status_Not_Found,
      Status_Hash_Mismatch,
      Status_Version_Mismatch
   );

   --  Single tool entry
   type Tool_Entry is record
      Logical_Name   : Short_String := Empty_Short;
      Binary_Name    : Short_String := Empty_Short;
      Resolved_Path  : Path_String := Empty_Path;
      SHA256_Hash    : Hash_Hex := Zero_Hash;
      Version_String : Medium_String := Empty_Medium;
      Requirement    : Tool_Requirement := Optional;
      Status         : Tool_Status := Status_Unknown;
   end record;

   --  Default (empty) tool entry
   Empty_Tool_Entry : constant Tool_Entry := (
      Logical_Name   => Empty_Short,
      Binary_Name    => Empty_Short,
      Resolved_Path  => Empty_Path,
      SHA256_Hash    => Zero_Hash,
      Version_String => Empty_Medium,
      Requirement    => Optional,
      Status         => Status_Unknown
   );

   --  Tool array for registry
   type Tool_Array is array (Positive range <>) of Tool_Entry;
   subtype Tool_Vector is Tool_Array (1 .. Max_Tools);

   --  Tool registry
   type Tool_Registry is record
      Entries        : Tool_Vector := (others => Empty_Tool_Entry);
      Count          : Natural := 0;
      Required_Count : Natural := 0;
      Resolved_Count : Natural := 0;
   end record;

   --  Empty registry
   Empty_Registry : constant Tool_Registry := (
      Entries        => (others => Empty_Tool_Entry),
      Count          => 0,
      Required_Count => 0,
      Resolved_Count => 0
   );

   --  Host platform info
   type Platform_Info is record
      OS_Name      : Short_String := Empty_Short;
      Architecture : Short_String := Empty_Short;
   end record;

   --  Lockfile data structure
   type Toolchain_Lockfile is record
      Registry      : Tool_Registry := Empty_Registry;
      Platform      : Platform_Info;
      Is_Valid      : Boolean := False;
   end record;

   --  Check if tool is resolved
   function Is_Resolved (E : Tool_Entry) return Boolean is
     (E.Status = Status_Resolved);

   --  Check if all required tools are resolved
   function All_Required_Resolved (Reg : Tool_Registry) return Boolean is
     (Reg.Required_Count <= Reg.Resolved_Count);

   --  Get registry entry by index
   function Get_Entry (
      Reg   : Tool_Registry;
      Index : Positive) return Tool_Entry
     with Pre => Index <= Reg.Count;

   --  Find tool by logical name
   function Find_Tool (
      Reg  : Tool_Registry;
      Name : Short_String) return Natural;

end Toolchain_Types;
