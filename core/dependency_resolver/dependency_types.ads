-------------------------------------------------------------------------------
--  STUNIR Dependency Types - Ada SPARK Specification
--  Part of Phase 2 SPARK Migration
--
--  This package provides dependency management types.
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

with Stunir_Strings; use Stunir_Strings;
with Stunir_Hashes;  use Stunir_Hashes;

package Dependency_Types is

   --  Maximum dependencies
   Max_Dependencies : constant := 128;

   --  Dependency status
   type Dependency_Status is (
      Dep_Unknown,
      Dep_Accepted,
      Dep_Rejected,
      Dep_Not_Found,
      Dep_Version_Mismatch,
      Dep_Hash_Mismatch
   );

   --  Dependency kind
   type Dependency_Kind is (
      Dep_Tool,       --  Binary tool (e.g., python3)
      Dep_Library,    --  Shared library
      Dep_Module,     --  Python/Node module
      Dep_File        --  Required file
   );

   --  Single dependency entry
   type Dependency_Entry is record
      Name            : Short_String := Empty_Short;
      Kind            : Dependency_Kind := Dep_Tool;
      Status          : Dependency_Status := Dep_Unknown;
      Resolved_Path   : Path_String := Empty_Path;
      Expected_Hash   : Hash_Hex := Zero_Hash;
      Actual_Hash     : Hash_Hex := Zero_Hash;
      Version_Spec    : Medium_String := Empty_Medium;
      Actual_Version  : Medium_String := Empty_Medium;
      Is_Optional     : Boolean := False;
      Is_Accepted     : Boolean := False;
   end record;

   --  Empty dependency
   Empty_Dependency : constant Dependency_Entry := (
      Name            => Empty_Short,
      Kind            => Dep_Tool,
      Status          => Dep_Unknown,
      Resolved_Path   => Empty_Path,
      Expected_Hash   => Zero_Hash,
      Actual_Hash     => Zero_Hash,
      Version_Spec    => Empty_Medium,
      Actual_Version  => Empty_Medium,
      Is_Optional     => False,
      Is_Accepted     => False
   );

   --  Dependency array
   type Dependency_Array is array (Positive range <>) of Dependency_Entry;
   subtype Dependency_Vector is Dependency_Array (1 .. Max_Dependencies);

   --  Dependency registry
   type Dependency_Registry is record
      Entries        : Dependency_Vector := (others => Empty_Dependency);
      Count          : Natural := 0;
      Accepted_Count : Natural := 0;
      Rejected_Count : Natural := 0;
   end record;

   --  Empty registry
   Empty_Dep_Registry : constant Dependency_Registry := (
      Entries        => (others => Empty_Dependency),
      Count          => 0,
      Accepted_Count => 0,
      Rejected_Count => 0
   );

   --  Acceptance receipt (for dep_receipt_tool compatibility)
   type Acceptance_Receipt is record
      Receipt_Type    : Short_String := Empty_Short;
      Accepted        : Boolean := False;
      Tool_Name       : Short_String := Empty_Short;
      Resolved_Path   : Path_String := Empty_Path;
      Resolved_Abs    : Path_String := Empty_Path;
   end record;

   --  Check if dependency is resolved
   function Is_Resolved (Dep : Dependency_Entry) return Boolean is
     (Dep.Status /= Dep_Unknown and Dep.Status /= Dep_Not_Found);

   --  Check if all required dependencies are accepted
   function All_Required_Accepted (Reg : Dependency_Registry) return Boolean;

   --  Status to string
   function Status_To_String (S : Dependency_Status) return String;

end Dependency_Types;
