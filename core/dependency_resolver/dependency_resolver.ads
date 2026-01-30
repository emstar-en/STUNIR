-------------------------------------------------------------------------------
--  STUNIR Dependency Resolver - Ada SPARK Specification
--  Part of Phase 2 SPARK Migration
--
--  This package provides dependency resolution logic.
--  Migrated from: tools/dep_receipt_tool.py
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

with Stunir_Strings;    use Stunir_Strings;
with Stunir_Hashes;     use Stunir_Hashes;
with Dependency_Types;  use Dependency_Types;

package Dependency_Resolver is

   --  Initialize dependency registry
   procedure Initialize_Registry (
      Reg : out Dependency_Registry)
     with
       Post => Reg.Count = 0;

   --  Add dependency to registry
   procedure Add_Dependency (
      Reg       : in out Dependency_Registry;
      Name      : Short_String;
      Kind      : Dependency_Kind;
      Optional  : Boolean := False;
      Success   : out Boolean)
     with
       Pre  => Reg.Count < Max_Dependencies;

   --  Resolve a single dependency
   procedure Resolve_Dependency (
      Dep     : in out Dependency_Entry;
      Success : out Boolean);

   --  Resolve all dependencies in registry
   procedure Resolve_All (
      Reg     : in out Dependency_Registry;
      Success : out Boolean);

   --  Parse acceptance receipt from JSON path
   procedure Parse_Acceptance_Receipt (
      Receipt_Path : Path_String;
      Receipt      : out Acceptance_Receipt;
      Success      : out Boolean)
     with
       Pre => Receipt_Path.Length > 0;

   --  Check if receipt indicates acceptance
   function Is_Receipt_Accepted (Receipt : Acceptance_Receipt) return Boolean;

   --  Get tool path from receipt
   function Get_Receipt_Tool_Path (
      Receipt : Acceptance_Receipt) return Path_String;

   --  Verify dependency hash
   procedure Verify_Hash (
      Dep     : in out Dependency_Entry;
      Matches : out Boolean);

   --  Verify dependency version
   procedure Verify_Version (
      Dep     : in out Dependency_Entry;
      Matches : out Boolean);

   --  Find dependency by name
   function Find_Dependency (
      Reg  : Dependency_Registry;
      Name : Short_String) return Natural;

   --  Get dependency by index
   function Get_Dependency (
      Reg   : Dependency_Registry;
      Index : Positive) return Dependency_Entry
     with
       Pre => Index <= Reg.Count;

end Dependency_Resolver;
