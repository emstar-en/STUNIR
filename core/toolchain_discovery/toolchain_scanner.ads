-------------------------------------------------------------------------------
--  STUNIR Toolchain Scanner - Ada SPARK Specification
--  Part of Phase 2 SPARK Migration
--
--  This package provides host toolchain scanning and lockfile generation.
--  Migrated from: scripts/discover_toolchain.sh
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

with Stunir_Strings;   use Stunir_Strings;
with Stunir_Hashes;    use Stunir_Hashes;
with Toolchain_Types;  use Toolchain_Types;

package Toolchain_Scanner is

   --  Built-in tool definitions
   --  These mirror the TOOLS array in discover_toolchain.sh
   type Builtin_Tool is (
      Tool_Python,
      Tool_Bash,
      Tool_Git,
      Tool_CC,
      Tool_Rustc,
      Tool_Cargo,
      Tool_Stunir_Native
   );

   --  Get builtin tool definition
   function Get_Builtin_Definition (T : Builtin_Tool) return Tool_Entry;

   --  Initialize registry with builtin tool definitions
   procedure Initialize_Registry (
      Reg : out Tool_Registry)
     with
       Post => Reg.Count >= 2;  --  At least python and bash

   --  Add custom tool to registry
   procedure Add_Tool (
      Reg          : in out Tool_Registry;
      Logical_Name : Short_String;
      Binary_Name  : Short_String;
      Requirement  : Tool_Requirement;
      Success      : out Boolean)
     with
       Pre  => Reg.Count < Max_Tools,
       Post => (not Success or else Reg.Count >= 1);

   --  Resolve a single tool (find in PATH, compute hash)
   procedure Resolve_Tool (
      Tool_Ent : in out Tool_Entry;
      Success : out Boolean)
     with
       Post => (not Success or else Tool_Ent.Status = Status_Resolved);

   --  Resolve all tools in registry
   procedure Resolve_All (
      Reg     : in out Tool_Registry;
      Success : out Boolean)
     with
       Post => (not Success or else All_Required_Resolved (Reg));

   --  Get platform information
   procedure Get_Platform_Info (
      Info : out Platform_Info);

   --  Generate toolchain lockfile
   procedure Generate_Lockfile (
      Reg      : Tool_Registry;
      Platform : Platform_Info;
      Lockfile : out Toolchain_Lockfile)
     with
       Post => Lockfile.Is_Valid = All_Required_Resolved (Reg);

   --  Scan entire toolchain (convenience procedure)
   procedure Scan_Toolchain (
      Lockfile : out Toolchain_Lockfile;
      Success  : out Boolean)
     with
       Post => (not Success or else Lockfile.Is_Valid);

   --  Find binary in PATH
   procedure Find_In_Path (
      Binary_Name   : Short_String;
      Resolved_Path : out Path_String;
      Found         : out Boolean);

   --  Compute file SHA256 hash
   procedure Compute_File_Hash (
      File_Path : Path_String;
      Hash      : out Hash_Hex;
      Success   : out Boolean)
     with
       Pre => File_Path.Length > 0;

   --  Get binary version string
   procedure Get_Version_String (
      Tool_Path : Path_String;
      Binary    : Short_String;
      Version   : out Medium_String);

end Toolchain_Scanner;
