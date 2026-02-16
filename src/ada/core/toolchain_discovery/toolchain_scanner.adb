-------------------------------------------------------------------------------
--  STUNIR Toolchain Scanner - Ada SPARK Implementation
--  Part of Phase 2 SPARK Migration
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

package body Toolchain_Scanner is

   --  Get builtin tool definition
   function Get_Builtin_Definition (T : Builtin_Tool) return Tool_Entry is
      Result : Tool_Entry := Empty_Tool_Entry;
   begin
      case T is
         when Tool_Python =>
            Result.Logical_Name := Make_Short ("python");
            Result.Binary_Name := Make_Short ("python3");
            Result.Requirement := Required;
         when Tool_Bash =>
            Result.Logical_Name := Make_Short ("bash");
            Result.Binary_Name := Make_Short ("bash");
            Result.Requirement := Required;
         when Tool_Git =>
            Result.Logical_Name := Make_Short ("git");
            Result.Binary_Name := Make_Short ("git");
            Result.Requirement := Optional;
         when Tool_CC =>
            Result.Logical_Name := Make_Short ("cc");
            Result.Binary_Name := Make_Short ("cc");
            Result.Requirement := Optional;
         when Tool_Rustc =>
            Result.Logical_Name := Make_Short ("rustc");
            Result.Binary_Name := Make_Short ("rustc");
            Result.Requirement := Optional;
         when Tool_Cargo =>
            Result.Logical_Name := Make_Short ("cargo");
            Result.Binary_Name := Make_Short ("cargo");
            Result.Requirement := Optional;
         when Tool_Stunir_Native =>
            Result.Logical_Name := Make_Short ("stunir_native");
            Result.Binary_Name := Make_Short ("stunir-native");
            Result.Requirement := Optional;
      end case;
      return Result;
   end Get_Builtin_Definition;

   --  Initialize registry with builtin tool definitions
   procedure Initialize_Registry (Reg : out Tool_Registry) is
   begin
      Reg := Empty_Registry;
      
      --  Add required tools first
      Reg.Entries (1) := Get_Builtin_Definition (Tool_Python);
      Reg.Entries (2) := Get_Builtin_Definition (Tool_Bash);
      Reg.Count := 2;
      Reg.Required_Count := 2;
      
      --  Add optional tools
      Reg.Entries (3) := Get_Builtin_Definition (Tool_Git);
      Reg.Entries (4) := Get_Builtin_Definition (Tool_CC);
      Reg.Entries (5) := Get_Builtin_Definition (Tool_Rustc);
      Reg.Entries (6) := Get_Builtin_Definition (Tool_Cargo);
      Reg.Entries (7) := Get_Builtin_Definition (Tool_Stunir_Native);
      Reg.Count := 7;
   end Initialize_Registry;

   --  Add custom tool to registry
   procedure Add_Tool (
      Reg          : in out Tool_Registry;
      Logical_Name : Short_String;
      Binary_Name  : Short_String;
      Requirement  : Tool_Requirement;
      Success      : out Boolean)
   is
   begin
      if Reg.Count >= Max_Tools then
         Success := False;
         return;
      end if;
      
      Reg.Count := Reg.Count + 1;
      Reg.Entries (Reg.Count).Logical_Name := Logical_Name;
      Reg.Entries (Reg.Count).Binary_Name := Binary_Name;
      Reg.Entries (Reg.Count).Requirement := Requirement;
      Reg.Entries (Reg.Count).Status := Status_Unknown;
      
      if Requirement = Required then
         Reg.Required_Count := Reg.Required_Count + 1;
      end if;
      
      Success := True;
   end Add_Tool;

   --  Find binary in PATH (stub - would use OS interface)
   procedure Find_In_Path (
      Binary_Name   : Short_String;
      Resolved_Path : out Path_String;
      Found         : out Boolean)
   is
      pragma Unreferenced (Binary_Name);
   begin
      --  SPARK-safe stub: actual implementation outside SPARK
      Resolved_Path := Empty_Path;
      Found := False;
   end Find_In_Path;

   --  Compute file SHA256 hash (stub - would use crypto library)
   procedure Compute_File_Hash (
      File_Path : Path_String;
      Hash      : out Hash_Hex;
      Success   : out Boolean)
   is
      pragma Unreferenced (File_Path);
   begin
      --  SPARK-safe stub: actual implementation outside SPARK
      Hash := Zero_Hash;
      Success := False;
   end Compute_File_Hash;

   --  Get binary version string (stub)
   procedure Get_Version_String (
      Tool_Path : Path_String;
      Binary    : Short_String;
      Version   : out Medium_String)
   is
      pragma Unreferenced (Tool_Path);
      pragma Unreferenced (Binary);
   begin
      --  SPARK-safe stub
      Version := Empty_Medium;
   end Get_Version_String;

   --  Resolve a single tool
   procedure Resolve_Tool (
      Tool_Ent : in out Tool_Entry;
      Success : out Boolean)
   is
      Found     : Boolean;
      Hash_Ok   : Boolean;
   begin
      --  Find in PATH
      Find_In_Path (Tool_Ent.Binary_Name, Tool_Ent.Resolved_Path, Found);
      
      if not Found then
         Tool_Ent.Status := Status_Not_Found;
         Success := Tool_Ent.Requirement = Optional;
         return;
      end if;
      
      --  Compute hash
      Compute_File_Hash (Tool_Ent.Resolved_Path, Tool_Ent.SHA256_Hash, Hash_Ok);
      
      --  Get version
      Get_Version_String (Tool_Ent.Resolved_Path, Tool_Ent.Binary_Name, 
                          Tool_Ent.Version_String);
      
      Tool_Ent.Status := Status_Resolved;
      Success := True;
   end Resolve_Tool;

   --  Resolve all tools in registry
   procedure Resolve_All (
      Reg     : in out Tool_Registry;
      Success : out Boolean)
   is
      Tool_Success : Boolean;
   begin
      Success := True;
      Reg.Resolved_Count := 0;
      
      for I in 1 .. Reg.Count loop
         Resolve_Tool (Reg.Entries (I), Tool_Success);
         
         if Reg.Entries (I).Status = Status_Resolved then
            Reg.Resolved_Count := Reg.Resolved_Count + 1;
         end if;
         
         --  Fail if required tool not resolved
         if not Tool_Success and Reg.Entries (I).Requirement = Required then
            Success := False;
         end if;
      end loop;
   end Resolve_All;

   --  Get platform information
   procedure Get_Platform_Info (Info : out Platform_Info) is
   begin
      --  SPARK-safe stub: would use OS calls
      Info.OS_Name := Make_Short ("Linux");
      Info.Architecture := Make_Short ("x86_64");
   end Get_Platform_Info;

   --  Generate toolchain lockfile
   procedure Generate_Lockfile (
      Reg      : Tool_Registry;
      Platform : Platform_Info;
      Lockfile : out Toolchain_Lockfile)
   is
   begin
      Lockfile.Registry := Reg;
      Lockfile.Platform := Platform;
      Lockfile.Is_Valid := All_Required_Resolved (Reg);
   end Generate_Lockfile;

   --  Scan entire toolchain (convenience procedure)
   procedure Scan_Toolchain (
      Lockfile : out Toolchain_Lockfile;
      Success  : out Boolean)
   is
      Platform : Platform_Info;
   begin
      --  Initialize registry
      Initialize_Registry (Lockfile.Registry);
      
      --  Resolve all tools
      Resolve_All (Lockfile.Registry, Success);
      
      --  Get platform info
      Get_Platform_Info (Platform);
      
      --  Generate lockfile
      Generate_Lockfile (Lockfile.Registry, Platform, Lockfile);
      
      Success := Lockfile.Is_Valid;
   end Scan_Toolchain;

end Toolchain_Scanner;
