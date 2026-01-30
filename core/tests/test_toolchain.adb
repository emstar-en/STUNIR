-------------------------------------------------------------------------------
--  STUNIR Toolchain Scanner Tests
--  Part of Phase 2 SPARK Migration
-------------------------------------------------------------------------------

with Ada.Text_IO; use Ada.Text_IO;
with Stunir_Strings;      use Stunir_Strings;
with Stunir_Hashes;       use Stunir_Hashes;
with Toolchain_Types;     use Toolchain_Types;
with Toolchain_Scanner;   use Toolchain_Scanner;

procedure Test_Toolchain is
   
   Tests_Passed : Natural := 0;
   Tests_Failed : Natural := 0;
   
   procedure Assert (Condition : Boolean; Name : String) is
   begin
      if Condition then
         Put_Line ("  [PASS] " & Name);
         Tests_Passed := Tests_Passed + 1;
      else
         Put_Line ("  [FAIL] " & Name);
         Tests_Failed := Tests_Failed + 1;
      end if;
   end Assert;
   
begin
   Put_Line ("=== Toolchain Scanner Tests ===");
   Put_Line ("");
   
   --  Test 1: Builtin tool definitions
   Put_Line ("Test Suite: Builtin Tool Definitions");
   declare
      Python_Tool : constant Tool_Entry := Get_Builtin_Definition (Tool_Python);
      Bash_Tool   : constant Tool_Entry := Get_Builtin_Definition (Tool_Bash);
      Git_Tool    : constant Tool_Entry := Get_Builtin_Definition (Tool_Git);
   begin
      Assert (Python_Tool.Requirement = Required, "Python is required");
      Assert (Bash_Tool.Requirement = Required, "Bash is required");
      Assert (Git_Tool.Requirement = Optional, "Git is optional");
      Assert (Python_Tool.Logical_Name.Length > 0, "Python has logical name");
      Assert (Python_Tool.Binary_Name.Length > 0, "Python has binary name");
   end;
   Put_Line ("");
   
   --  Test 2: Registry initialization
   Put_Line ("Test Suite: Registry Initialization");
   declare
      Reg : Tool_Registry;
   begin
      Initialize_Registry (Reg);
      Assert (Reg.Count >= 2, "Registry has at least 2 tools");
      Assert (Reg.Required_Count = 2, "Registry has 2 required tools");
   end;
   Put_Line ("");
   
   --  Test 3: Add custom tool
   Put_Line ("Test Suite: Add Custom Tool");
   declare
      Reg     : Tool_Registry;
      Success : Boolean;
   begin
      Initialize_Registry (Reg);
      declare
         Initial_Count : constant Natural := Reg.Count;
      begin
         Add_Tool (Reg, Make_Short ("custom"), Make_Short ("custom-bin"), 
                   Optional, Success);
         Assert (Success, "Add tool succeeded");
         Assert (Reg.Count = Initial_Count + 1, "Count increased");
      end;
   end;
   Put_Line ("");
   
   --  Test 4: Find tool in registry
   Put_Line ("Test Suite: Find Tool");
   declare
      Reg : Tool_Registry;
   begin
      Initialize_Registry (Reg);
      Assert (Find_Tool (Reg, Make_Short ("python")) > 0, "Find python");
      Assert (Find_Tool (Reg, Make_Short ("bash")) > 0, "Find bash");
      Assert (Find_Tool (Reg, Make_Short ("nonexistent")) = 0, "Nonexistent not found");
   end;
   Put_Line ("");
   
   --  Test 5: Platform info
   Put_Line ("Test Suite: Platform Info");
   declare
      Info : Platform_Info;
   begin
      Get_Platform_Info (Info);
      Assert (Info.OS_Name.Length > 0, "Has OS name");
      Assert (Info.Architecture.Length > 0, "Has architecture");
   end;
   Put_Line ("");
   
   --  Test 6: All required resolved check
   Put_Line ("Test Suite: Required Resolved Check");
   declare
      Reg : Tool_Registry := Empty_Registry;
   begin
      --  Empty registry should pass (no required)
      Assert (All_Required_Resolved (Reg), "Empty registry passes");
      
      --  Add required but unresolved
      Reg.Count := 1;
      Reg.Entries (1).Requirement := Required;
      Reg.Entries (1).Status := Status_Unknown;
      Reg.Required_Count := 1;
      Reg.Resolved_Count := 0;
      Assert (not All_Required_Resolved (Reg), "Unresolved required fails");
   end;
   Put_Line ("");
   
   --  Summary
   Put_Line ("=== Summary ===");
   Put_Line ("Tests Passed:" & Natural'Image (Tests_Passed));
   Put_Line ("Tests Failed:" & Natural'Image (Tests_Failed));
   
end Test_Toolchain;
