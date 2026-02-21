-------------------------------------------------------------------------------
--  STUNIR Build Orchestrator Tests
--  Part of Phase 2 SPARK Migration
-------------------------------------------------------------------------------

with Ada.Text_IO; use Ada.Text_IO;
with Stunir_Strings;       use Stunir_Strings;
with Build_Config;         use Build_Config;
with Toolchain_Types;      use Toolchain_Types;
with Build_Orchestrator;   use Build_Orchestrator;

procedure Test_Build is
   
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
   Put_Line ("=== Build Orchestrator Tests ===");
   Put_Line ("");
   
   --  Test 1: Default result
   Put_Line ("Test Suite: Default Result");
   declare
      Result : Build_Result := Default_Result;
   begin
      Assert (Result.Status = Status_Not_Started, "Status is not started");
      Assert (Result.Final_Profile = Profile_Auto, "Profile is auto");
      Assert (Result.Receipt_Count = 0, "No receipts");
   end;
   Put_Line ("");
   
   --  Test 2: Runtime detection - explicit profile
   Put_Line ("Test Suite: Runtime Detection - Explicit");
   declare
      Config   : Configuration;
      Lockfile : Toolchain_Lockfile := (others => <>);
   begin
      Initialize_Config (Config);
      Config.Profile := Profile_Python;
      Assert (Detect_Runtime (Config, Lockfile) = Profile_Python,
              "Explicit Python profile");
      
      Config.Profile := Profile_Shell;
      Assert (Detect_Runtime (Config, Lockfile) = Profile_Shell,
              "Explicit Shell profile");
   end;
   Put_Line ("");
   
   --  Test 3: Python available check
   Put_Line ("Test Suite: Python Available Check");
   declare
      Lockfile : Toolchain_Lockfile := (others => <>);
   begin
      --  Empty lockfile - Python not available
      Assert (not Python_Available (Lockfile), "Empty lockfile - no Python");
   end;
   Put_Line ("");
   
   --  Test 4: Set error on result
   Put_Line ("Test Suite: Set Error");
   declare
      Result : Build_Result := Default_Result;
   begin
      Set_Error (Result, "Test error message");
      Assert (Result.Status = Status_Failed, "Status is failed");
      Assert (Result.Error_Message.Length > 0, "Error message set");
   end;
   Put_Line ("");
   
   --  Test 5: All phases succeeded check
   Put_Line ("Test Suite: All Phases Succeeded");
   declare
      Result : Build_Result := Default_Result;
   begin
      --  Default: all not started, should pass (not failed)
      Assert (All_Phases_Succeeded (Result), "Default passes");
      
      --  Set one to failed
      Result.Phase_Results (Phase_Discovery) := Status_Failed;
      Assert (not All_Phases_Succeeded (Result), "With failed phase");
   end;
   Put_Line ("");
   
   --  Test 6: Phase status tracking
   Put_Line ("Test Suite: Phase Status Tracking");
   declare
      Result : Build_Result := Default_Result;
   begin
      Result.Phase_Results (Phase_Discovery) := Status_Success;
      Result.Phase_Results (Phase_Epoch) := Status_Success;
      Result.Phase_Results (Phase_Compile) := Status_Skipped;
      
      Assert (Result.Phase_Results (Phase_Discovery) = Status_Success,
              "Discovery succeeded");
      Assert (Result.Phase_Results (Phase_Compile) = Status_Skipped,
              "Compile skipped");
      Assert (Result.Phase_Results (Phase_Verify) = Status_Not_Started,
              "Verify not started");
   end;
   Put_Line ("");
   
   --  Summary
   Put_Line ("=== Summary ===");
   Put_Line ("Tests Passed:" & Natural'Image (Tests_Passed));
   Put_Line ("Tests Failed:" & Natural'Image (Tests_Failed));
   
end Test_Build;
