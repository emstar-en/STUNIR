-------------------------------------------------------------------------------
--  STUNIR Configuration Manager Tests
--  Part of Phase 2 SPARK Migration
-------------------------------------------------------------------------------

with Ada.Text_IO; use Ada.Text_IO;
with Stunir_Strings; use Stunir_Strings;
with Build_Config;   use Build_Config;
with Config_Parser;  use Config_Parser;

procedure Test_Config is
   
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
   Put_Line ("=== Configuration Manager Tests ===");
   Put_Line ("");
   
   --  Test 1: Profile conversion
   Put_Line ("Test Suite: Profile Conversion");
   Assert (Profile_To_String (Profile_Auto) = "auto", "Auto profile string");
   Assert (Profile_To_String (Profile_Native) = "native", "Native profile string");
   Assert (Profile_To_String (Profile_Python) = "python", "Python profile string");
   Assert (Profile_To_String (Profile_Shell) = "shell", "Shell profile string");
   Put_Line ("");
   
   --  Test 2: String to profile
   Put_Line ("Test Suite: String to Profile");
   Assert (String_To_Profile ("native") = Profile_Native, "Parse native");
   Assert (String_To_Profile ("python") = Profile_Python, "Parse python");
   Assert (String_To_Profile ("invalid") = Profile_Auto, "Invalid defaults to auto");
   Put_Line ("");
   
   --  Test 3: Phase conversion
   Put_Line ("Test Suite: Phase Conversion");
   Assert (Phase_To_String (Phase_Discovery) = "discovery", "Discovery phase");
   Assert (Phase_To_String (Phase_Epoch) = "epoch", "Epoch phase");
   Assert (Phase_To_String (Phase_Verify) = "verify", "Verify phase");
   Put_Line ("");
   
   --  Test 4: Configuration initialization
   Put_Line ("Test Suite: Configuration Initialization");
   declare
      Config : Configuration;
   begin
      Initialize_Config (Config);
      Assert (Config.Profile = Profile_Auto, "Default profile is auto");
      Assert (not Config.Is_Valid, "Initially not valid");
      Assert (not Config.Strict_Mode, "Strict mode off by default");
   end;
   Put_Line ("");
   
   --  Test 5: Set default paths
   Put_Line ("Test Suite: Set Default Paths");
   declare
      Config : Configuration;
   begin
      Initialize_Config (Config);
      Set_Default_Paths (Config);
      Assert (Config.Spec_Root.Length > 0, "Spec root set");
      Assert (Config.Output_IR.Length > 0, "Output IR set");
      Assert (Config.Lock_File.Length > 0, "Lock file set");
   end;
   Put_Line ("");
   
   --  Test 6: Configuration validation
   Put_Line ("Test Suite: Configuration Validation");
   declare
      Config : Configuration;
   begin
      Initialize_Config (Config);
      Assert (not Validate_Config (Config), "Empty config invalid");
      
      Set_Default_Paths (Config);
      Assert (Validate_Config (Config), "Config with paths valid");
   end;
   Put_Line ("");
   
   --  Test 7: Is auto profile check
   Put_Line ("Test Suite: Auto Profile Check");
   declare
      Config : Configuration;
   begin
      Initialize_Config (Config);
      Assert (Is_Auto_Profile (Config), "Default is auto");
      
      Config.Profile := Profile_Native;
      Assert (not Is_Auto_Profile (Config), "Native is not auto");
   end;
   Put_Line ("");
   
   --  Test 8: Parse from environment
   Put_Line ("Test Suite: Parse From Environment");
   declare
      Config  : Configuration;
      Success : Boolean;
   begin
      Parse_From_Environment (Config, Success);
      Assert (Config.Spec_Root.Length > 0, "Has spec root after env parse");
   end;
   Put_Line ("");
   
   --  Summary
   Put_Line ("=== Summary ===");
   Put_Line ("Tests Passed:" & Natural'Image (Tests_Passed));
   Put_Line ("Tests Failed:" & Natural'Image (Tests_Failed));
   
end Test_Config;
