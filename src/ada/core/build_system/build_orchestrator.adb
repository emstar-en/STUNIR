-------------------------------------------------------------------------------
--  STUNIR Build Orchestrator - Ada SPARK Implementation
--  Part of Phase 2 SPARK Migration
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

with Toolchain_Scanner;   use Toolchain_Scanner;
with Epoch_Selector;      use Epoch_Selector;

package body Build_Orchestrator is

   --  Detect runtime profile
   function Detect_Runtime (
      Config   : Configuration;
      Lockfile : Toolchain_Lockfile) return Build_Profile
   is
   begin
      --  If profile explicitly set, use it
      if Config.Profile /= Profile_Auto then
         return Config.Profile;
      end if;
      
      --  Auto-detection priority: Native -> Python -> Shell
      if Native_Binary_Available (Config, Lockfile) then
         return Profile_Native;
      elsif Python_Available (Lockfile) then
         return Profile_Python;
      else
         return Profile_Shell;
      end if;
   end Detect_Runtime;

   --  Check if native binary exists
   function Native_Binary_Available (
      Config   : Configuration;
      Lockfile : Toolchain_Lockfile) return Boolean
   is
      Tool_Idx : constant Natural := Find_Tool (
         Lockfile.Registry, Make_Short ("stunir_native"));
   begin
      if Tool_Idx = 0 then
         return False;
      end if;
      return Lockfile.Registry.Entries (Tool_Idx).Status = Status_Resolved;
   end Native_Binary_Available;

   --  Check if Python is available
   function Python_Available (Lockfile : Toolchain_Lockfile) return Boolean is
      Tool_Idx : constant Natural := Find_Tool (
         Lockfile.Registry, Make_Short ("python"));
   begin
      if Tool_Idx = 0 then
         return False;
      end if;
      return Lockfile.Registry.Entries (Tool_Idx).Status = Status_Resolved;
   end Python_Available;

   --  Execute build phase
   procedure Execute_Phase (
      Config  : Configuration;
      Phase   : Build_Phase;
      Result  : in out Build_Result)
   is
      Lockfile : Toolchain_Lockfile;
   begin
      Result.Phase_Results (Phase) := Status_In_Progress;
      
      case Phase is
         when Phase_Discovery =>
            Run_Discovery_Phase (Config, Lockfile, Result);
         when Phase_Epoch =>
            Run_Epoch_Phase (Config, Result);
         when Phase_Spec_Parse =>
            Run_Spec_Parse_Phase (Config, Result.Final_Profile, Result);
         when Phase_IR_Emit =>
            Run_IR_Emit_Phase (Config, Result.Final_Profile, Result);
         when Phase_Code_Gen =>
            Run_Code_Gen_Phase (Config, Result.Final_Profile, Result);
         when Phase_Compile =>
            --  Optional: skip for now
            Result.Phase_Results (Phase) := Status_Skipped;
         when Phase_Receipt =>
            Run_Receipt_Phase (Config, Result);
         when Phase_Verify =>
            Run_Verify_Phase (Config, Result);
      end case;
   end Execute_Phase;

   --  Run discovery phase
   procedure Run_Discovery_Phase (
      Config   : Configuration;
      Lockfile : out Toolchain_Lockfile;
      Result   : in out Build_Result)
   is
      Success : Boolean;
      pragma Unreferenced (Config);
   begin
      Scan_Toolchain (Lockfile, Success);
      
      if Success then
         Result.Phase_Results (Phase_Discovery) := Status_Success;
      else
         Result.Phase_Results (Phase_Discovery) := Status_Failed;
         Set_Error (Result, "Required tools not found");
      end if;
   end Run_Discovery_Phase;

   --  Run epoch selection phase
   procedure Run_Epoch_Phase (
      Config : Configuration;
      Result : in out Build_Result)
   is
      Selection : Epoch_Selection;
   begin
      Select_Epoch (Config.Spec_Root, False, Selection);
      Result.Epoch := Selection;
      
      if Is_Valid_Selection (Selection) then
         Result.Phase_Results (Phase_Epoch) := Status_Success;
      else
         Result.Phase_Results (Phase_Epoch) := Status_Failed;
         Set_Error (Result, "Epoch selection failed");
      end if;
   end Run_Epoch_Phase;

   --  Run spec parse phase (stub - would invoke runtime-specific parser)
   procedure Run_Spec_Parse_Phase (
      Config  : Configuration;
      Profile : Build_Profile;
      Result  : in out Build_Result)
   is
      pragma Unreferenced (Config);
      pragma Unreferenced (Profile);
   begin
      --  SPARK-safe stub: actual implementation would dispatch
      Result.Phase_Results (Phase_Spec_Parse) := Status_Success;
   end Run_Spec_Parse_Phase;

   --  Run IR emission phase (stub)
   procedure Run_IR_Emit_Phase (
      Config  : Configuration;
      Profile : Build_Profile;
      Result  : in out Build_Result)
   is
      pragma Unreferenced (Config);
      pragma Unreferenced (Profile);
   begin
      --  SPARK-safe stub
      Result.Phase_Results (Phase_IR_Emit) := Status_Success;
   end Run_IR_Emit_Phase;

   --  Run code generation phase (stub)
   procedure Run_Code_Gen_Phase (
      Config  : Configuration;
      Profile : Build_Profile;
      Result  : in out Build_Result)
   is
      pragma Unreferenced (Config);
      pragma Unreferenced (Profile);
   begin
      --  SPARK-safe stub
      Result.Phase_Results (Phase_Code_Gen) := Status_Success;
   end Run_Code_Gen_Phase;

   --  Run receipt generation phase (stub)
   procedure Run_Receipt_Phase (
      Config  : Configuration;
      Result  : in out Build_Result)
   is
      pragma Unreferenced (Config);
   begin
      --  SPARK-safe stub
      Result.Phase_Results (Phase_Receipt) := Status_Success;
      Result.Receipt_Count := 1;  --  At least one receipt
   end Run_Receipt_Phase;

   --  Run verification phase (stub)
   procedure Run_Verify_Phase (
      Config  : Configuration;
      Result  : in out Build_Result)
   is
      pragma Unreferenced (Config);
   begin
      --  SPARK-safe stub
      Result.Phase_Results (Phase_Verify) := Status_Success;
   end Run_Verify_Phase;

   --  Full build pipeline
   procedure Run_Build (
      Config : Configuration;
      Result : out Build_Result)
   is
      Lockfile : Toolchain_Lockfile;
   begin
      Result := Default_Result;
      Result.Status := Status_In_Progress;
      
      --  Phase 1: Discovery
      Run_Discovery_Phase (Config, Lockfile, Result);
      if Result.Phase_Results (Phase_Discovery) = Status_Failed then
         Result.Status := Status_Failed;
         return;
      end if;
      
      --  Detect runtime
      Result.Final_Profile := Detect_Runtime (Config, Lockfile);
      
      --  Phase 2: Epoch
      Run_Epoch_Phase (Config, Result);
      if Result.Phase_Results (Phase_Epoch) = Status_Failed then
         Result.Status := Status_Failed;
         return;
      end if;
      
      --  Phase 3: Spec Parse
      Run_Spec_Parse_Phase (Config, Result.Final_Profile, Result);
      if Result.Phase_Results (Phase_Spec_Parse) = Status_Failed then
         Result.Status := Status_Failed;
         return;
      end if;
      
      --  Phase 4: IR Emit
      Run_IR_Emit_Phase (Config, Result.Final_Profile, Result);
      if Result.Phase_Results (Phase_IR_Emit) = Status_Failed then
         Result.Status := Status_Failed;
         return;
      end if;
      
      --  Phase 5: Code Gen
      Run_Code_Gen_Phase (Config, Result.Final_Profile, Result);
      if Result.Phase_Results (Phase_Code_Gen) = Status_Failed then
         Result.Status := Status_Failed;
         return;
      end if;
      
      --  Phase 6: Compile (optional - skip)
      Result.Phase_Results (Phase_Compile) := Status_Skipped;
      
      --  Phase 7: Receipt
      Run_Receipt_Phase (Config, Result);
      if Result.Phase_Results (Phase_Receipt) = Status_Failed then
         Result.Status := Status_Failed;
         return;
      end if;
      
      --  Phase 8: Verify
      Run_Verify_Phase (Config, Result);
      if Result.Phase_Results (Phase_Verify) = Status_Failed then
         Result.Status := Status_Failed;
         return;
      end if;
      
      --  All phases complete
      Result.Status := Status_Success;
   end Run_Build;

   --  Log message (stub)
   procedure Log_Message (
      Config  : Configuration;
      Message : String)
   is
      pragma Unreferenced (Config);
      pragma Unreferenced (Message);
   begin
      --  SPARK-safe stub: would output to console in verbose mode
      null;
   end Log_Message;

   --  Set error on result
   procedure Set_Error (
      Result  : in out Build_Result;
      Message : String)
   is
   begin
      Result.Status := Status_Failed;
      if Message'Length <= Max_Medium_String then
         Result.Error_Message := Make_Medium (Message);
      end if;
   end Set_Error;

   --  Check if all phases succeeded
   function All_Phases_Succeeded (Result : Build_Result) return Boolean is
   begin
      for Phase in Build_Phase loop
         if Result.Phase_Results (Phase) = Status_Failed then
            return False;
         end if;
      end loop;
      return True;
   end All_Phases_Succeeded;

end Build_Orchestrator;
