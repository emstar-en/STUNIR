-------------------------------------------------------------------------------
--  STUNIR Build Orchestrator - Ada SPARK Specification
--  Part of Phase 2 SPARK Migration
--
--  This package provides build pipeline orchestration.
--  Migrated from: scripts/build.sh
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

with Stunir_Strings;        use Stunir_Strings;
with Stunir_Hashes;         use Stunir_Hashes;
with Build_Config;          use Build_Config;
with Epoch_Types;           use Epoch_Types;
with Toolchain_Types;       use Toolchain_Types;
with Receipt_Types;         use Receipt_Types;

package Build_Orchestrator is

   --  Build result
   type Build_Result is record
      Status        : Build_Status := Status_Not_Started;
      Final_Profile : Build_Profile := Profile_Auto;
      Error_Message : Medium_String := Empty_Medium;
      Phase_Results : Phase_Status_Array := (others => Status_Not_Started);
      Epoch         : Epoch_Selection;
      Receipt_Count : Natural := 0;
   end record;

   --  Default result
   Default_Result : constant Build_Result := (
      Status        => Status_Not_Started,
      Final_Profile => Profile_Auto,
      Error_Message => Empty_Medium,
      Phase_Results => (others => Status_Not_Started),
      Epoch         => Default_Epoch_Selection,
      Receipt_Count => 0
   );

   --  Detect runtime profile based on available tools
   function Detect_Runtime (
      Config   : Configuration;
      Lockfile : Toolchain_Lockfile) return Build_Profile
     with
       Post => Detect_Runtime'Result /= Profile_Auto;

   --  Check if native binary exists and is executable
   function Native_Binary_Available (
      Config   : Configuration;
      Lockfile : Toolchain_Lockfile) return Boolean;

   --  Check if Python is available
   function Python_Available (
      Lockfile : Toolchain_Lockfile) return Boolean;

   --  Execute build phase
   procedure Execute_Phase (
      Config  : Configuration;
      Phase   : Build_Phase;
      Result  : in out Build_Result)
     with
       Pre  => Config.Is_Valid,
       Post => Result.Phase_Results (Phase) in Status_Success | Status_Failed | Status_Skipped;

   --  Run discovery phase
   procedure Run_Discovery_Phase (
      Config   : Configuration;
      Lockfile : out Toolchain_Lockfile;
      Result   : in out Build_Result);

   --  Run epoch selection phase
   procedure Run_Epoch_Phase (
      Config : Configuration;
      Result : in out Build_Result);

   --  Run spec parse phase (dispatches to appropriate runtime)
   procedure Run_Spec_Parse_Phase (
      Config  : Configuration;
      Profile : Build_Profile;
      Result  : in out Build_Result);

   --  Run IR emission phase
   procedure Run_IR_Emit_Phase (
      Config  : Configuration;
      Profile : Build_Profile;
      Result  : in out Build_Result);

   --  Run code generation phase
   procedure Run_Code_Gen_Phase (
      Config  : Configuration;
      Profile : Build_Profile;
      Result  : in out Build_Result);

   --  Run receipt generation phase
   procedure Run_Receipt_Phase (
      Config  : Configuration;
      Result  : in out Build_Result);

   --  Run verification phase
   procedure Run_Verify_Phase (
      Config  : Configuration;
      Result  : in out Build_Result);

   --  Full build pipeline
   procedure Run_Build (
      Config : Configuration;
      Result : out Build_Result)
     with
       Pre  => Config.Is_Valid,
       Post => Result.Status in Status_Success | Status_Failed;

   --  Log message (for verbose mode)
   procedure Log_Message (
      Config  : Configuration;
      Message : String);

   --  Set error on result
   procedure Set_Error (
      Result  : in out Build_Result;
      Message : String)
     with
       Post => Result.Status = Status_Failed;

   --  Check if all phases succeeded
   function All_Phases_Succeeded (Result : Build_Result) return Boolean;

end Build_Orchestrator;
