-------------------------------------------------------------------------------
--  STUNIR Build Configuration - Ada SPARK Specification
--  Part of Phase 2 SPARK Migration
--
--  This package provides build configuration types and management.
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

with Stunir_Strings; use Stunir_Strings;

package Build_Config is

   --  Build profile (runtime selection)
   type Build_Profile is (
      Profile_Auto,     --  Auto-detect best runtime
      Profile_Native,   --  Use native (Rust/Haskell) tools
      Profile_Python,   --  Use Python tools
      Profile_Shell     --  Use shell-native tools
   );

   --  Build phase enumeration
   type Build_Phase is (
      Phase_Discovery,     --  Toolchain discovery
      Phase_Epoch,         --  Epoch selection
      Phase_Spec_Parse,    --  Spec parsing
      Phase_IR_Emit,       --  IR emission
      Phase_Code_Gen,      --  Code generation
      Phase_Compile,       --  Optional compilation
      Phase_Receipt,       --  Receipt generation
      Phase_Verify         --  Verification
   );

   --  Build result status
   type Build_Status is (
      Status_Not_Started,
      Status_In_Progress,
      Status_Success,
      Status_Failed,
      Status_Skipped
   );

   --  Configuration record
   type Configuration is record
      Profile        : Build_Profile := Profile_Auto;
      Spec_Root      : Path_String := Empty_Path;
      Output_IR      : Path_String := Empty_Path;
      Output_Code    : Path_String := Empty_Path;
      Lock_File      : Path_String := Empty_Path;
      Native_Binary  : Path_String := Empty_Path;
      Strict_Mode    : Boolean := False;
      Verbose        : Boolean := False;
      Is_Valid       : Boolean := False;
   end record;

   --  Default configuration
   Default_Config : constant Configuration := (
      Profile        => Profile_Auto,
      Spec_Root      => Empty_Path,
      Output_IR      => Empty_Path,
      Output_Code    => Empty_Path,
      Lock_File      => Empty_Path,
      Native_Binary  => Empty_Path,
      Strict_Mode    => False,
      Verbose        => False,
      Is_Valid       => False
   );

   --  Phase status tracking
   type Phase_Status_Array is array (Build_Phase) of Build_Status;

   --  Build state (tracks execution progress)
   type Build_State is record
      Config       : Configuration := Default_Config;
      Phases       : Phase_Status_Array := (others => Status_Not_Started);
      Current      : Build_Phase := Phase_Discovery;
      Error_Count  : Natural := 0;
      Is_Complete  : Boolean := False;
   end record;

   --  Initialize configuration with defaults
   procedure Initialize_Config (
      Config : out Configuration)
     with
       Post => not Config.Is_Valid;  --  Not valid until paths set

   --  Set default paths based on conventional layout
   procedure Set_Default_Paths (
      Config : in out Configuration)
     with
       Post => Config.Spec_Root.Length > 0;

   --  Validate configuration
   function Validate_Config (Config : Configuration) return Boolean;

   --  Profile string conversion
   function Profile_To_String (P : Build_Profile) return String;
   function String_To_Profile (S : String) return Build_Profile;

   --  Phase string conversion
   function Phase_To_String (P : Build_Phase) return String;

   --  Check if profile is auto
   function Is_Auto_Profile (Config : Configuration) return Boolean is
     (Config.Profile = Profile_Auto);

end Build_Config;
