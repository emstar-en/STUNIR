--  STUNIR Test Orchestrator Types
--  SPARK Migration Phase 3 - Test Infrastructure
--  Test orchestration types and contracts

pragma SPARK_Mode (On);

with Stunir_Hashes; use Stunir_Hashes;

package Orchestrator_Types is

   --  ===========================================
   --  Constants
   --  ===========================================

   Max_Tools : constant := 16;
   Max_Tool_Name : constant := 64;
   Max_Output_Size : constant := 8192;
   Max_Args : constant := 32;
   Max_Arg_Length : constant := 256;

   --  ===========================================
   --  Tool Status
   --  ===========================================

   type Tool_Status is (
      Available,      --  Tool is available and ready
      Not_Found,      --  Tool not found in path
      Error,          --  Tool encountered error
      Timeout,        --  Tool execution timed out
      Success         --  Tool executed successfully
   );

   --  ===========================================
   --  Comparison Result
   --  ===========================================

   type Comparison_Result is (
      Match,          --  Outputs are identical
      Mismatch,       --  Outputs differ
      Err,            --  Comparison failed
      Incomplete      --  Not all tools completed
   );

   --  ===========================================
   --  String Types
   --  ===========================================

   subtype Tool_Name_String is String (1 .. Max_Tool_Name);
   subtype Output_Buffer is String (1 .. Max_Output_Size);
   subtype Arg_String is String (1 .. Max_Arg_Length);

   --  ===========================================
   --  Tool Output
   --  ===========================================

   type Tool_Output is record
      Tool_Name  : Tool_Name_String;
      Name_Len   : Natural;
      Output     : Output_Buffer;
      Output_Len : Natural;
      Hash       : String (1 .. Hash_Length);
      Exit_Code  : Integer;
      Status     : Tool_Status;
   end record;

   --  Empty output constant
   Empty_Output : constant Tool_Output := (
      Tool_Name  => (others => ' '),
      Name_Len   => 0,
      Output     => (others => ' '),
      Output_Len => 0,
      Hash       => (others => '0'),
      Exit_Code  => -1,
      Status     => Not_Found
   );

   --  ===========================================
   --  Tool Entry
   --  ===========================================

   type Tool_Entry is record
      Name     : Tool_Name_String;
      Name_Len : Natural;
      Path     : Tool_Name_String;  --  Reuse same size for path
      Path_Len : Natural;
      Is_Ready : Boolean;
   end record;

   --  Empty tool entry constant
   Empty_Tool_Entry : constant Tool_Entry := (
      Name     => (others => ' '),
      Name_Len => 0,
      Path     => (others => ' '),
      Path_Len => 0,
      Is_Ready => False
   );

   --  ===========================================
   --  Tool Array
   --  ===========================================

   type Tool_Index is range 0 .. Max_Tools;
   subtype Valid_Tool_Index is Tool_Index range 1 .. Max_Tools;

   type Tool_Array is array (Valid_Tool_Index) of Tool_Entry;
   type Output_Array is array (Valid_Tool_Index) of Tool_Output;

   --  ===========================================
   --  Orchestration Statistics
   --  ===========================================

   type Orchestration_Stats is record
      Total_Tools   : Natural := 0;
      Ready_Tools   : Natural := 0;
      Success_Count : Natural := 0;
      Error_Count   : Natural := 0;
      Timeout_Count : Natural := 0;
   end record;

   --  Empty stats constant
   Empty_Orch_Stats : constant Orchestration_Stats := (
      Total_Tools   => 0,
      Ready_Tools   => 0,
      Success_Count => 0,
      Error_Count   => 0,
      Timeout_Count => 0
   );

   --  ===========================================
   --  Orchestration Session
   --  ===========================================

   type Orchestration_Session is record
      Tools       : Tool_Array;
      Outputs     : Output_Array;
      Tool_Count  : Tool_Index;
      Stats       : Orchestration_Stats;
      Reference   : Tool_Index;  --  Reference tool for comparison
      Is_Complete : Boolean;
   end record;

   --  ===========================================
   --  Conformance Result
   --  ===========================================

   type Conformance_Result is record
      Total_Compared : Natural;
      Matching       : Natural;
      Mismatching    : Natural;
      Reference_Hash : String (1 .. Hash_Length);
      All_Match      : Boolean;
   end record;

   --  Empty conformance result constant
   Empty_Conformance : constant Conformance_Result := (
      Total_Compared => 0,
      Matching       => 0,
      Mismatching    => 0,
      Reference_Hash => (others => '0'),
      All_Match      => True
   );

   --  ===========================================
   --  Helper Functions
   --  ===========================================

   function Is_Success_Status (S : Tool_Status) return Boolean is
      (S = Success);

   function Is_Error_Status (S : Tool_Status) return Boolean is
      (S in Error | Timeout | Not_Found);

   function Is_Match_Result (R : Comparison_Result) return Boolean is
      (R = Match);

   --  Initialize functions
   function Init_Tool_Array return Tool_Array;
   function Init_Output_Array return Output_Array;
   function Empty_Session return Orchestration_Session;

end Orchestrator_Types;
