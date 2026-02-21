--  STUNIR Test Harness Types
--  SPARK Migration Phase 3 - Test Infrastructure
--  Part of 100% SPARK Migration

pragma SPARK_Mode (On);

package Test_Harness_Types is

   --  ===========================================
   --  Constants
   --  ===========================================

   Max_Test_Name_Length : constant := 128;
   Max_Tests : constant := 256;
   Max_Message_Length : constant := 512;
   Max_Output_Length : constant := 4096;

   --  ===========================================
   --  Test Status Enumeration
   --  ===========================================

   type Test_Status is (
      Pending,    --  Test not yet executed
      Running,    --  Test currently executing
      Passed,     --  Test completed successfully
      Failed,     --  Test completed with failures
      Skipped,    --  Test was skipped
      Error,      --  Test encountered an error
      Timeout     --  Test exceeded time limit
   );

   --  ===========================================
   --  Test Category
   --  ===========================================

   type Test_Category is (
      Unit_Test,          --  Individual function tests
      Integration_Test,   --  Cross-module tests
      Conformance_Test,   --  Multi-tool conformance
      Performance_Test,   --  Timing/resource tests
      Regression_Test     --  Bug fix verification
   );

   --  ===========================================
   --  Test Priority
   --  ===========================================

   type Test_Priority is (Critical, High, Medium, Low);

   --  ===========================================
   --  Test Name Type
   --  ===========================================

   subtype Test_Name_String is String (1 .. Max_Test_Name_Length);
   subtype Message_String is String (1 .. Max_Message_Length);
   subtype Output_String is String (1 .. Max_Output_Length);

   --  ===========================================
   --  Test Case Record
   --  ===========================================

   type Test_Case is record
      Name       : Test_Name_String;
      Name_Len   : Natural;
      Category   : Test_Category;
      Priority   : Test_Priority;
      Timeout_Ms : Natural;
      Is_Enabled : Boolean;
   end record;

   --  Empty test case constant
   Empty_Test : constant Test_Case := (
      Name       => (others => ' '),
      Name_Len   => 0,
      Category   => Unit_Test,
      Priority   => Medium,
      Timeout_Ms => 5000,
      Is_Enabled => False
   );

   --  ===========================================
   --  Test Result Record
   --  ===========================================

   type Test_Result is record
      Name        : Test_Name_String;
      Name_Len    : Natural;
      Status      : Test_Status;
      Duration_Ms : Natural;
      Message     : Message_String;
      Msg_Len     : Natural;
      Output      : Output_String;
      Output_Len  : Natural;
   end record;

   --  Empty result constant
   Empty_Result : constant Test_Result := (
      Name        => (others => ' '),
      Name_Len    => 0,
      Status      => Pending,
      Duration_Ms => 0,
      Message     => (others => ' '),
      Msg_Len     => 0,
      Output      => (others => ' '),
      Output_Len  => 0
   );

   --  ===========================================
   --  Test Suite Statistics
   --  ===========================================

   type Suite_Stats is record
      Total    : Natural := 0;
      Passed   : Natural := 0;
      Failed   : Natural := 0;
      Skipped  : Natural := 0;
      Errors   : Natural := 0;
      Timeouts : Natural := 0;
   end record;

   --  Empty stats constant
   Empty_Stats : constant Suite_Stats := (
      Total    => 0,
      Passed   => 0,
      Failed   => 0,
      Skipped  => 0,
      Errors   => 0,
      Timeouts => 0
   );

   --  ===========================================
   --  Test Queue
   --  ===========================================

   type Test_Index is range 0 .. Max_Tests;
   subtype Valid_Test_Index is Test_Index range 1 .. Max_Tests;

   type Test_Queue is array (Valid_Test_Index) of Test_Case;
   type Result_Queue is array (Valid_Test_Index) of Test_Result;

   --  ===========================================
   --  Test Suite Record
   --  ===========================================

   type Test_Suite is record
      Tests   : Test_Queue;
      Results : Result_Queue;
      Count   : Test_Index;
      Stats   : Suite_Stats;
   end record;

   --  Init functions
   function Init_Test_Queue return Test_Queue;
   function Init_Result_Queue return Result_Queue;
   function Empty_Suite return Test_Suite;

   --  ===========================================
   --  Helper Functions
   --  ===========================================

   function Is_Terminal_Status (S : Test_Status) return Boolean is
      (S in Passed | Failed | Skipped | Error | Timeout);

   function Is_Success_Status (S : Test_Status) return Boolean is
      (S = Passed);

   function Is_Failure_Status (S : Test_Status) return Boolean is
      (S in Failed | Error | Timeout);

   function Get_Success_Rate (Stats : Suite_Stats) return Natural
      with Pre => Stats.Total > 0,
           Post => Get_Success_Rate'Result <= 100;

end Test_Harness_Types;
