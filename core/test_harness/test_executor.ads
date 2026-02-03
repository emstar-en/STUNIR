--  STUNIR Test Executor
--  SPARK Migration Phase 3 - Test Infrastructure
--  Handles test discovery, execution, and scheduling

pragma SPARK_Mode (On);

with Test_Harness_Types; use Test_Harness_Types;

package Test_Executor is

   --  ===========================================
   --  Suite Management
   --  ===========================================

   --  Initialize a new test suite
   procedure Initialize_Suite (Suite : out Test_Suite)
      with Post => Suite.Count = 0 and Suite.Stats = Empty_Stats;

   --  Register a test case
   procedure Register_Test (
      Suite   : in out Test_Suite;
      TC      : in Test_Case;
      Success : out Boolean)
      with Pre  => Suite.Count < Max_Tests and TC.Name_Len > 0,
           Post => (if Success then Suite.Count = Suite.Count'Old + 1
                    else Suite.Count = Suite.Count'Old);

   --  Create a test case from parameters
   function Create_Test (
      Name     : String;
      Category : Test_Category;
      Priority : Test_Priority;
      Timeout  : Natural) return Test_Case
      with Pre  => Name'Length > 0 and Name'Length <= Max_Test_Name_Length,
           Post => Create_Test'Result.Name_Len = Name'Length and
                   Create_Test'Result.Is_Enabled;

   --  ===========================================
   --  Test Execution
   --  ===========================================

   --  Execute a single test
   procedure Execute_Test (
      TC     : in Test_Case;
      Result : out Test_Result)
      with Pre  => TC.Name_Len > 0 and TC.Is_Enabled,
           Post => Result.Status /= Pending and Result.Status /= Running;

   --  Execute all tests in suite
   procedure Execute_All (
      Suite : in out Test_Suite)
      with Pre  => Suite.Count > 0,
           Post => Suite.Stats.Total = Natural (Suite.Count);

   --  Execute tests by category
   procedure Execute_By_Category (
      Suite    : in out Test_Suite;
      Category : in Test_Category);

   --  Execute tests by priority (Critical first)
   procedure Execute_By_Priority (
      Suite : in out Test_Suite);

   --  ===========================================
   --  Result Handling
   --  ===========================================

   --  Update statistics based on result
   procedure Update_Stats (
      Stats  : in out Suite_Stats;
      Result : in Test_Result)
      with Pre  => Stats.Total < Natural'Last,
           Post => Stats.Total = Stats.Total'Old + 1;

   --  Get test result by index
   function Get_Result (
      Suite : Test_Suite;
      Index : Valid_Test_Index) return Test_Result
      with Pre => Index <= Suite.Count;

   --  ===========================================
   --  Reporting
   --  ===========================================

   --  Check if all tests passed
   function All_Tests_Passed (Suite : Test_Suite) return Boolean is
      (Suite.Stats.Failed = 0 and Suite.Stats.Errors = 0 and
       Suite.Stats.Timeouts = 0);

   --  Get total execution time
   function Total_Duration_Ms (Suite : Test_Suite) return Natural;

   --  Format summary line
   procedure Format_Summary (
      Stats  : in Suite_Stats;
      Output : out Message_String;
      Length : out Natural)
      with Post => Length <= Max_Message_Length;

end Test_Executor;
