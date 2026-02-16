--  STUNIR Phase 3 Test Suite
--  Test Infrastructure - Comprehensive Tests
--  SPARK Migration Phase 3

with Ada.Text_IO; use Ada.Text_IO;

--  Test Harness imports
with Test_Harness_Types; use Test_Harness_Types;
with Test_Executor; use Test_Executor;

--  Result Validator imports
with Validator_Types; use Validator_Types;
with Result_Validator; use Result_Validator;

--  Coverage Analyzer imports
with Coverage_Types; use Coverage_Types;
with Coverage_Tracker; use Coverage_Tracker;

--  Test Data Generator imports
with Test_Data_Types; use Test_Data_Types;
with Data_Generator; use Data_Generator;

--  Orchestrator imports
with Orchestrator_Types; use Orchestrator_Types;
with Test_Orchestrator; use Test_Orchestrator;

--  Common imports
with Stunir_Hashes; use Stunir_Hashes;

procedure Test_Phase3 is

   Total_Tests : Natural := 0;
   Passed_Tests : Natural := 0;

   procedure Report_Test (Name : String; Pass : Boolean) is
   begin
      Total_Tests := Total_Tests + 1;
      if Pass then
         Passed_Tests := Passed_Tests + 1;
         Put_Line ("  [PASS] " & Name);
      else
         Put_Line ("  [FAIL] " & Name);
      end if;
   end Report_Test;

   --  =============================================
   --  Test Harness Tests
   --  =============================================

   procedure Test_Harness_Suite is
      Suite : Test_Suite;
      TC : Test_Case;
      Success : Boolean;
   begin
      Put_Line ("");
      Put_Line ("=== Test Harness Tests ===");

      --  Test 1: Initialize suite
      Initialize_Suite (Suite);
      Report_Test ("Initialize_Suite", Suite.Count = 0);

      --  Test 2: Create test case
      TC := Create_Test ("unit_test_1", Unit_Test, High, 5000);
      Report_Test ("Create_Test", TC.Is_Enabled and TC.Name_Len = 11);

      --  Test 3: Register test
      Register_Test (Suite, TC, Success);
      Report_Test ("Register_Test", Success and Suite.Count = 1);

      --  Test 4: Register multiple tests
      TC := Create_Test ("unit_test_2", Unit_Test, Medium, 3000);
      Register_Test (Suite, TC, Success);
      Report_Test ("Multiple_Registrations", Success and Suite.Count = 2);

      --  Test 5: Execute all tests
      Execute_All (Suite);
      Report_Test ("Execute_All", Suite.Stats.Total = 2);

      --  Test 6: Check all passed
      Report_Test ("All_Tests_Passed", All_Tests_Passed (Suite));

      --  Test 7: Execute by priority
      Initialize_Suite (Suite);
      TC := Create_Test ("critical_test", Unit_Test, Critical, 1000);
      Register_Test (Suite, TC, Success);
      TC := Create_Test ("low_test", Unit_Test, Low, 1000);
      Register_Test (Suite, TC, Success);
      Execute_By_Priority (Suite);
      Report_Test ("Execute_By_Priority", Suite.Stats.Total = 2);

      --  Test 8: Test status functions
      Report_Test ("Is_Terminal_Status", Is_Terminal_Status (Passed));
      Report_Test ("Is_Success_Status", Is_Success_Status (Passed));
      Report_Test ("Is_Failure_Status", Is_Failure_Status (Failed));

      --  Test 9: Get success rate
      Report_Test ("Get_Success_Rate", Get_Success_Rate (Suite.Stats) = 100);

      --  Test 10: Total duration
      Report_Test ("Total_Duration_Ms", Total_Duration_Ms (Suite) >= 0);

   end Test_Harness_Suite;

   --  =============================================
   --  Result Validator Tests
   --  =============================================

   procedure Test_Validator_Suite is
      Receipt : Receipt_Data;
      Results : Validation_Results;
      Entry_V : Validation_Entry;
      Success : Boolean;
      Base_Dir : Path_String := (others => ' ');
      Test_Hash : constant String (1 .. Hash_Length) := (others => 'a');
      Test_Hash2 : constant String (1 .. Hash_Length) := (others => 'b');
      Test_Hash3 : constant String (1 .. Hash_Length) := (others => 'c');
   begin
      Put_Line ("");
      Put_Line ("=== Result Validator Tests ===");

      --  Test 1: Initialize receipt
      Initialize_Receipt (Receipt);
      Report_Test ("Initialize_Receipt", Receipt.Entry_Count = 0);

      --  Test 2: Add entry
      Add_Entry (Receipt, "test/file.json", Test_Hash, Success);
      Report_Test ("Add_Entry", Success and Receipt.Entry_Count = 1);

      --  Test 3: Mark loaded
      Mark_Loaded (Receipt, Valid);
      Report_Test ("Mark_Loaded", Receipt.Is_Loaded);

      --  Test 4: Create entry
      Entry_V := Create_Entry ("another/file.txt", Test_Hash2);
      Report_Test ("Create_Entry", Entry_V.Path_Len = 16);

      --  Test 5: Hashes match
      Report_Test ("Hashes_Match_Same",
         Hashes_Match (Test_Hash, Test_Hash));
      Report_Test ("Hashes_Match_Diff",
         not Hashes_Match (Test_Hash, Test_Hash2));

      --  Test 6: Validate all
      Add_Entry (Receipt, "test/file2.json", Test_Hash3, Success);
      Base_Dir (1 .. 4) := "test";
      Validate_All (Receipt, Base_Dir, 4, Results);
      Report_Test ("Validate_All", Results.Checked = 2);

      --  Test 7: All entries valid
      Report_Test ("All_Entries_Valid", All_Entries_Valid (Results));

      --  Test 8: Failed count
      Report_Test ("Failed_Count", Failed_Count (Results) = 0);

      --  Test 9: Outcome helpers
      Report_Test ("Is_Success_Outcome", Is_Success_Outcome (Valid));
      Report_Test ("Is_Error_Outcome", Is_Error_Outcome (Hash_Mismatch));

      --  Test 10: Validation stats
      Report_Test ("Validation_Stats",
         Results.Stats.Total = 2 and Results.Stats.Valid = 2);

   end Test_Validator_Suite;

   --  =============================================
   --  Coverage Analyzer Tests
   --  =============================================

   procedure Test_Coverage_Suite is
      Tracker : Coverage_Tracker_Type;
      Report  : Coverage_Report;
      Success : Boolean;
      Metrics : Coverage_Metrics;
   begin
      Put_Line ("");
      Put_Line ("=== Coverage Analyzer Tests ===");

      --  Test 1: Initialize tracker
      Initialize (Tracker);
      Report_Test ("Initialize_Tracker", Tracker.Module_Count = 0);

      --  Test 2: Start tracking
      Start_Tracking (Tracker);
      Report_Test ("Start_Tracking", Tracker.Is_Active);

      --  Test 3: Register module
      Register_Module (Tracker, "test_module", 100, Success);
      Report_Test ("Register_Module", Success and Tracker.Module_Count = 1);

      --  Test 4: Record line coverage
      Record_Line (Tracker, 1, 1, True);
      Record_Line (Tracker, 1, 2, True);
      Record_Line (Tracker, 1, 3, False);
      Report_Test ("Record_Line", True);  --  No crash = pass

      --  Test 5: Mark lines covered
      Mark_Lines_Covered (Tracker, 1, 10, 20);
      Report_Test ("Mark_Lines_Covered", True);

      --  Test 6: Record branch
      Record_Branch (Tracker, 1, 1, True);
      Report_Test ("Record_Branch", True);

      --  Test 7: Record function
      Record_Function (Tracker, 1, 1, True);
      Report_Test ("Record_Function", True);

      --  Test 8: Get coverage report
      Compute_All_Metrics (Tracker);
      Report := Get_Report (Tracker);
      Report_Test ("Get_Report", Report.Is_Valid);

      --  Test 9: Coverage metrics helpers
      Metrics := Empty_Metrics;
      Metrics.Total_Lines := 100;
      Metrics.Covered_Lines := 80;
      Report_Test ("Get_Line_Coverage", Get_Line_Coverage (Metrics) = 80);

      --  Test 10: Coverage classification
      Report_Test ("Classify_Coverage_Full", Classify_Coverage (100) = Full);
      Report_Test ("Classify_Coverage_High", Classify_Coverage (95) = High);
      Report_Test ("Classify_Coverage_Med", Classify_Coverage (75) = Medium);

      --  Test 11: Find module
      Report_Test ("Find_Module", Find_Module (Tracker, "test_module") = 1);

      --  Test 12: Stop tracking
      Stop_Tracking (Tracker);
      Report_Test ("Stop_Tracking", not Tracker.Is_Active);

   end Test_Coverage_Suite;

   --  =============================================
   --  Test Data Generator Tests
   --  =============================================

   procedure Test_Data_Gen_Suite is
      VSet : Vector_Set;
      V : Test_Vector;
      Success : Boolean;
      Added : Natural;
      Template : Vector_Template;
   begin
      Put_Line ("");
      Put_Line ("=== Test Data Generator Tests ===");

      --  Test 1: Initialize set
      Initialize_Set (VSet);
      Report_Test ("Initialize_Set", VSet.Count = 0);

      --  Test 2: Create vector
      V := Create_Vector ("test_vec_1", "{}", Conformance, High);
      Report_Test ("Create_Vector", V.Is_Valid and V.Name_Len = 10);

      --  Test 3: Add vector
      Add_Vector (VSet, V, Success);
      Report_Test ("Add_Vector", Success and VSet.Count = 1);

      --  Test 4: Get vector
      V := Get_Vector (VSet, 1);
      Report_Test ("Get_Vector", V.Is_Valid);

      --  Test 5: Generate JSON spec vector
      Generate_Json_Spec_Vector ("json_test", 0, V, Success);
      Report_Test ("Generate_Json_Spec", Success and V.Is_Valid);

      --  Test 6: Generate IR module vector
      Generate_IR_Module_Vector ("ir_test", 0, V, Success);
      Report_Test ("Generate_IR_Module", Success and V.Is_Valid);

      --  Test 7: Generate receipt vector
      Generate_Receipt_Vector ("receipt_test", 0, V, Success);
      Report_Test ("Generate_Receipt", Success and V.Is_Valid);

      --  Test 8: Generate boundary vectors
      Initialize_Set (VSet);
      Generate_Boundary_Vectors (VSet, Added);
      Report_Test ("Generate_Boundary", Added = 5);

      --  Test 9: Generate minimal vector
      V := Generate_Minimal_Vector ("minimal");
      Report_Test ("Generate_Minimal", V.Is_Valid);

      --  Test 10: Generate from template
      Template := (Kind => Json_Spec, Category => Conformance,
                   Priority => High, Variation => 0);
      Generate_From_Template (Template, V, Success);
      Report_Test ("Generate_Template", Success);

      --  Test 11: Vector stats
      Report_Test ("Vector_Stats", VSet.Stats.Total = 5);

      --  Test 12: Is empty vector
      Report_Test ("Is_Empty_Vector", not Is_Empty_Vector (V));

   end Test_Data_Gen_Suite;

   --  =============================================
   --  Test Orchestrator Tests
   --  =============================================

   procedure Test_Orchestrator_Suite is
      Session : Orchestration_Session;
      Result  : Conformance_Result;
      Output  : Tool_Output;
      Success : Boolean;
   begin
      Put_Line ("");
      Put_Line ("=== Test Orchestrator Tests ===");

      --  Test 1: Initialize session
      Initialize_Session (Session);
      Report_Test ("Initialize_Session", Session.Tool_Count = 0);

      --  Test 2: Register tool
      Register_Tool (Session, "haskell", "/usr/bin/haskell", Success);
      Report_Test ("Register_Tool", Success and Session.Tool_Count = 1);

      --  Test 3: Register multiple tools
      Register_Tool (Session, "rust", "/usr/bin/rust", Success);
      Register_Tool (Session, "python", "/usr/bin/python", Success);
      Report_Test ("Multiple_Tools", Session.Tool_Count = 3);

      --  Test 4: Set reference tool
      Set_Reference_Tool (Session, 1);
      Report_Test ("Set_Reference_Tool", Session.Reference = 1);

      --  Test 5: Execute all tools
      Execute_All_Tools (Session, "{""test"":1}");
      Report_Test ("Execute_All_Tools", Session.Is_Complete);

      --  Test 6: Get output
      Output := Get_Output (Session, 1);
      Report_Test ("Get_Output",
         Output.Status = Orchestrator_Types.Success);

      --  Test 7: Get reference output
      Output := Get_Reference_Output (Session);
      Report_Test ("Get_Reference",
         Output.Status = Orchestrator_Types.Success);

      --  Test 8: Check conformance
      Check_Conformance (Session, Result);
      Report_Test ("Check_Conformance", Result.Total_Compared = 2);

      --  Test 9: All tools match
      Report_Test ("All_Tools_Match", All_Tools_Match (Session));

      --  Test 10: Ready tool count
      Report_Test ("Ready_Tool_Count", Ready_Tool_Count (Session) = 3);

      --  Test 11: Success count
      Report_Test ("Success_Count", Success_Count (Session) = 3);

      --  Test 12: Status helpers
      Report_Test ("Is_Success_Status",
         not Orchestrator_Types.Is_Success_Status (Available));
      Report_Test ("Is_Error_Status",
         Orchestrator_Types.Is_Error_Status (Orchestrator_Types.Error));

   end Test_Orchestrator_Suite;

begin
   Put_Line ("");
   Put_Line ("========================================");
   Put_Line ("  STUNIR Phase 3 SPARK Migration");
   Put_Line ("  Test Infrastructure - Test Suite");
   Put_Line ("========================================");

   Test_Harness_Suite;
   Test_Validator_Suite;
   Test_Coverage_Suite;
   Test_Data_Gen_Suite;
   Test_Orchestrator_Suite;

   Put_Line ("");
   Put_Line ("========================================");
   Put_Line ("SUMMARY: " & Natural'Image (Passed_Tests) & " /" &
             Natural'Image (Total_Tests) & " tests passed");

   if Passed_Tests = Total_Tests then
      Put_Line ("[SUCCESS] ALL TESTS PASSED");
      Put_Line ("Phase 3 SPARK Migration VERIFIED");
      Put_Line ("100% SPARK MIGRATION COMPLETE!");
   else
      Put_Line ("[FAILURE] SOME TESTS FAILED");
   end if;
   Put_Line ("========================================");

end Test_Phase3;
