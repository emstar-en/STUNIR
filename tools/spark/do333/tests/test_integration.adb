--  STUNIR DO-333 Integration Tests
--  End-to-end verification of DO-333 components
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

with Ada.Text_IO;            use Ada.Text_IO;
with Formal_Spec;            use Formal_Spec;
with Spec_Parser;            use Spec_Parser;
with Proof_Obligation;       use Proof_Obligation;
with PO_Manager;             use PO_Manager;
with Verification_Condition; use Verification_Condition;
with VC_Tracker;             use VC_Tracker;
with SPARK_Integration;      use SPARK_Integration;
with Evidence_Generator;     use Evidence_Generator;

procedure Test_Integration is

   Total_Tests  : Natural := 0;
   Passed_Tests : Natural := 0;

   procedure Test (Name : String; Condition : Boolean) is
   begin
      Total_Tests := Total_Tests + 1;
      if Condition then
         Passed_Tests := Passed_Tests + 1;
         Put_Line ("  [PASS] " & Name);
      else
         Put_Line ("  [FAIL] " & Name);
      end if;
   end Test;

begin
   Put_Line ("=== DO-333 Integration Tests ===");
   New_Line;

   --  ================================================================
   --  Test 1: Complete formal verification workflow
   --  ================================================================
   Put_Line ("Test 1: Complete formal verification workflow");

   declare
      --  Formal spec components
      Contract : Contract_Spec;
      Stats    : Parse_Statistics;
      P_Result : Parse_Result;

      --  Sample source with contracts
      Source : constant String :=
        "procedure Check_Altitude (Alt : Altitude_Type)" & ASCII.LF &
        "  with Pre => Alt >= 0," & ASCII.LF &
        "       Post => Current_Alt in 0 .. Max_Alt;" & ASCII.LF;

      --  PO components
      PO_Coll  : PO_Collection;
      PO       : Proof_Obligation_Record;

      --  VC components
      VC_Coll  : VC_Collection;
      VC       : VC_Record;

      Success  : Boolean;
   begin
      --  Step 1: Parse specifications
      Parse_Source_Content (Source, Contract, Stats, P_Result);

      Test ("1.1: Parse source succeeds", P_Result = Parse_Success);
      Test ("1.2: Found preconditions", Contract.Pre_Count > 0);
      Test ("1.3: Found postconditions", Contract.Post_Count > 0);

      --  Step 2: Generate proof obligations
      Initialize (PO_Coll);

      --  PO for precondition
      Create_PO (1, PO_Precondition, DAL_A, "altitude.ads",
                 "Check_Altitude", 2, 8, PO);
      Update_Status (PO, PO_Proved, 100, 500);
      Add_PO (PO_Coll, PO, Success);

      --  PO for postcondition
      Create_PO (2, PO_Postcondition, DAL_A, "altitude.ads",
                 "Check_Altitude", 3, 8, PO);
      Update_Status (PO, PO_Proved, 150, 800);
      Add_PO (PO_Coll, PO, Success);

      --  PO for range check
      Create_PO (3, PO_Range_Check, DAL_B, "altitude.adb",
                 "Check_Altitude", 10, 5, PO);
      Update_Status (PO, PO_Proved, 50, 100);
      Add_PO (PO_Coll, PO, Success);

      Test ("1.4: Created 3 POs", PO_Coll.Count = 3);

      --  Step 3: Track verification conditions
      Initialize (VC_Coll);

      for I in 1 .. 6 loop
         Create_VC (I, (I + 2) / 3, VC);  --  2 VCs per PO
         Update_VC_Status (VC, VC_Valid, I * 50, I * 100);
         Set_VC_Prover (VC, "z3");
         Add_VC (VC_Coll, VC, Success);
      end loop;

      Test ("1.5: Created 6 VCs", VC_Coll.Count = 6);
      Test ("1.6: All VCs discharged", All_Discharged (VC_Coll));

      --  Step 4: Verify coverage
      declare
         PO_Metrics : constant Coverage_Metrics := Get_Metrics (PO_Coll);
         VC_Report  : constant VC_Coverage_Report := Get_Coverage (VC_Coll);
      begin
         Test ("1.7: 100% PO coverage", PO_Metrics.Coverage_Pct = 100);
         Test ("1.8: 100% VC coverage", VC_Report.Coverage_Pct = 100);
         Test ("1.9: All critical proved", All_Critical_Proved (PO_Coll));
      end;
   end;

   New_Line;

   --  ================================================================
   --  Test 2: DAL-based prioritization
   --  ================================================================
   Put_Line ("Test 2: DAL-based prioritization");

   declare
      Coll    : PO_Collection;
      PO      : Proof_Obligation_Record;
      Success : Boolean;
   begin
      Initialize (Coll);

      --  Add POs in mixed order
      Create_PO (1, PO_Assert, DAL_E, "t.ads", "P", 50, 1, PO);
      Add_PO (Coll, PO, Success);

      Create_PO (2, PO_Assert, DAL_A, "t.ads", "P", 10, 1, PO);
      Add_PO (Coll, PO, Success);

      Create_PO (3, PO_Assert, DAL_C, "t.ads", "P", 30, 1, PO);
      Add_PO (Coll, PO, Success);

      Create_PO (4, PO_Assert, DAL_B, "t.ads", "P", 20, 1, PO);
      Add_PO (Coll, PO, Success);

      --  Prioritize by criticality
      Prioritize_By_Criticality (Coll);

      Test ("2.1: First PO is DAL-A",
            Get_PO (Coll, 1).Criticality = DAL_A);
      Test ("2.2: Second PO is DAL-B",
            Get_PO (Coll, 2).Criticality = DAL_B);
      Test ("2.3: Last PO is DAL-E",
            Get_PO (Coll, 4).Criticality = DAL_E);
   end;

   New_Line;

   --  ================================================================
   --  Test 3: Evidence generation pipeline
   --  ================================================================
   Put_Line ("Test 3: Evidence generation pipeline");

   declare
      PO_Coll : PO_Collection;
      VC_Coll : VC_Collection;
      PO      : Proof_Obligation_Record;
      VC      : VC_Record;
      Success : Boolean;

      PO_Report   : String (1 .. Max_Report_Size);
      VC_Report   : String (1 .. Max_Report_Size);
      Cov_Report  : String (1 .. Max_Report_Size);
      Matrix_Rep  : String (1 .. Max_Report_Size);
      Length      : Natural;
      Gen_Result  : Generation_Result;
   begin
      --  Setup test data
      Initialize (PO_Coll);
      Initialize (VC_Coll);

      for I in 1 .. 5 loop
         Create_PO (I, PO_Assert, DAL_C, "test.ads", "Proc", I * 10, 1, PO);
         Update_Status (PO, PO_Proved, I * 20, I * 100);
         Add_PO (PO_Coll, PO, Success);

         Create_VC (I, I, VC);
         Update_VC_Status (VC, VC_Valid, I * 50, I * 100);
         Add_VC (VC_Coll, VC, Success);
      end loop;

      --  Generate all reports
      Generate_PO_Report (PO_Coll, Format_JSON, PO_Report, Length, Gen_Result);
      Test ("3.1: PO report generated", Gen_Result = Gen_Success);

      Generate_VC_Report (VC_Coll, Format_JSON, VC_Report, Length, Gen_Result);
      Test ("3.2: VC report generated", Gen_Result = Gen_Success);

      Generate_Coverage_Report (PO_Coll, VC_Coll, Format_Text,
                                Cov_Report, Length, Gen_Result);
      Test ("3.3: Coverage report generated", Gen_Result = Gen_Success);

      Generate_Compliance_Matrix (PO_Coll, VC_Coll, Format_JSON,
                                  Matrix_Rep, Length, Gen_Result);
      Test ("3.4: Compliance matrix generated", Gen_Result = Gen_Success);
   end;

   New_Line;

   --  ================================================================
   --  Test 4: SPARK integration configuration
   --  ================================================================
   Put_Line ("Test 4: SPARK integration configuration");

   declare
      Config  : SPARK_Config;
      Command : Command_String;
      Length  : Command_Length;
   begin
      --  Test different configurations
      Config := Default_Config;
      Test ("4.1: Default config valid", Is_Valid_Config (Config));

      Config := High_Assurance_Config;
      Test ("4.2: High assurance config valid", Is_Valid_Config (Config));

      --  Build command
      Build_Command ("project.gpr", Default_Config, Command, Length);
      Test ("4.3: Command generated", Length > 0);

      --  Test result analysis
      declare
         Result : SPARK_Result := Empty_Result;
      begin
         Result.Success := True;
         Result.Total_VCs := 100;
         Result.Proved_VCs := 98;
         Result.Unproved_VCs := 2;
         Result.Flow_Errors := 0;

         Test ("4.4: 98% proved", Proof_Percentage (Result) = 98);
         Test ("4.5: Flow passed", Flow_Passed (Result));
         Test ("4.6: Not all proved", not All_Proved (Result));
      end;
   end;

   New_Line;

   --  ================================================================
   --  Test 5: Unproven PO analysis
   --  ================================================================
   Put_Line ("Test 5: Unproven PO analysis");

   declare
      Coll     : PO_Collection;
      PO       : Proof_Obligation_Record;
      Success  : Boolean;
      Analysis : Unproven_Analysis;
   begin
      Initialize (Coll);

      --  Proved critical PO
      Create_PO (1, PO_Precondition, DAL_A, "t.ads", "P", 10, 1, PO);
      Update_Status (PO, PO_Proved, 100, 500);
      Add_PO (Coll, PO, Success);

      --  Unproved critical PO
      Create_PO (2, PO_Postcondition, DAL_A, "t.ads", "P", 20, 1, PO);
      Add_PO (Coll, PO, Success);

      --  Unproved safety PO (DAL-C)
      Create_PO (3, PO_Range_Check, DAL_C, "t.ads", "P", 30, 1, PO);
      Add_PO (Coll, PO, Success);

      --  Unproved non-safety PO (DAL-E)
      Create_PO (4, PO_Assert, DAL_E, "t.ads", "P", 40, 1, PO);
      Add_PO (Coll, PO, Success);

      Analysis := Get_Unproven_Analysis (Coll);

      Test ("5.1: 1 critical unproven", Analysis.Critical_Unproven = 1);
      Test ("5.2: 2 safety unproven", Analysis.Safety_Unproven = 2);
      Test ("5.3: Not all critical proved", not All_Critical_Proved (Coll));
      Test ("5.4: Not all safety proved", not All_Safety_Proved (Coll));
   end;

   --  ================================================================
   --  Summary
   --  ================================================================
   New_Line;
   Put_Line ("=== Integration Test Summary ===");
   Put_Line ("Passed:" & Natural'Image (Passed_Tests) &
             " /" & Natural'Image (Total_Tests));

   if Passed_Tests = Total_Tests then
      Put_Line ("All integration tests PASSED!");
      Put_Line ("DO-333 components verified.");
   else
      Put_Line ("Some integration tests FAILED!");
   end if;

end Test_Integration;
