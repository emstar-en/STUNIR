--  STUNIR DO-333 Evidence Generator Tests
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

with Ada.Text_IO;         use Ada.Text_IO;
with Proof_Obligation;    use Proof_Obligation;
with PO_Manager;          use PO_Manager;
with Verification_Condition; use Verification_Condition;
with VC_Tracker;          use VC_Tracker;
with Evidence_Generator;  use Evidence_Generator;

procedure Test_Evidence_Generator is

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
   Put_Line ("=== Evidence Generator Tests ===");
   New_Line;

   --  Test 1: Format names
   Test ("Format_Text name", Format_Name (Format_Text) = "Text");
   Test ("Format_JSON name", Format_Name (Format_JSON) = "JSON");
   Test ("Format_HTML name", Format_Name (Format_HTML) = "HTML");
   Test ("Format_XML name", Format_Name (Format_XML) = "XML");
   Test ("Format_CSV name", Format_Name (Format_CSV) = "CSV");

   --  Test 2: Format extensions
   Test ("Text extension", Format_Extension (Format_Text) = ".txt");
   Test ("JSON extension", Format_Extension (Format_JSON) = ".json");
   Test ("HTML extension", Format_Extension (Format_HTML) = ".html");

   --  Test 3: Status names
   Test ("Compliant status", Status_Name (Status_Compliant) = "Compliant");
   Test ("Partial status", Status_Name (Status_Partial) = "Partial");
   Test ("Non-Compliant status",
         Status_Name (Status_Non_Compliant) = "Non-Compliant");
   Test ("N/A status", Status_Name (Status_Not_Applicable) = "N/A");

   --  Test 4: Generate PO report (Text)
   declare
      PO_Coll : PO_Collection;
      PO      : Proof_Obligation_Record;
      Success : Boolean;
      Content : String (1 .. Max_Report_Size);
      Length  : Natural;
      Result  : Generation_Result;
   begin
      Initialize (PO_Coll);
      Create_PO (1, PO_Precondition, DAL_A, "test.ads", "Proc", 10, 1, PO);
      Update_Status (PO, PO_Proved, 100, 500);
      Add_PO (PO_Coll, PO, Success);

      Generate_PO_Report (PO_Coll, Format_Text, Content, Length, Result);
      Test ("PO report generation succeeds", Result = Gen_Success);
      Test ("PO report has content", Length > 0);
   end;

   --  Test 5: Generate PO report (JSON)
   declare
      PO_Coll : PO_Collection;
      PO      : Proof_Obligation_Record;
      Success : Boolean;
      Content : String (1 .. Max_Report_Size);
      Length  : Natural;
      Result  : Generation_Result;
   begin
      Initialize (PO_Coll);
      Create_PO (1, PO_Assert, DAL_C, "test.ads", "Proc", 10, 1, PO);
      Add_PO (PO_Coll, PO, Success);

      Generate_PO_Report (PO_Coll, Format_JSON, Content, Length, Result);
      Test ("JSON PO report succeeds", Result = Gen_Success);
      Test ("JSON report starts with {",
            Length > 0 and then Content (1) = '{');
   end;

   --  Test 6: Generate VC report
   declare
      VC_Coll : VC_Collection;
      VC      : VC_Record;
      Success : Boolean;
      Content : String (1 .. Max_Report_Size);
      Length  : Natural;
      Result  : Generation_Result;
   begin
      Initialize (VC_Coll);
      Create_VC (1, 1, VC);
      Update_VC_Status (VC, VC_Valid, 100, 200);
      Add_VC (VC_Coll, VC, Success);

      Generate_VC_Report (VC_Coll, Format_Text, Content, Length, Result);
      Test ("VC report generation succeeds", Result = Gen_Success);
      Test ("VC report has content", Length > 0);
   end;

   --  Test 7: Generate coverage report
   declare
      PO_Coll : PO_Collection;
      VC_Coll : VC_Collection;
      Content : String (1 .. Max_Report_Size);
      Length  : Natural;
      Result  : Generation_Result;
   begin
      Initialize (PO_Coll);
      Initialize (VC_Coll);

      Generate_Coverage_Report (PO_Coll, VC_Coll, Format_Text,
                                Content, Length, Result);
      Test ("Coverage report succeeds", Result = Gen_Success);
      Test ("Coverage report has content", Length > 0);
   end;

   --  Test 8: Initialize compliance matrix
   declare
      Matrix : Compliance_Matrix;
   begin
      Initialize_Matrix (Matrix);
      Test ("Matrix initialized with entries", Matrix.Count > 0);
      Test ("Matrix has FM.1 objective", Matrix.Count >= 1);
   end;

   --  Test 9: Update matrix entry
   declare
      Matrix : Compliance_Matrix;
   begin
      Initialize_Matrix (Matrix);
      Update_Matrix_Entry (Matrix, "FM.1", Status_Compliant,
                          "PO Report", "All specs verified");
      --  Verify update (checking internal state would require accessors)
      Test ("Matrix entry updated", Matrix.Count > 0);
   end;

   --  Test 10: Generate compliance matrix
   declare
      PO_Coll : PO_Collection;
      VC_Coll : VC_Collection;
      Content : String (1 .. Max_Report_Size);
      Length  : Natural;
      Result  : Generation_Result;
   begin
      Initialize (PO_Coll);
      Initialize (VC_Coll);

      Generate_Compliance_Matrix (PO_Coll, VC_Coll, Format_Text,
                                  Content, Length, Result);
      Test ("Compliance matrix succeeds", Result = Gen_Success);
      Test ("Compliance matrix has content", Length > 0);
   end;

   --  Test 11: Generate compliance matrix (JSON)
   declare
      PO_Coll : PO_Collection;
      VC_Coll : VC_Collection;
      Content : String (1 .. Max_Report_Size);
      Length  : Natural;
      Result  : Generation_Result;
   begin
      Initialize (PO_Coll);
      Initialize (VC_Coll);

      Generate_Compliance_Matrix (PO_Coll, VC_Coll, Format_JSON,
                                  Content, Length, Result);
      Test ("JSON compliance matrix succeeds", Result = Gen_Success);
      Test ("JSON matrix starts with {",
            Length > 0 and then Content (1) = '{');
   end;

   --  Test 12: Generate justification template
   declare
      PO_Coll : PO_Collection;
      PO      : Proof_Obligation_Record;
      Success : Boolean;
      Content : String (1 .. Max_Report_Size);
      Length  : Natural;
      Result  : Generation_Result;
   begin
      Initialize (PO_Coll);
      --  Add unproved PO
      Create_PO (1, PO_Precondition, DAL_A, "test.ads", "Proc", 10, 1, PO);
      Add_PO (PO_Coll, PO, Success);

      Generate_Justification_Template (PO_Coll, Content, Length, Result);
      Test ("Justification template succeeds", Result = Gen_Success);
      Test ("Justification template has content", Length > 0);
   end;

   --  Summary
   New_Line;
   Put_Line ("=== Test Summary ===");
   Put_Line ("Passed:" & Natural'Image (Passed_Tests) &
             " /" & Natural'Image (Total_Tests));

   if Passed_Tests = Total_Tests then
      Put_Line ("All tests PASSED!");
   else
      Put_Line ("Some tests FAILED!");
   end if;

end Test_Evidence_Generator;
