--  STUNIR DO-330 Data Collector Tests
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

with Ada.Text_IO;           use Ada.Text_IO;
with Ada.Command_Line;
with Templates;             use Templates;
with Data_Collector;        use Data_Collector;

procedure Test_Collector is

   --  Test counters
   Total_Tests  : Natural := 0;
   Passed_Tests : Natural := 0;
   Failed_Tests : Natural := 0;

   --  Helper procedure to report test results
   procedure Report_Test (Name : String; Passed : Boolean) is
   begin
      Total_Tests := Total_Tests + 1;
      if Passed then
         Passed_Tests := Passed_Tests + 1;
         Put_Line ("  [PASS] " & Name);
      else
         Failed_Tests := Failed_Tests + 1;
         Put_Line ("  [FAIL] " & Name);
      end if;
   end Report_Test;

   --  ============================================================
   --  Test: Initialize_Tool_Data
   --  ============================================================
   procedure Test_Initialize_Tool_Data is
      Data : Tool_Data;
   begin
      Initialize_Tool_Data (
         Data      => Data,
         Tool_Name => "test_tool",
         Version   => "1.0.0",
         TQL       => TQL_4,
         DAL       => DAL_C
      );

      Report_Test ("Initialize tool data",
                   Data.Tool_Name_Len = 9 and
                   Data.Version_Len = 5 and
                   Data.TQL = TQL_4 and
                   Data.DAL = DAL_C);
   end Test_Initialize_Tool_Data;

   --  ============================================================
   --  Test: Collect_DO331_Data
   --  ============================================================
   procedure Test_Collect_DO331_Data is
      Data   : Tool_Data;
      Status : Collect_Status;
   begin
      Initialize_Tool_Data (Data, "test", "1.0", TQL_5, DAL_E);
      Collect_DO331_Data (Data, "./nonexistent", Status);

      Report_Test ("Collect DO-331 data",
                   Status = Success and Data.DO331.Available);
   end Test_Collect_DO331_Data;

   --  ============================================================
   --  Test: Collect_DO332_Data
   --  ============================================================
   procedure Test_Collect_DO332_Data is
      Data   : Tool_Data;
      Status : Collect_Status;
   begin
      Initialize_Tool_Data (Data, "test", "1.0", TQL_5, DAL_E);
      Collect_DO332_Data (Data, "./nonexistent", Status);

      Report_Test ("Collect DO-332 data",
                   Status = Success and Data.DO332.Available);
   end Test_Collect_DO332_Data;

   --  ============================================================
   --  Test: Collect_DO333_Data
   --  ============================================================
   procedure Test_Collect_DO333_Data is
      Data   : Tool_Data;
      Status : Collect_Status;
   begin
      Initialize_Tool_Data (Data, "test", "1.0", TQL_5, DAL_E);
      Collect_DO333_Data (Data, "./nonexistent", Status);

      Report_Test ("Collect DO-333 data",
                   Status = Success and Data.DO333.Available);
   end Test_Collect_DO333_Data;

   --  ============================================================
   --  Test: Collect_All_Data
   --  ============================================================
   procedure Test_Collect_All_Data is
      Data   : Tool_Data;
      Status : Collect_Status;
   begin
      Initialize_Tool_Data (Data, "test", "1.0", TQL_5, DAL_E);
      Collect_All_Data (Data, ".", Status);

      Report_Test ("Collect all data",
                   Status = Success and
                   Data.DO331.Available and
                   Data.DO332.Available and
                   Data.DO333.Available);
   end Test_Collect_All_Data;

   --  ============================================================
   --  Test: Is_Data_Complete
   --  ============================================================
   procedure Test_Is_Data_Complete is
      Data   : Tool_Data;
      Status : Collect_Status;
   begin
      Initialize_Tool_Data (Data, "test", "1.0", TQL_5, DAL_E);
      Collect_All_Data (Data, ".", Status);

      Report_Test ("Is data complete",
                   Is_Data_Complete (Data));
   end Test_Is_Data_Complete;

   --  ============================================================
   --  Test: Meets_TQL_Requirements (TQL-5)
   --  ============================================================
   procedure Test_Meets_TQL5_Requirements is
      Data   : Tool_Data;
      Status : Collect_Status;
   begin
      Initialize_Tool_Data (Data, "test", "1.0", TQL_5, DAL_E);
      Collect_All_Data (Data, ".", Status);

      Report_Test ("Meets TQL-5 requirements (always true)",
                   Meets_TQL_Requirements (Data, TQL_5));
   end Test_Meets_TQL5_Requirements;

   --  ============================================================
   --  Test: Calculate_Qualification_Score
   --  ============================================================
   procedure Test_Qualification_Score is
      Data  : Tool_Data;
      Score : Coverage_Percentage;
   begin
      Initialize_Tool_Data (Data, "test", "1.0", TQL_5, DAL_E);
      Data.DO331.Data_Valid := True;
      Data.DO331.Total_Coverage := 50.0;
      Data.Tests.Data_Valid := True;
      Data.Tests.Statement_Cov := 80.0;

      Score := Calculate_Qualification_Score (Data);

      Report_Test ("Calculate qualification score",
                   Score >= 0.0 and Score <= 100.0);
   end Test_Qualification_Score;

   --  ============================================================
   --  Test: Generate_Data_Summary
   --  ============================================================
   procedure Test_Generate_Summary is
      Data    : Tool_Data;
      Summary : Value_String;
      Length  : Value_Length_Type;
   begin
      Initialize_Tool_Data (Data, "test_tool", "1.0", TQL_4, DAL_C);
      Generate_Data_Summary (Data, Summary, Length);

      Report_Test ("Generate data summary",
                   Length > 0 and Length <= Max_Value_Length);
   end Test_Generate_Summary;

   --  ============================================================
   --  Test: Status_Message
   --  ============================================================
   procedure Test_Status_Message is
      Msg : constant String := Status_Message (Success);
   begin
      Report_Test ("Status message",
                   Msg'Length > 0);
   end Test_Status_Message;

begin
   Put_Line ("============================================================");
   Put_Line ("STUNIR DO-330 Data Collector Tests");
   Put_Line ("============================================================");
   Put_Line ("");

   --  Run all tests
   Test_Initialize_Tool_Data;
   Test_Collect_DO331_Data;
   Test_Collect_DO332_Data;
   Test_Collect_DO333_Data;
   Test_Collect_All_Data;
   Test_Is_Data_Complete;
   Test_Meets_TQL5_Requirements;
   Test_Qualification_Score;
   Test_Generate_Summary;
   Test_Status_Message;

   --  Summary
   Put_Line ("");
   Put_Line ("============================================================");
   Put_Line ("Test Summary:");
   Put_Line ("  Total:  " & Natural'Image (Total_Tests));
   Put_Line ("  Passed: " & Natural'Image (Passed_Tests));
   Put_Line ("  Failed: " & Natural'Image (Failed_Tests));
   Put_Line ("============================================================");

   if Failed_Tests > 0 then
      Ada.Command_Line.Set_Exit_Status (1);
   end if;

end Test_Collector;
