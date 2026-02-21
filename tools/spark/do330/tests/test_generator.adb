--  STUNIR DO-330 Package Generator Tests
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

with Ada.Text_IO;           use Ada.Text_IO;
with Ada.Command_Line;
with Templates;             use Templates;
with Template_Engine;       use Template_Engine;
with Data_Collector;        use Data_Collector;
with Package_Generator;     use Package_Generator;

procedure Test_Generator is

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
   --  Test: Set_Default_Config
   --  ============================================================
   procedure Test_Set_Default_Config is
      Config : Package_Config;
   begin
      Set_Default_Config (
         Config    => Config,
         Tool_Name => "test_tool",
         Version   => "1.0.0",
         TQL       => TQL_4,
         DAL       => DAL_C
      );

      Report_Test ("Set default config",
                   Config.Tool_Name_Len = 9 and
                   Config.TQL = TQL_4 and
                   Config.Generate_TOR and
                   Config.Generate_TQP and
                   Config.Generate_TAS);
   end Test_Set_Default_Config;

   --  ============================================================
   --  Test: Initialize_Generator
   --  ============================================================
   procedure Test_Initialize_Generator is
      Config : Package_Config;
      State  : Generator_State;
      Status : Generate_Status;
   begin
      Set_Default_Config (Config, "test", "1.0", TQL_4, DAL_C);
      Initialize_Generator (State, Config, Status);

      Report_Test ("Initialize generator",
                   Status = Success and State.Initialized);
   end Test_Initialize_Generator;

   --  ============================================================
   --  Test: Collect_Qualification_Data
   --  ============================================================
   procedure Test_Collect_Data is
      Config : Package_Config;
      State  : Generator_State;
      Status : Generate_Status;
   begin
      Set_Default_Config (Config, "test", "1.0", TQL_4, DAL_C);
      Initialize_Generator (State, Config, Status);
      Collect_Qualification_Data (State, ".", Status);

      Report_Test ("Collect qualification data",
                   Status = Success and State.Data_Ready);
   end Test_Collect_Data;

   --  ============================================================
   --  Test: Generate_TOR
   --  ============================================================
   procedure Test_Generate_TOR is
      Config : Package_Config;
      State  : Generator_State;
      Status : Generate_Status;
      Output : Output_Content;
      Length : Output_Length_Type;
   begin
      Set_Default_Config (Config, "test", "1.0", TQL_4, DAL_C);
      Initialize_Generator (State, Config, Status);
      Collect_Qualification_Data (State, ".", Status);
      Generate_TOR (State, Output, Length, Status);

      Report_Test ("Generate TOR",
                   Status = Success and Length > 0);
   end Test_Generate_TOR;

   --  ============================================================
   --  Test: Generate_TQP
   --  ============================================================
   procedure Test_Generate_TQP is
      Config : Package_Config;
      State  : Generator_State;
      Status : Generate_Status;
      Output : Output_Content;
      Length : Output_Length_Type;
   begin
      Set_Default_Config (Config, "test", "1.0", TQL_4, DAL_C);
      Initialize_Generator (State, Config, Status);
      Collect_Qualification_Data (State, ".", Status);
      Generate_TQP (State, Output, Length, Status);

      Report_Test ("Generate TQP",
                   Status = Success and Length > 0);
   end Test_Generate_TQP;

   --  ============================================================
   --  Test: Generate_TAS
   --  ============================================================
   procedure Test_Generate_TAS is
      Config : Package_Config;
      State  : Generator_State;
      Status : Generate_Status;
      Output : Output_Content;
      Length : Output_Length_Type;
   begin
      Set_Default_Config (Config, "test", "1.0", TQL_4, DAL_C);
      Initialize_Generator (State, Config, Status);
      Collect_Qualification_Data (State, ".", Status);
      Generate_TAS (State, Output, Length, Status);

      Report_Test ("Generate TAS",
                   Status = Success and Length > 0);
   end Test_Generate_TAS;

   --  ============================================================
   --  Test: Generate_TOR_Traceability
   --  ============================================================
   procedure Test_Generate_Traceability is
      Config : Package_Config;
      State  : Generator_State;
      Status : Generate_Status;
      Output : Output_Content;
      Length : Output_Length_Type;
   begin
      Set_Default_Config (Config, "test", "1.0", TQL_4, DAL_C);
      Initialize_Generator (State, Config, Status);
      Collect_Qualification_Data (State, ".", Status);
      Generate_TOR_Traceability (State, Output, Length, Status);

      Report_Test ("Generate traceability",
                   Status = Success and Length > 0);
   end Test_Generate_Traceability;

   --  ============================================================
   --  Test: Generate_Config_Index
   --  ============================================================
   procedure Test_Generate_Config_Index is
      Config : Package_Config;
      State  : Generator_State;
      Status : Generate_Status;
      Output : Output_Content;
      Length : Output_Length_Type;
   begin
      Set_Default_Config (Config, "test", "1.0", TQL_4, DAL_C);
      Initialize_Generator (State, Config, Status);
      Collect_Qualification_Data (State, ".", Status);
      Generate_Config_Index (State, Output, Length, Status);

      Report_Test ("Generate config index",
                   Status = Success and Length > 0);
   end Test_Generate_Config_Index;

   --  ============================================================
   --  Test: Validate_Package
   --  ============================================================
   procedure Test_Validate_Package is
      Config : Package_Config;
      State  : Generator_State;
      Status : Generate_Status;
      Report : Validation_Report;
   begin
      Set_Default_Config (Config, "test", "1.0", TQL_5, DAL_E);
      Initialize_Generator (State, Config, Status);
      Collect_Qualification_Data (State, ".", Status);
      Validate_Package (State, Report);

      Report_Test ("Validate package",
                   Report.Is_Valid);
   end Test_Validate_Package;

   --  ============================================================
   --  Test: Is_Ready_For_Generation
   --  ============================================================
   procedure Test_Is_Ready is
      Config : Package_Config;
      State  : Generator_State;
      Status : Generate_Status;
   begin
      Set_Default_Config (Config, "test", "1.0", TQL_4, DAL_C);
      Initialize_Generator (State, Config, Status);
      Collect_Qualification_Data (State, ".", Status);

      Report_Test ("Is ready for generation",
                   Is_Ready_For_Generation (State));
   end Test_Is_Ready;

   --  ============================================================
   --  Test: Status_Message
   --  ============================================================
   procedure Test_Status_Message is
      Msg : constant String := Status_Message (Success);
   begin
      Report_Test ("Status message",
                   Msg'Length > 0);
   end Test_Status_Message;

   --  ============================================================
   --  Test: Generate_All_Documents
   --  ============================================================
   procedure Test_Generate_All is
      Config : Package_Config;
      State  : Generator_State;
      Status : Generate_Status;
   begin
      Set_Default_Config (Config, "test", "1.0", TQL_4, DAL_C);
      Initialize_Generator (State, Config, Status);
      Collect_Qualification_Data (State, ".", Status);
      Generate_All_Documents (State, Status);

      Report_Test ("Generate all documents",
                   Status = Success and State.Generated);
   end Test_Generate_All;

begin
   Put_Line ("============================================================");
   Put_Line ("STUNIR DO-330 Package Generator Tests");
   Put_Line ("============================================================");
   Put_Line ("");

   --  Run all tests
   Test_Set_Default_Config;
   Test_Initialize_Generator;
   Test_Collect_Data;
   Test_Generate_TOR;
   Test_Generate_TQP;
   Test_Generate_TAS;
   Test_Generate_Traceability;
   Test_Generate_Config_Index;
   Test_Validate_Package;
   Test_Is_Ready;
   Test_Status_Message;
   Test_Generate_All;

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

end Test_Generator;
