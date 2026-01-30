--  STUNIR DO-330 Template System Tests
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

with Ada.Text_IO;           use Ada.Text_IO;
with Templates;             use Templates;
with Template_Engine;       use Template_Engine;

procedure Test_Templates is

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
   --  Test: TQL_To_String
   --  ============================================================
   procedure Test_TQL_To_String is
      Result : String (1 .. 6);
   begin
      Result := TQL_To_String (TQL_1) & " ";
      Report_Test ("TQL_To_String (TQL_1)", Result (1 .. 5) = "TQL-1");

      Result := TQL_To_String (TQL_5) & " ";
      Report_Test ("TQL_To_String (TQL_5)", Result (1 .. 5) = "TQL-5");
   end Test_TQL_To_String;

   --  ============================================================
   --  Test: DAL_To_String
   --  ============================================================
   procedure Test_DAL_To_String is
      Result : String (1 .. 6);
   begin
      Result := DAL_To_String (DAL_A) & " ";
      Report_Test ("DAL_To_String (DAL_A)", Result (1 .. 5) = "DAL-A");

      Result := DAL_To_String (DAL_E) & " ";
      Report_Test ("DAL_To_String (DAL_E)", Result (1 .. 5) = "DAL-E");
   end Test_DAL_To_String;

   --  ============================================================
   --  Test: Is_Valid_Requirement_ID
   --  ============================================================
   procedure Test_Requirement_ID_Validation is
   begin
      Report_Test ("Valid ID: TOR-001",
                   Is_Valid_Requirement_ID ("TOR-001"));
      Report_Test ("Valid ID: TOR-FUNC-001",
                   Is_Valid_Requirement_ID ("TOR-FUNC-001"));
      Report_Test ("Invalid ID: 123 (no dash)",
                   not Is_Valid_Requirement_ID ("123"));
      Report_Test ("Invalid ID: too short",
                   not Is_Valid_Requirement_ID ("AB"));
   end Test_Requirement_ID_Validation;

   --  ============================================================
   --  Test: Is_Valid_Test_Case_ID
   --  ============================================================
   procedure Test_Case_ID_Validation is
   begin
      Report_Test ("Valid TC ID: TC-001",
                   Is_Valid_Test_Case_ID ("TC-001"));
      Report_Test ("Valid TC ID: TC_FUNC_001",
                   Is_Valid_Test_Case_ID ("TC_FUNC_001"));
      Report_Test ("Invalid TC ID: TEST-001 (no TC prefix)",
                   not Is_Valid_Test_Case_ID ("TEST-001"));
   end Test_Case_ID_Validation;

   --  ============================================================
   --  Test: Template Context Initialization
   --  ============================================================
   procedure Test_Context_Init is
      Context : Template_Context;
   begin
      Initialize_Context (Context, TOR_Template);
      Report_Test ("Context initialization",
                   not Context.Is_Loaded and Context.Var_Count = 0);
   end Test_Context_Init;

   --  ============================================================
   --  Test: Load Template Content
   --  ============================================================
   procedure Test_Load_Template is
      Context : Template_Context;
      Status  : Process_Status;
      Test_Content : constant String := "Hello {{NAME}}!";
   begin
      Initialize_Context (Context, TOR_Template);
      Load_Template_Content (Context, Test_Content, Status);
      Report_Test ("Load template content",
                   Status = Success and Context.Is_Loaded);
   end Test_Load_Template;

   --  ============================================================
   --  Test: Set Variable
   --  ============================================================
   procedure Test_Set_Variable is
      Context : Template_Context;
      Status  : Process_Status;
   begin
      Initialize_Context (Context, TOR_Template);
      Load_Template_Content (Context, "{{VAR}}", Status);

      Set_Variable (Context, "VAR", "VALUE", Status);
      Report_Test ("Set variable",
                   Status = Success and Variable_Exists (Context, "VAR"));
   end Test_Set_Variable;

   --  ============================================================
   --  Test: Process Template
   --  ============================================================
   procedure Test_Process_Template is
      Context : Template_Context;
      Status  : Process_Status;
      Output  : Output_Content;
      Length  : Output_Length_Type;
   begin
      Initialize_Context (Context, TOR_Template);
      Load_Template_Content (Context, "Hello {{NAME}}!", Status);
      Set_Variable (Context, "NAME", "World", Status);
      Process_Template (Context, Output, Length, Status);

      Report_Test ("Process template",
                   Status = Success and Output (1 .. 12) = "Hello World!");
   end Test_Process_Template;

   --  ============================================================
   --  Test: Set DO330 Standard Variables
   --  ============================================================
   procedure Test_Standard_Variables is
      Context : Template_Context;
      Status  : Process_Status;
   begin
      Initialize_Context (Context, TOR_Template);
      Load_Template_Content (Context, "{{TOOL_NAME}} {{TQL_LEVEL}}", Status);

      Set_DO330_Standard_Variables (
         Context   => Context,
         Tool_Name => "test_tool",
         Version   => "1.0.0",
         TQL       => TQL_4,
         DAL       => DAL_C,
         Author    => "Test Author",
         Date      => "2026-01-29",
         Status    => Status
      );

      Report_Test ("Set DO330 standard variables",
                   Status = Success and
                   Variable_Exists (Context, "TOOL_NAME") and
                   Variable_Exists (Context, "TQL_LEVEL"));
   end Test_Standard_Variables;

begin
   Put_Line ("============================================================");
   Put_Line ("STUNIR DO-330 Template System Tests");
   Put_Line ("============================================================");
   Put_Line ("");

   --  Run all tests
   Test_TQL_To_String;
   Test_DAL_To_String;
   Test_Requirement_ID_Validation;
   Test_Case_ID_Validation;
   Test_Context_Init;
   Test_Load_Template;
   Test_Set_Variable;
   Test_Process_Template;
   Test_Standard_Variables;

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

end Test_Templates;
