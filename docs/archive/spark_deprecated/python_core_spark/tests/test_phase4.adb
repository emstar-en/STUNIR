--  STUNIR Phase 4 Test Suite
--  Tool Integration Component Tests
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

with Ada.Text_IO; use Ada.Text_IO;
with Report_Types;
with Report_Formatter;
with External_Tool;
with DO331_Types;
with DO331_Interface;
with DO332_Types;
with DO332_Interface;
with DO333_Types;
with DO333_Interface;
with Package_Types;
with Artifact_Collector;
with Trace_Matrix;

procedure Test_Phase4 is

   Total_Tests  : Natural := 0;
   Passed_Tests : Natural := 0;
   Failed_Tests : Natural := 0;

   procedure Assert (Condition : Boolean; Test_Name : String) is
   begin
      Total_Tests := Total_Tests + 1;
      if Condition then
         Passed_Tests := Passed_Tests + 1;
         Put_Line ("  [PASS] " & Test_Name);
      else
         Failed_Tests := Failed_Tests + 1;
         Put_Line ("  [FAIL] " & Test_Name);
      end if;
   end Assert;

   procedure Test_Report_Generator is
      use Report_Types;
      use Report_Formatter;
      Data   : Report_Data;
      Output : Output_Buffer;
      Length : Output_Length;
      Status : Report_Status;
   begin
      Put_Line ("Testing Report Generator...");
      Initialize_Report (Data, "stunir.test.v1", "Test Report", JSON_Format, Status);
      Assert (Status = Success and Data.Is_Valid, "Initialize report");
      Add_String_Item (Data, "name", "test_value", Status);
      Assert (Status = Success and Data.Flat_Count = 1, "Add string item");
      Add_Integer_Item (Data, "count", 42, Status);
      Assert (Status = Success and Data.Flat_Count = 2, "Add integer item");
      Add_Boolean_Item (Data, "active", True, Status);
      Assert (Status = Success and Data.Flat_Count = 3, "Add boolean item");
      Generate_JSON_Report (Data, Output, Length, Status);
      Assert (Status = Success and Length > 0, "Generate JSON report");
      Data.Format := Text_Format;
      Generate_Text_Report (Data, Output, Length, Status);
      Assert (Status = Success and Length > 0, "Generate Text report");
      Generate_HTML_Report (Data, Output, Length, Status);
      Assert (Status = Success and Length > 0, "Generate HTML report");
      Generate_XML_Report (Data, Output, Length, Status);
      Assert (Status = Success and Length > 0, "Generate XML report");
      Add_Section (Data, "Details", Status);
      Assert (Status = Success and Data.Section_Total = 1, "Add section");
      Clear_Report (Data);
      Assert (Data.Flat_Count = 0 and Data.Section_Total = 0, "Clear report");
   end Test_Report_Generator;

   procedure Test_External_Tool is
      use External_Tool;
      Registry : Tool_Registry;
      Item     : Tool_Item;
      Result   : Command_Result;
      Args     : Argument_Array;
      Arg_Cnt  : Argument_Count;
      Status   : Tool_Status;
   begin
      Put_Line ("Testing External Tool Interface...");
      Initialize_Registry (Registry);
      Assert (Registry.Initialized and Registry.Count = 0, "Initialize registry");
      Discover_Tool ("gnatprove", Item, Status);
      Assert (Status = Success and Item.Available, "Discover gnatprove");
      Register_Tool (Registry, Item, Status);
      Assert (Status = Success and Registry.Count = 1, "Register tool");
      declare
         Idx : Tool_Count;
      begin
         Idx := Find_Tool (Registry, "gnatprove");
         Assert (Idx = 1, "Find tool");
      end;
      Build_Arguments (Args, Arg_Cnt, "--version");
      Assert (Arg_Cnt = 1, "Build arguments");
      Execute_Command ("echo", Args, Arg_Cnt, 1000, Result, Status);
      Assert (Status = Success, "Execute command");
      Get_Tool_Version (Item, Status);
      Assert (Status = Success and Item.Verified, "Get tool version");
      Assert (Tool_Exists (Item) = Item.Available, "Tool exists");
      declare
         Msg : constant String := Status_Message (Success);
      begin
         Assert (Msg'Length > 0, "Status message");
      end;
   end Test_External_Tool;

   procedure Test_DO331_Integration is
      use DO331_Types;
      use DO331_Interface;
      Config : Transform_Config;
      Result : DO331_Result;
      Status : DO331_Status;
   begin
      Put_Line ("Testing DO-331 Integration...");
      Initialize_Config (Config, "asm/ir", "models/do331", DAL_C);
      Assert (Config.IR_Path_Len > 0 and Config.DAL = DAL_C, "Initialize DO-331 config");
      Transform_To_SysML (Config, Result, Status);
      Assert (Status = Success and Result.Success, "Transform to SysML");
      Add_Model (Result, "test_module", "models/test.sysml", Block_Model, Status);
      Assert (Status = Success and Result.Model_Total = 2, "Add model");
      Add_Coverage_Item (Result, "STMT_001", Statement_Coverage, True, Status);
      Assert (Status = Success and Result.Coverage_Total = 1, "Add coverage item");
      Add_Trace_Link (Result, "REQ-002", "IMPL-002", Forward, Status);
      Assert (Status = Success and Result.Trace_Total = 1, "Add trace link");
      Assert (Validate_Model_Completeness (Result), "Validate model completeness");
      declare
         Pct : Percentage_Type;
      begin
         Pct := Calculate_Coverage_Percentage (Result);
         Assert (Pct = 100.0, "Calculate coverage percentage");
      end;
      Assert (Meets_DAL_Requirements (Result, DAL_C), "Meets DAL requirements");
      Finalize_Result (Result);
      Assert (Result.Is_Complete, "Finalize DO-331 result");
      declare
         Msg : constant String := DO331_Types.Status_Message (Success);
      begin
         Assert (Msg'Length > 0, "DO-331 status message");
      end;
   end Test_DO331_Integration;

   procedure Test_DO332_Integration is
      use DO332_Types;
      use DO332_Interface;
      Config : Analysis_Config;
      Result : DO332_Result;
      Status : DO332_Status;
   begin
      Put_Line ("Testing DO-332 Integration...");
      Initialize_Config (Config, "asm/ir", "receipts/do332", 8);
      Assert (Config.IR_Path_Len > 0 and Config.Max_Depth = 8, "Initialize DO-332 config");
      Analyze_OOP (Config, Result, Status);
      Assert (Status = Success and Result.Success, "Analyze OOP");
      Add_Class (Result, "MyClass", "BaseClass", Single_Inheritance, 2, Status);
      Assert (Status = Success and Result.Class_Total = 1, "Add class");
      Add_Polymorphic_Call (Result, "MyClass", "virtual_method", True, Status);
      Assert (Status = Success and Result.Poly_Total = 1, "Add polymorphic call");
      Assert (Check_Depth_Limits (Result, 8), "Check depth limits");
      Assert (Check_Coupling_Limits (Result, 10), "Check coupling limits");
      Assert (All_Polymorphism_Safe (Result), "All polymorphism safe");
      Finalize_Result (Result, 8, 10);
      Assert (Result.Inheritance_OK, "Finalize DO-332 result");
      declare
         Msg : constant String := DO332_Types.Status_Message (Success);
      begin
         Assert (Msg'Length > 0, "DO-332 status message");
      end;
   end Test_DO332_Integration;

   procedure Test_DO333_Integration is
      use DO333_Types;
      use DO333_Interface;
      Config : Verify_Config;
      Result : DO333_Result;
      Status : DO333_Status;
   begin
      Put_Line ("Testing DO-333 Integration...");
      Initialize_Config (Config, "src", "gnatprove", "default.gpr");
      Assert (Config.Source_Len > 0, "Initialize DO-333 config");
      Run_Verification (Config, Result, Status);
      Assert (Status = Success and Result.Success, "Run verification");
      Add_VC (Result, "range_check_1", "main.adb", 25, 1, Range_Check, Proven, Status);
      Assert (Status = Success and Result.VC_Total = 4, "Add VC");
      Add_PO (Result, "proc_validate", "main.adb", 5, 5, Status);
      Assert (Status = Success and Result.PO_Total = 1, "Add PO");
      Finalize_Result (Result);
      declare
         Rate : Percentage_Type;
      begin
         Rate := Result.Proof_Rate;
         Assert (Rate >= 0.0 and Rate <= 100.0, "Calculate proof rate");
      end;
      Assert (All_VCs_Proven (Result), "All VCs proven");
      Assert (Meets_Requirements (Result, 90.0), "Meets requirements");
      declare
         Msg : constant String := DO333_Types.Status_Message (Success);
      begin
         Assert (Msg'Length > 0, "DO-333 status message");
      end;
   end Test_DO333_Integration;

   procedure Test_Compliance_Package is
      use Package_Types;
      use Artifact_Collector;
      use Trace_Matrix;
      Config   : Collector_Config;
      Comp_Pkg : Compliance_Package := Null_Compliance_Package;
      Output   : Output_Buffer;
      Length   : Output_Length;
      Status   : Package_Status;
   begin
      Put_Line ("Testing Compliance Package...");
      Initialize_Config (Config, "/home/ubuntu/stunir_repo", "certification_package");
      Assert (Config.Base_Dir_Len > 0, "Initialize collector config");
      Add_Artifact (Comp_Pkg, "main.adb", "src/main.adb", Source_Code, Status => Status);
      Assert (Status = Success and Comp_Pkg.Artifact_Total = 1, "Add artifact");
      Add_Trace_Link (Comp_Pkg, "REQ-001", "IMPL-001", Req_To_Design, True, Status);
      Assert (Status = Success and Comp_Pkg.Trace_Total = 1, "Add trace link");
      Assert (Trace_Exists (Comp_Pkg, "REQ-001", "IMPL-001"), "Trace exists");
      Generate_Text_Matrix (Comp_Pkg, Output, Length, Status);
      Assert (Status = Success and Length > 0, "Generate text matrix");
      Generate_HTML_Matrix (Comp_Pkg, Output, Length, Status);
      Assert (Status = Success and Length > 0, "Generate HTML matrix");
      Generate_CSV_Matrix (Comp_Pkg, Output, Length, Status);
      Assert (Status = Success and Length > 0, "Generate CSV matrix");
      Assert (Count_Verified_Traces (Comp_Pkg) = 1, "Count verified traces");
      declare
         Cov : Float;
      begin
         Cov := Calculate_Trace_Coverage (Comp_Pkg);
         Assert (Cov = 100.0, "Calculate trace coverage");
      end;
      Assert (Has_Complete_Traceability (Comp_Pkg), "Has complete traceability");
      Assert (Verify_All_Artifacts (Comp_Pkg), "Verify all artifacts");
      Assert (Verify_All_Traces (Comp_Pkg), "Verify all traces");
      declare
         Msg : constant String := Package_Types.Status_Message (Success);
      begin
         Assert (Msg'Length > 0, "Package status message");
      end;
      Assert (TQL_Name (TQL_1) = "TQL-1", "TQL name");
      Assert (Package_Types.DAL_Name (DAL_A) = "DAL-A", "DAL name");
   end Test_Compliance_Package;

begin
   Put_Line ("========================================");
   Put_Line ("STUNIR Phase 4 Test Suite");
   Put_Line ("Tool Integration Components");
   Put_Line ("========================================");
   New_Line;
   Test_Report_Generator;
   New_Line;
   Test_External_Tool;
   New_Line;
   Test_DO331_Integration;
   New_Line;
   Test_DO332_Integration;
   New_Line;
   Test_DO333_Integration;
   New_Line;
   Test_Compliance_Package;
   New_Line;
   Put_Line ("========================================");
   Put_Line ("Test Summary");
   Put_Line ("========================================");
   Put_Line ("Total:  " & Natural'Image (Total_Tests));
   Put_Line ("Passed: " & Natural'Image (Passed_Tests));
   Put_Line ("Failed: " & Natural'Image (Failed_Tests));
   Put_Line ("========================================");
   if Failed_Tests = 0 then
      Put_Line ("All tests passed!");
   else
      Put_Line ("Some tests failed!");
   end if;
end Test_Phase4;
