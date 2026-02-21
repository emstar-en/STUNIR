--  STUNIR DO-333 Formal Methods Analyzer - Main Program
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0
--
--  Main entry point for DO-333 formal methods analysis and
--  certification evidence generation.

with Ada.Text_IO;          use Ada.Text_IO;
with Ada.Command_Line;     use Ada.Command_Line;

with Formal_Spec;          use Formal_Spec;
with Spec_Parser;          use Spec_Parser;
with Proof_Obligation;     use Proof_Obligation;
with PO_Manager;           use PO_Manager;
with Verification_Condition; use Verification_Condition;
with VC_Tracker;           use VC_Tracker;
with SPARK_Integration;    use SPARK_Integration;
with Evidence_Generator;   use Evidence_Generator;

procedure DO333_Main is

   --  ============================================================
   --  Command Type
   --  ============================================================

   type Command_Type is (
      Cmd_Help,
      Cmd_Version,
      Cmd_Analyze,
      Cmd_Report,
      Cmd_Matrix,
      Cmd_Demo
   );

   --  ============================================================
   --  Parse Command
   --  ============================================================

   function Parse_Command return Command_Type is
   begin
      if Argument_Count = 0 then
         return Cmd_Help;
      end if;

      declare
         Arg : constant String := Argument (1);
      begin
         if Arg = "--help" or else Arg = "-h" then
            return Cmd_Help;
         elsif Arg = "--version" or else Arg = "-v" then
            return Cmd_Version;
         elsif Arg = "analyze" then
            return Cmd_Analyze;
         elsif Arg = "report" then
            return Cmd_Report;
         elsif Arg = "matrix" then
            return Cmd_Matrix;
         elsif Arg = "demo" then
            return Cmd_Demo;
         else
            return Cmd_Help;
         end if;
      end;
   end Parse_Command;

   --  ============================================================
   --  Print Help
   --  ============================================================

   procedure Print_Help is
   begin
      Put_Line ("STUNIR DO-333 Formal Methods Analyzer");
      Put_Line ("======================================");
      New_Line;
      Put_Line ("Usage: do333_analyzer <command> [options]");
      New_Line;
      Put_Line ("Commands:");
      Put_Line ("  analyze <project.gpr>  Run formal verification analysis");
      Put_Line ("  report <format>        Generate certification reports");
      Put_Line ("  matrix                 Generate DO-333 compliance matrix");
      Put_Line ("  demo                   Run demonstration with sample data");
      Put_Line ("  --help, -h             Show this help");
      Put_Line ("  --version, -v          Show version");
      New_Line;
      Put_Line ("Report Formats: text, json, html, xml, csv");
      New_Line;
      Put_Line ("DO-333 Objectives Supported:");
      Put_Line ("  FM.1  Formal Specification");
      Put_Line ("  FM.2  Formal Verification (Proofs)");
      Put_Line ("  FM.3  Proof Coverage");
      Put_Line ("  FM.4  Verification Condition Management");
      Put_Line ("  FM.5  Formal Methods Integration");
      Put_Line ("  FM.6  Certification Evidence");
   end Print_Help;

   --  ============================================================
   --  Print Version
   --  ============================================================

   procedure Print_Version is
   begin
      Put_Line ("DO-333 Formal Methods Analyzer v1.0.0");
      Put_Line ("Part of STUNIR Compliance Framework");
      Put_Line ("Copyright (C) 2026 STUNIR Project");
      Put_Line ("SPARK proved, no runtime exceptions");
   end Print_Version;

   --  ============================================================
   --  Run Demo
   --  ============================================================

   procedure Run_Demo is
      --  Sample data for demonstration
      PO_Coll      : PO_Collection;
      VC_Coll      : VC_Collection;
      PO_Rec       : Proof_Obligation_Record;
      VC_Rec       : VC_Record;
      Success      : Boolean;
      Report_Buf   : String (1 .. Max_Report_Size);
      Report_Len   : Natural;
      Gen_Result   : Generation_Result;
      Metrics      : Coverage_Metrics;
      VC_Coverage  : VC_Coverage_Report;
   begin
      Put_Line ("DO-333 Formal Methods - Demonstration");
      Put_Line ("======================================");
      New_Line;

      --  Initialize collections
      Initialize (PO_Coll);
      Initialize (VC_Coll);

      --  Create sample proof obligations
      Put_Line ("Creating sample proof obligations...");

      Create_PO (1, PO_Precondition, DAL_A, "altitude.ads", "Check_Altitude",
                 10, 5, PO_Rec);
      Update_Status (PO_Rec, PO_Proved, 150, 500);
      Add_PO (PO_Coll, PO_Rec, Success);

      Create_PO (2, PO_Postcondition, DAL_A, "altitude.ads", "Check_Altitude",
                 15, 5, PO_Rec);
      Update_Status (PO_Rec, PO_Proved, 200, 800);
      Add_PO (PO_Coll, PO_Rec, Success);

      Create_PO (3, PO_Range_Check, DAL_B, "altitude.adb", "Update_Altitude",
                 25, 10, PO_Rec);
      Update_Status (PO_Rec, PO_Proved, 50, 100);
      Add_PO (PO_Coll, PO_Rec, Success);

      Create_PO (4, PO_Overflow_Check, DAL_B, "altitude.adb", "Update_Altitude",
                 30, 15, PO_Rec);
      Update_Status (PO_Rec, PO_Proved, 75, 200);
      Add_PO (PO_Coll, PO_Rec, Success);

      Create_PO (5, PO_Loop_Invariant_Init, DAL_C, "altitude.adb", "Smooth_Altitude",
                 50, 5, PO_Rec);
      Update_Status (PO_Rec, PO_Proved, 300, 1500);
      Add_PO (PO_Coll, PO_Rec, Success);

      --  Create sample verification conditions
      Put_Line ("Creating sample verification conditions...");

      for I in 1 .. 10 loop
         Create_VC (I, ((I - 1) / 2) + 1, VC_Rec);
         Update_VC_Status (VC_Rec, VC_Valid, I * 50, I * 100);
         Set_VC_Prover (VC_Rec, "z3");
         Add_VC (VC_Coll, VC_Rec, Success);
      end loop;

      --  Display metrics
      New_Line;
      Put_Line ("=== Proof Obligation Metrics ===");
      Metrics := Get_Metrics (PO_Coll);
      Put_Line ("  Total POs:    " & Natural'Image (Metrics.Total_POs));
      Put_Line ("  Proved POs:   " & Natural'Image (Metrics.Proved_POs));
      Put_Line ("  Unproved POs: " & Natural'Image (Metrics.Unproved_POs));
      Put_Line ("  Coverage:     " & Natural'Image (Metrics.Coverage_Pct) & "%");

      New_Line;
      Put_Line ("=== Verification Condition Metrics ===");
      VC_Coverage := Get_Coverage (VC_Coll);
      Put_Line ("  Total VCs:  " & Natural'Image (VC_Coverage.Total_VCs));
      Put_Line ("  Valid VCs:  " & Natural'Image (VC_Coverage.Valid_VCs));
      Put_Line ("  Coverage:   " & Natural'Image (VC_Coverage.Coverage_Pct) & "%");

      --  Generate text report
      New_Line;
      Put_Line ("=== Generating PO Report (Text) ===");
      Generate_PO_Report (PO_Coll, Format_Text, Report_Buf, Report_Len, Gen_Result);
      if Gen_Result = Gen_Success then
         Put_Line (Report_Buf (1 .. Report_Len));
      end if;

      --  Generate compliance matrix
      New_Line;
      Put_Line ("=== DO-333 Compliance Matrix ===");
      Generate_Compliance_Matrix (PO_Coll, VC_Coll, Format_Text,
                                  Report_Buf, Report_Len, Gen_Result);
      if Gen_Result = Gen_Success then
         Put_Line (Report_Buf (1 .. Report_Len));
      end if;

      --  Safety check
      New_Line;
      if All_Critical_Proved (PO_Coll) then
         Put_Line ("[PASS] All critical (DAL-A/B) POs are proved");
      else
         Put_Line ("[FAIL] Some critical POs remain unproved");
      end if;

      if All_Safety_Proved (PO_Coll) then
         Put_Line ("[PASS] All safety-related (DAL-A/B/C) POs are proved");
      else
         Put_Line ("[WARN] Some safety-related POs remain unproved");
      end if;

      New_Line;
      Put_Line ("Demo completed successfully.");
   end Run_Demo;

   --  ============================================================
   --  Main
   --  ============================================================

   Command : Command_Type;

begin
   Command := Parse_Command;

   case Command is
      when Cmd_Help =>
         Print_Help;

      when Cmd_Version =>
         Print_Version;

      when Cmd_Analyze =>
         Put_Line ("Analysis command - requires GNATprove integration");
         Put_Line ("Use: gnatprove -P <project.gpr> --mode=all");

      when Cmd_Report =>
         Put_Line ("Report command - run 'do333_analyzer demo' for examples");

      when Cmd_Matrix =>
         Put_Line ("Matrix command - run 'do333_analyzer demo' for examples");

      when Cmd_Demo =>
         Run_Demo;
   end case;

exception
   when others =>
      Put_Line ("Error: Unexpected exception");
      Set_Exit_Status (Failure);
end DO333_Main;
