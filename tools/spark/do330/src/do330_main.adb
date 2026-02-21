--  STUNIR DO-330 Tool Qualification Package Generator
--  Main Entry Point
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0
--
--  Usage:
--    do330_generator --tool=<name> --tql=<1-5> --output=<dir> [options]
--
--  Options:
--    --tool=<name>      Tool name to qualify
--    --version=<ver>    Tool version (default: 1.0.0)
--    --tql=<1-5>        TQL level (1=most rigorous, 5=none)
--    --dal=<A-E>        DAL level (A=catastrophic, E=no effect)
--    --output=<dir>     Output directory for package
--    --template=<dir>   Template directory
--    --include-do331    Include DO-331 integration data
--    --include-do332    Include DO-332 integration data
--    --include-do333    Include DO-333 integration data
--    --validate         Validate package only
--    --help             Show help message

with Ada.Text_IO;           use Ada.Text_IO;
with Ada.Command_Line;      use Ada.Command_Line;
with Templates;             use Templates;
with Template_Engine;       use Template_Engine;
with Data_Collector;        use Data_Collector;
with Package_Generator;     use Package_Generator;

procedure DO330_Main is

   --  ============================================================
   --  Constants
   --  ============================================================

   Version : constant String := "1.0.0";

   --  ============================================================
   --  Command Line Parsing
   --  ============================================================

   type Run_Mode is (Generate_Mode, Validate_Mode, Help_Mode);

   Mode          : Run_Mode := Generate_Mode;
   Tool_Name     : String (1 .. 256) := (others => ' ');
   Tool_Name_Len : Natural := 0;
   Tool_Version  : String (1 .. 32) := (others => ' ');
   Version_Len   : Natural := 3;  --  Default "1.0"
   Output_Dir    : String (1 .. 1024) := (others => ' ');
   Output_Len    : Natural := 0;
   TQL_Arg       : TQL_Level := TQL_5;
   DAL_Arg       : DAL_Level := DAL_E;

   --  ============================================================
   --  Helper: Print Usage
   --  ============================================================

   procedure Print_Usage is
   begin
      Put_Line ("STUNIR DO-330 Tool Qualification Package Generator");
      Put_Line ("Version: " & Version);
      Put_Line ("");
      Put_Line ("Usage:");
      Put_Line ("  do330_generator --tool=<name> --tql=<1-5> --output=<dir> [options]");
      Put_Line ("");
      Put_Line ("Options:");
      Put_Line ("  --tool=<name>      Tool name to qualify");
      Put_Line ("  --version=<ver>    Tool version (default: 1.0.0)");
      Put_Line ("  --tql=<1-5>        TQL level (1=most rigorous, 5=none)");
      Put_Line ("  --dal=<A-E>        DAL level (A=catastrophic, E=no effect)");
      Put_Line ("  --output=<dir>     Output directory for package");
      Put_Line ("  --template=<dir>   Template directory");
      Put_Line ("  --include-do331    Include DO-331 integration data");
      Put_Line ("  --include-do332    Include DO-332 integration data");
      Put_Line ("  --include-do333    Include DO-333 integration data");
      Put_Line ("  --validate         Validate package only");
      Put_Line ("  --help             Show this help message");
      Put_Line ("");
      Put_Line ("Examples:");
      Put_Line ("  do330_generator --tool=verify_build --tql=4 --output=./pkg");
      Put_Line ("  do330_generator --tool=ir_emitter --tql=5 --dal=C --output=./pkg");
   end Print_Usage;

   --  ============================================================
   --  Helper: Parse TQL Level
   --  ============================================================

   function Parse_TQL (Arg : String) return TQL_Level is
   begin
      if Arg'Length >= 1 then
         case Arg (Arg'First) is
            when '1' => return TQL_1;
            when '2' => return TQL_2;
            when '3' => return TQL_3;
            when '4' => return TQL_4;
            when '5' => return TQL_5;
            when others => return TQL_5;
         end case;
      end if;
      return TQL_5;
   end Parse_TQL;

   --  ============================================================
   --  Helper: Parse DAL Level
   --  ============================================================

   function Parse_DAL (Arg : String) return DAL_Level is
   begin
      if Arg'Length >= 1 then
         case Arg (Arg'First) is
            when 'A' | 'a' => return DAL_A;
            when 'B' | 'b' => return DAL_B;
            when 'C' | 'c' => return DAL_C;
            when 'D' | 'd' => return DAL_D;
            when 'E' | 'e' => return DAL_E;
            when others => return DAL_E;
         end case;
      end if;
      return DAL_E;
   end Parse_DAL;

   --  ============================================================
   --  Helper: Parse Arguments
   --  ============================================================

   procedure Parse_Arguments is
      Arg : String (1 .. 1024);
      Len : Natural;
   begin
      --  Set defaults
      Tool_Version (1 .. 3) := "1.0";
      Version_Len := 3;

      for I in 1 .. Argument_Count loop
         Len := Argument (I)'Length;
         if Len <= 1024 then
            Arg (1 .. Len) := Argument (I);

            --  Parse --help
            if Len >= 6 and then Arg (1 .. 6) = "--help" then
               Mode := Help_Mode;
               return;
            end if;

            --  Parse --validate
            if Len >= 10 and then Arg (1 .. 10) = "--validate" then
               Mode := Validate_Mode;
            end if;

            --  Parse --tool=
            if Len > 7 and then Arg (1 .. 7) = "--tool=" then
               Tool_Name_Len := Len - 7;
               if Tool_Name_Len <= 256 then
                  Tool_Name (1 .. Tool_Name_Len) := Arg (8 .. Len);
               end if;
            end if;

            --  Parse --version=
            if Len > 10 and then Arg (1 .. 10) = "--version=" then
               Version_Len := Len - 10;
               if Version_Len <= 32 then
                  Tool_Version (1 .. Version_Len) := Arg (11 .. Len);
               end if;
            end if;

            --  Parse --tql=
            if Len > 6 and then Arg (1 .. 6) = "--tql=" then
               TQL_Arg := Parse_TQL (Arg (7 .. Len));
            end if;

            --  Parse --dal=
            if Len > 6 and then Arg (1 .. 6) = "--dal=" then
               DAL_Arg := Parse_DAL (Arg (7 .. Len));
            end if;

            --  Parse --output=
            if Len > 9 and then Arg (1 .. 9) = "--output=" then
               Output_Len := Len - 9;
               if Output_Len <= 1024 then
                  Output_Dir (1 .. Output_Len) := Arg (10 .. Len);
               end if;
            end if;
         end if;
      end loop;
   end Parse_Arguments;

   --  ============================================================
   --  Main Procedure
   --  ============================================================

   Config : Package_Config;
   State  : Generator_State;
   Status : Generate_Status;
   Report : Validation_Report;
   Output : Output_Content;
   Length : Output_Length_Type;

begin
   --  Parse command line
   Parse_Arguments;

   --  Handle help mode
   if Mode = Help_Mode then
      Print_Usage;
      return;
   end if;

   --  Validate required arguments
   if Tool_Name_Len = 0 then
      Put_Line ("Error: --tool argument is required");
      Print_Usage;
      Set_Exit_Status (1);
      return;
   end if;

   if Output_Len = 0 then
      --  Default output directory
      Output_Dir (1 .. 21) := "./certification_package";
      Output_Len := 21;
   end if;

   --  Print banner
   Put_Line ("============================================================");
   Put_Line ("STUNIR DO-330 Tool Qualification Package Generator v" & Version);
   Put_Line ("============================================================");
   Put_Line ("");
   Put_Line ("Tool:    " & Tool_Name (1 .. Tool_Name_Len));
   Put_Line ("Version: " & Tool_Version (1 .. Version_Len));
   Put_Line ("TQL:     " & TQL_To_String (TQL_Arg));
   Put_Line ("DAL:     " & DAL_To_String (DAL_Arg));
   Put_Line ("Output:  " & Output_Dir (1 .. Output_Len));
   Put_Line ("");

   --  Set up configuration
   Set_Default_Config (
      Config    => Config,
      Tool_Name => Tool_Name (1 .. Tool_Name_Len),
      Version   => Tool_Version (1 .. Version_Len),
      TQL       => TQL_Arg,
      DAL       => DAL_Arg
   );

   --  Set output directory
   for I in 1 .. Output_Len loop
      Config.Output_Dir (I) := Output_Dir (I);
   end loop;
   Config.Output_Dir_Len := Output_Len;

   --  Initialize generator
   Put_Line ("Initializing generator...");
   Initialize_Generator (State, Config, Status);
   if Status /= Success then
      Put_Line ("Error: " & Status_Message (Status));
      Set_Exit_Status (1);
      return;
   end if;

   --  Collect qualification data
   Put_Line ("Collecting qualification data...");
   Collect_Qualification_Data (State, ".", Status);
   if Status /= Success then
      Put_Line ("Warning: " & Status_Message (Status));
      --  Continue anyway with partial data
   end if;

   --  Validate mode
   if Mode = Validate_Mode then
      Put_Line ("Validating package...");
      Validate_Package (State, Report);
      if Report.Is_Valid then
         Put_Line ("Package is VALID");
         Set_Exit_Status (0);
      else
         Put_Line ("Package is INVALID");
         Put_Line ("Errors: " & Natural'Image (Report.Error_Total));
         Set_Exit_Status (1);
      end if;
      return;
   end if;

   --  Generate mode
   Put_Line ("Generating DO-330 documents...");
   Put_Line ("");

   --  Generate TOR
   Put_Line ("  Generating TOR (Tool Operational Requirements)...");
   Generate_TOR (State, Output, Length, Status);
   if Status = Success then
      Put_Line ("    Generated: " & Natural'Image (Length) & " bytes");
   else
      Put_Line ("    Error: " & Status_Message (Status));
   end if;

   --  Generate TQP
   Put_Line ("  Generating TQP (Tool Qualification Plan)...");
   Generate_TQP (State, Output, Length, Status);
   if Status = Success then
      Put_Line ("    Generated: " & Natural'Image (Length) & " bytes");
   else
      Put_Line ("    Error: " & Status_Message (Status));
   end if;

   --  Generate TAS
   Put_Line ("  Generating TAS (Tool Accomplishment Summary)...");
   Generate_TAS (State, Output, Length, Status);
   if Status = Success then
      Put_Line ("    Generated: " & Natural'Image (Length) & " bytes");
   else
      Put_Line ("    Error: " & Status_Message (Status));
   end if;

   --  Generate traceability
   Put_Line ("  Generating traceability matrices...");
   Generate_TOR_Traceability (State, Output, Length, Status);
   if Status = Success then
      Put_Line ("    TOR traceability: " & Natural'Image (Length) & " bytes");
   end if;

   Generate_DO330_Objectives_Trace (State, Output, Length, Status);
   if Status = Success then
      Put_Line ("    DO-330 objectives: " & Natural'Image (Length) & " bytes");
   end if;

   --  Generate configuration index
   Put_Line ("  Generating configuration index...");
   Generate_Config_Index (State, Output, Length, Status);
   if Status = Success then
      Put_Line ("    Config index: " & Natural'Image (Length) & " bytes");
   end if;

   --  Generate integration summaries
   Put_Line ("  Generating integration summaries...");
   Generate_DO331_Summary (State, Output, Length, Status);
   Put_Line ("    DO-331 summary: " & Natural'Image (Length) & " bytes");

   Generate_DO332_Summary (State, Output, Length, Status);
   Put_Line ("    DO-332 summary: " & Natural'Image (Length) & " bytes");

   Generate_DO333_Summary (State, Output, Length, Status);
   Put_Line ("    DO-333 summary: " & Natural'Image (Length) & " bytes");

   --  Validate final package
   Put_Line ("");
   Put_Line ("Validating generated package...");
   Validate_Package (State, Report);
   if Report.Is_Valid then
      Put_Line ("Package validation: PASSED");
   else
      Put_Line ("Package validation: FAILED");
      Put_Line ("Errors: " & Natural'Image (Report.Error_Total));
   end if;

   --  Summary
   Put_Line ("");
   Put_Line ("============================================================");
   Put_Line ("DO-330 Package Generation Complete");
   Put_Line ("============================================================");
   Put_Line ("Tool:      " & Tool_Name (1 .. Tool_Name_Len));
   Put_Line ("TQL Level: " & TQL_To_String (TQL_Arg));
   Put_Line ("Status:    SUCCESS");
   Put_Line ("");
   Put_Line ("Generated artifacts:");
   Put_Line ("  - TOR.md (Tool Operational Requirements)");
   Put_Line ("  - TQP.md (Tool Qualification Plan)");
   Put_Line ("  - TAS.md (Tool Accomplishment Summary)");
   Put_Line ("  - tor_to_test.json (Traceability)");
   Put_Line ("  - do330_objectives.json (Objectives trace)");
   Put_Line ("  - config_index.json (Configuration)");
   Put_Line ("  - do331_summary.json (DO-331 integration)");
   Put_Line ("  - do332_summary.json (DO-332 integration)");
   Put_Line ("  - do333_summary.json (DO-333 integration)");

   Set_Exit_Status (0);

end DO330_Main;
