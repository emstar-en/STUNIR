--  spec_validate_schema - Validate spec JSON against schema
--  Orchestrates schema_check_required, schema_check_types, schema_check_format, validation_reporter
--  Phase 4 Powertool for STUNIR

pragma SPARK_Mode (Off);

with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;
with GNAT.Command_Line;
with GNAT.OS_Lib;
with GNAT.Strings;
with Command_Utils;

procedure Spec_Validate_Schema is
   use Ada.Command_Line;
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;
   use GNAT.Strings;

   --  Exit codes
   Exit_Success    : constant := 0;
   Exit_Invalid    : constant := 1;

   --  Configuration
   Schema_File   : aliased GNAT.Strings.String_Access := null;
   Report_Format : aliased GNAT.Strings.String_Access := new String'("text");
   Verbose       : aliased Boolean := False;
   Show_Version  : aliased Boolean := False;
   Show_Help     : aliased Boolean := False;
   Show_Describe : aliased Boolean := False;

   Version : constant String := "0.1.0-alpha";

   Describe_Output : constant String :=
     "{" & ASCII.LF &
     "  ""tool"": ""spec_validate_schema""," & ASCII.LF &
     "  ""version"": ""0.1.0-alpha""," & ASCII.LF &
     "  ""description"": ""Validate spec JSON against schema""," & ASCII.LF &
     "  ""inputs"": [{" & ASCII.LF &
     "    ""name"": ""spec_json""," & ASCII.LF &
     "    ""type"": ""json""," & ASCII.LF &
     "    ""source"": [""stdin""]," & ASCII.LF &
     "    ""required"": true" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  ""outputs"": [{" & ASCII.LF &
     "    ""name"": ""validation_report""," & ASCII.LF &
     "    ""type"": ""json""," & ASCII.LF &
     "    ""source"": ""stdout""" & ASCII.LF &
     "  }]" & ASCII.LF &
     "}";

   procedure Print_Usage is
   begin
      Put_Line ("spec_validate_schema - Validate spec JSON against schema");
      Put_Line ("Version: " & Version);
      Put_Line ("");
      Put_Line ("Usage: spec_validate_schema [OPTIONS] < spec.json");
      Put_Line ("");
      Put_Line ("Options:");
      Put_Line ("  --help, -h        Show this help message");
      Put_Line ("  --version, -v     Show version information");
      Put_Line ("  --describe        Show tool description (JSON)");
      Put_Line ("  --schema FILE     Schema definition file");
      Put_Line ("  --format FORMAT   Report format: text|json (default: text)");
      Put_Line ("  --verbose         Show all checks, not just errors");
   end Print_Usage;

   procedure Print_Error (Msg : String) is
   begin
      Put_Line (Standard_Error, "ERROR: " & Msg);
   end Print_Error;

   function Read_Stdin return String is
      Result : Unbounded_String := Null_Unbounded_String;
      Line   : String (1 .. 4096);
      Last   : Natural;
   begin
      while not End_Of_File loop
         Get_Line (Line, Last);
         Append (Result, Line (1 .. Last));
         Append (Result, ASCII.LF);
      end loop;
      return To_String (Result);
   end Read_Stdin;

   function Run_Command (Cmd : String; Input : String := "") return String is
      Success : aliased Boolean;
      Result  : constant String :=
        Command_Utils.Get_Command_Output (Cmd, Input, Success'Access);
   begin
      return Result;
   end Run_Command;

   Config : GNAT.Command_Line.Command_Line_Configuration;

begin
   GNAT.Command_Line.Define_Switch (Config, Show_Help'Access, "-h", "--help");
   GNAT.Command_Line.Define_Switch (Config, Show_Version'Access, "-v", "--version");
   GNAT.Command_Line.Define_Switch (Config, Show_Describe'Access, "", "--describe");
   GNAT.Command_Line.Define_Switch (Config, Verbose'Access, "", "--verbose");
   GNAT.Command_Line.Define_Switch (Config, Schema_File'Access, "", "--schema=");
   GNAT.Command_Line.Define_Switch (Config, Report_Format'Access, "", "--format=");

   begin
      GNAT.Command_Line.Getopt (Config);
   exception
      when others =>
         Print_Error ("Invalid arguments");
         Set_Exit_Status (Exit_Invalid);
         return;
   end;

   if Show_Help then
      Print_Usage;
      Set_Exit_Status (Exit_Success);
      return;
   end if;

   if Show_Version then
      Put_Line (Version);
      Set_Exit_Status (Exit_Success);
      return;
   end if;

   if Show_Describe then
      Put_Line (Describe_Output);
      Set_Exit_Status (Exit_Success);
      return;
   end if;

   declare
      Spec_JSON : constant String := Read_Stdin;
   begin
      if Spec_JSON'Length = 0 then
         Print_Error ("Empty input");
         Set_Exit_Status (Exit_Invalid);
         return;
      end if;

      --  Run validation checks
      declare
         Schema_Flag : constant String :=
           (if Schema_File = null then "" else " --schema " & Schema_File.all);
         Required_Result : constant String :=
           Run_Command ("schema_check_required" & Schema_Flag, Spec_JSON);
         Types_Result : constant String :=
           Run_Command ("schema_check_types" & Schema_Flag, Spec_JSON);
         Format_Result : constant String :=
           Run_Command ("schema_check_format" & Schema_Flag, Spec_JSON);
         All_Results : constant String :=
           "[" & Required_Result & "," & Types_Result & "," & Format_Result & "]";
         Format_Flag : constant String := " --format " & Report_Format.all;
         Verbose_Flag : constant String := (if Verbose then " --verbose" else "");
         Report : constant String :=
           Run_Command ("validation_reporter" & Format_Flag & Verbose_Flag, All_Results);
      begin
         Put_Line (Report);
         --  Check if validation passed (all results non-empty and no "INVALID" in report)
         if Required_Result'Length > 0 and then
            Types_Result'Length > 0 and then
            Format_Result'Length > 0 and then
            (Report'Length = 0 or else
             (for all I in Report'Range =>
                Report (I .. I + 6) /= "INVALID" or else I + 6 > Report'Last)) then
            Set_Exit_Status (Exit_Success);
         else
            Set_Exit_Status (Exit_Invalid);
         end if;
      end;
   end;

exception
   when others =>
      Print_Error ("Unexpected error");
      Set_Exit_Status (Exit_Invalid);
end Spec_Validate_Schema;