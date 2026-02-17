--  ir_validate - Validate IR structure and syntax (Refactored)
--  Orchestrates ir_check_required, ir_check_functions, ir_check_types, validation_reporter
--  Phase 4 Powertool for STUNIR - Refactored for Unix philosophy

with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;
with GNAT.OS_Lib;
with Command_Utils;

procedure IR_Validate is
   use Ada.Command_Line;
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;

   --  Exit codes
   Exit_Success    : constant := 0;
   Exit_Invalid    : constant := 1;

   Show_Help     : Boolean := False;
   Show_Describe : Boolean := False;
   Output_Json   : Boolean := False;

   Version : constant String := "0.1.0-alpha";

   procedure Print_Usage is
   begin
      Put_Line (Standard_Error, "ir_validate - Validate IR structure");
      Put_Line (Standard_Error, "Version: " & Version);
      Put_Line (Standard_Error, "Usage: ir_validate [options] < ir.json");
      Put_Line (Standard_Error, "Options:");
      Put_Line (Standard_Error, "  --json       Output result as JSON");
      Put_Line (Standard_Error, "  --describe   Show tool description");
      Put_Line (Standard_Error, "  --help       Show this help");
   end Print_Usage;

   procedure Print_Describe is
   begin
      Put_Line ("{");
      Put_Line ("  ""tool"": ""ir_validate"",");
      Put_Line ("  ""description"": ""Validate IR structure and syntax"",");
      Put_Line ("  ""version"": ""0.1.0-alpha"",");
      Put_Line ("  ""inputs"": [{""name"": ""ir"", ""type"": ""json"", ""source"": [""stdin""]}],");
      Put_Line ("  ""outputs"": [{""name"": ""validation_report"", ""type"": ""json"", ""source"": ""stdout""}]");
      Put_Line ("}");
   end Print_Describe;

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

   function Run_Command (Cmd : String; Input : String) return String is
      Success : aliased Boolean;
      Result  : constant String :=
        Command_Utils.Get_Command_Output (Cmd, Input, Success'Access);
   begin
      return Result;
   end Run_Command;

begin
   for I in 1 .. Argument_Count loop
      declare
         Arg : constant String := Argument (I);
      begin
         if Arg = "--help" then
            Show_Help := True;
         elsif Arg = "--describe" then
            Show_Describe := True;
         elsif Arg = "--json" then
            Output_Json := True;
         end if;
      end;
   end loop;

   if Show_Help then
      Print_Usage;
      Set_Exit_Status (Exit_Success);
      return;
   end if;

   if Show_Describe then
      Print_Describe;
      Set_Exit_Status (Exit_Success);
      return;
   end if;

   declare
      Content : constant String := Read_Stdin;
   begin
      if Content'Length = 0 then
         Put_Line (Standard_Error, "ERROR: Empty input");
         Set_Exit_Status (Exit_Invalid);
         return;
      end if;

      --  Run validation checks
      declare
         Required_Result : constant String := Run_Command ("ir_check_required", Content);
         Functions_Result : constant String := Run_Command ("ir_check_functions", Content);
         Types_Result : constant String := Run_Command ("ir_check_types", Content);
         All_Results : constant String :=
           "[" & Required_Result & "," & Functions_Result & "," & Types_Result & "]";
         Format_Flag : constant String := (if Output_Json then " --format json" else "");
         Report : constant String := Run_Command ("validation_reporter" & Format_Flag, All_Results);
      begin
         Put_Line (Report);
         --  Check if any validation failed
         if Required_Result'Length > 0 and then
            Functions_Result'Length > 0 and then
            Types_Result'Length > 0 then
            Set_Exit_Status (Exit_Success);
         else
            Set_Exit_Status (Exit_Invalid);
         end if;
      end;
   end;
end IR_Validate;