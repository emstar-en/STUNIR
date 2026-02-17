--  spec_extract_module - Extract module information from STUNIR spec
--  Uses json_extract and json_formatter utilities
--  Phase 2 Powertool for STUNIR

pragma SPARK_Mode (Off);

with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;
with GNAT.Command_Line;
with GNAT.OS_Lib;
with GNAT.Strings;

procedure Spec_Extract_Module is
   use Ada.Command_Line;
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;

   --  Exit codes
   Exit_Success    : constant := 0;
   Exit_Extraction_Error : constant := 1;

   --  Configuration
   Show_Version  : aliased Boolean := False;
   Show_Help     : aliased Boolean := False;
   Show_Describe : aliased Boolean := False;
   Output_Field  : Unbounded_String := Null_Unbounded_String;

   Version : constant String := "0.1.0-alpha";

   Describe_Output : constant String :=
     "{" & ASCII.LF &
     "  ""tool"": ""spec_extract_module""," & ASCII.LF &
     "  ""version"": ""0.1.0-alpha""," & ASCII.LF &
     "  ""description"": ""Extract module information from STUNIR spec""," & ASCII.LF &
     "  ""inputs"": [{" & ASCII.LF &
     "    ""name"": ""spec_json""," & ASCII.LF &
     "    ""type"": ""json""," & ASCII.LF &
     "    ""source"": [""stdin""]," & ASCII.LF &
     "    ""required"": true" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  ""outputs"": [{" & ASCII.LF &
     "    ""name"": ""module_info""," & ASCII.LF &
     "    ""type"": ""json""," & ASCII.LF &
     "    ""source"": ""stdout""" & ASCII.LF &
     "  }]" & ASCII.LF &
     "}";

   procedure Print_Usage is
   begin
      Put_Line ("spec_extract_module - Extract module information from spec");
      Put_Line ("Version: " & Version);
      Put_Line ("");
      Put_Line ("Usage: spec_extract_module [OPTIONS] < spec.json");
      Put_Line ("");
      Put_Line ("Options:");
      Put_Line ("  --help, -h        Show this help message");
      Put_Line ("  --version, -v     Show version information");
      Put_Line ("  --describe        Show tool description (JSON)");
      Put_Line ("  --field NAME      Extract specific field (name, version, functions)");
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

   function Run_Command (Cmd : String; Input : String) return String is
      use GNAT.OS_Lib;
      Args    : Argument_List_Access;
      Success : aliased Boolean;
   begin
      Args := Argument_String_To_List (Cmd);
      declare
         Result : constant String :=
           Get_Command_Output ("sh", Args.all, Input, Success'Access);
      begin
         Free (Args);
         if Success then
            return Result;
         else
            return "";
         end if;
      end;
   exception
      when others =>
         if Args /= null then
            Free (Args);
         end if;
         return "";
   end Run_Command;

   Config : GNAT.Command_Line.Command_Line_Configuration;

begin
   GNAT.Command_Line.Define_Switch (Config, Show_Help'Access, "-h", "--help");
   GNAT.Command_Line.Define_Switch (Config, Show_Version'Access, "-v", "--version");
   GNAT.Command_Line.Define_Switch (Config, Show_Describe'Access, "", "--describe");
   GNAT.Command_Line.Define_Switch (Config, Output_Field'Access, "", "--field=");

   begin
      GNAT.Command_Line.Getopt (Config);
   exception
      when others =>
         Print_Error ("Invalid arguments");
         Set_Exit_Status (Exit_Extraction_Error);
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
         Set_Exit_Status (Exit_Extraction_Error);
         return;
      end if;

      --  Extract module information using json_extract
      declare
         Field_Path : constant String :=
           (if Output_Field = Null_Unbounded_String then
               "module"
            else
               "module." & To_String (Output_Field));
         Extracted : constant String :=
           Run_Command ("json_extract --path " & Field_Path, Spec_JSON);
      begin
         if Extracted'Length = 0 then
            Print_Error ("Module field not found: " & Field_Path);
            Set_Exit_Status (Exit_Extraction_Error);
            return;
         end if;

         --  Format output using json_formatter
         declare
            Formatted : constant String :=
              Run_Command ("json_formatter", Extracted);
         begin
            if Formatted'Length > 0 then
               Put_Line (Formatted);
            else
               Put_Line (Extracted);
            end if;
         end;
      end;
   end;

   Set_Exit_Status (Exit_Success);

exception
   when others =>
      Print_Error ("Unexpected error");
      Set_Exit_Status (Exit_Extraction_Error);
end Spec_Extract_Module;