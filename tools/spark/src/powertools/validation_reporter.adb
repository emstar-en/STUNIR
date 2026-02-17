--  validation_reporter - Format validation reports
--  Validation reporting utility for STUNIR powertools
--  Phase 3 Utility for STUNIR

pragma SPARK_Mode (Off);

with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;

procedure Validation_Reporter is
   use Ada.Command_Line;
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;

   Exit_Success : constant := 0;

   Show_Version  : Boolean := False;
   Show_Help     : Boolean := False;
   Show_Describe : Boolean := False;
   JSON_Output   : Boolean := False;

   Version : constant String := "0.1.0-alpha";

   Describe_Output : constant String :=
     "{" & ASCII.LF &
     "  ""tool"": ""validation_reporter""," & ASCII.LF &
     "  ""version"": ""0.1.0-alpha""," & ASCII.LF &
     "  ""description"": ""Format validation reports""," & ASCII.LF &
     "  ""inputs"": [{" & ASCII.LF &
     "    ""name"": ""validation_results""," & ASCII.LF &
     "    ""type"": ""json""," & ASCII.LF &
     "    ""source"": ""stdin""," & ASCII.LF &
     "    ""required"": true" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  ""outputs"": [{" & ASCII.LF &
     "    ""name"": ""formatted_report""," & ASCII.LF &
     "    ""type"": ""text""," & ASCII.LF &
     "    ""source"": ""stdout""" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  ""complexity"": ""O(n)""," & ASCII.LF &
     "  ""options"": [" & ASCII.LF &
     "    ""--help"", ""--version"", ""--describe""," & ASCII.LF &
     "    ""--json""" & ASCII.LF &
     "  ]" & ASCII.LF &
     "}";

   procedure Print_Usage;
   function Read_Stdin return String;
   procedure Format_Report (Input : String);

   procedure Print_Usage is
   begin
      Put_Line ("validation_reporter - Format validation reports");
      Put_Line ("Version: " & Version);
      Put_Line ("");
      Put_Line ("Usage: validation_reporter [OPTIONS]");
      Put_Line ("");
      Put_Line ("Options:");
      Put_Line ("  --help, -h        Show this help message");
      Put_Line ("  --version, -v     Show version information");
      Put_Line ("  --describe        Show tool description (JSON)");
      Put_Line ("  --json            Output in JSON format");
   end Print_Usage;

   function Read_Stdin return String is
      Result : Unbounded_String := Null_Unbounded_String;
      Line   : String (1 .. 4096);
      Last   : Natural;
   begin
      while not End_Of_File (Standard_Input) loop
         Get_Line (Standard_Input, Line, Last);
         Append (Result, Line (1 .. Last));
         Append (Result, ASCII.LF);
      end loop;
      return To_String (Result);
   end Read_Stdin;

   procedure Format_Report (Input : String) is
   begin
      if JSON_Output then
         Put_Line ("{");
         Put_Line ("  \"status\": \"validation_report\",");
         Put_Line ("  \"results\": [");
         Put_Line ("    \"Validation completed\"");
         Put_Line ("  ]");
         Put_Line ("}");
      else
         Put_Line ("=== Validation Report ===");
         Put_Line (Input);
         Put_Line ("========================");
      end if;
   end Format_Report;

begin
   for I in 1 .. Argument_Count loop
      declare
         Arg : constant String := Argument (I);
      begin
         if Arg = "--help" or Arg = "-h" then
            Show_Help := True;
         elsif Arg = "--version" or Arg = "-v" then
            Show_Version := True;
         elsif Arg = "--describe" then
            Show_Describe := True;
         elsif Arg = "--json" then
            JSON_Output := True;
         end if;
      end;
   end loop;

   if Show_Help then
      Print_Usage;
      return;
   end if;

   if Show_Version then
      Put_Line ("validation_reporter " & Version);
      return;
   end if;

   if Show_Describe then
      Put_Line (Describe_Output);
      return;
   end if;

   declare
      Input : constant String := Read_Stdin;
   begin
      Format_Report (Input);
      Set_Exit_Status (Exit_Success);
   end;

end Validation_Reporter;
