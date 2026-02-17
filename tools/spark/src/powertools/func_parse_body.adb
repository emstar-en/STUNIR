--  func_parse_body - Parse function bodies from spec
--  Phase 1 Powertool for STUNIR

pragma SPARK_Mode (Off);

with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;
with Ada.Exceptions;

procedure Func_Parse_Body is
   use Ada.Command_Line;
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;

   --  Exit codes
   Exit_Success : constant := 0;
   Exit_Error   : constant := 1;

   --  Configuration
   Show_Version  : Boolean := False;
   Show_Help     : Boolean := False;
   Show_Describe : Boolean := False;

   Version : constant String := "0.1.0-alpha";

   Describe_Output : constant String :=
     "{" & ASCII.LF &
     "  ""tool"": ""func_parse_body""," & ASCII.LF &
     "  ""version"": ""0.1.0-alpha""," & ASCII.LF &
     "  ""description"": ""Parse function bodies from spec""," & ASCII.LF &
     "  ""inputs"": [{" & ASCII.LF &
     "    ""name"": ""body""," & ASCII.LF &
     "    ""type"": ""string""," & ASCII.LF &
     "    ""source"": [""stdin""]," & ASCII.LF &
     "    ""required"": true" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  ""outputs"": [{" & ASCII.LF &
     "    ""name"": ""parsed_body""," & ASCII.LF &
     "    ""type"": ""object""," & ASCII.LF &
     "    ""source"": ""stdout""" & ASCII.LF &
     "  }]" & ASCII.LF &
     "}";

   procedure Print_Usage is
   begin
      Put_Line ("func_parse_body - Parse function bodies");
      Put_Line ("Version: " & Version);
      Put_Line ("");
      Put_Line ("Usage: func_parse_body [OPTIONS] < body.txt");
      Put_Line ("");
      Put_Line ("Options:");
      Put_Line ("  --help, -h        Show this help message");
      Put_Line ("  --version, -v     Show version information");
      Put_Line ("  --describe        Show tool description (JSON)");
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
      end loop;
      return To_String (Result);
   end Read_Stdin;

   function Parse_Body (Body_Text : String) return String is
   begin
      return "{ ""body"": """ & Body_Text & """ }";
   end Parse_Body;

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
         end if;
      end;
   end loop;

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
      Content : constant String := Read_Stdin;
   begin
      if Content'Length = 0 then
         Print_Error ("Empty input");
         Set_Exit_Status (Exit_Error);
         return;
      end if;

      Put_Line (Parse_Body (Content));
      Set_Exit_Status (Exit_Success);
   end;

exception
   when others =>
      Print_Error ("Unexpected error");
      Set_Exit_Status (Exit_Error);
end Func_Parse_Body;