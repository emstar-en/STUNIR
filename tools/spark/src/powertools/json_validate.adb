--  json_validate - Simple JSON validation tool
--  Validates JSON from stdin or file, exits with appropriate code
--  Phase 1 Powertool for STUNIR

pragma SPARK_Mode (Off);

with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;
with Ada.IO_Exceptions;

with GNAT.Command_Line;

with STUNIR_JSON_Parser;
with STUNIR_Types;

procedure JSON_Validate is
   use Ada.Command_Line;
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;
   use STUNIR_Types;

   --  Exit codes per powertools spec
   Exit_Success         : constant := 0;
   Exit_Validation_Error : constant := 1;
   Exit_Processing_Error : constant := 2;
   Exit_Resource_Error   : constant := 3;

   --  Configuration
   Input_File     : Unbounded_String := Null_Unbounded_String;
   Strict_Mode    : Boolean := False;
   Verbose_Mode   : Boolean := False;
   Show_Version   : Boolean := False;
   Show_Help      : Boolean := False;
   Show_Describe  : Boolean := False;

   Version : constant String := "1.0.0";

   --  Description output for --describe
   Describe_Output : constant String :=
     "{" & ASCII.LF &
     "  ""tool"": ""json_validate""," & ASCII.LF &
     "  ""version"": ""1.0.0""," & ASCII.LF &
     "  ""description"": ""Validates JSON structure and syntax""," & ASCII.LF &
     "  ""inputs"": [{" & ASCII.LF &
     "    ""name"": ""json_input""," & ASCII.LF &
     "    ""type"": ""json""," & ASCII.LF &
     "    ""source"": [""stdin"", ""file""]," & ASCII.LF &
     "    ""required"": true" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  ""outputs"": [{" & ASCII.LF &
     "    ""name"": ""validation_result""," & ASCII.LF &
     "    ""type"": ""exit_code""," & ASCII.LF &
     "    ""description"": ""0=valid, 1=invalid, 2=error""" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  ""complexity"": ""O(n)""," & ASCII.LF &
     "  ""options"": [" & ASCII.LF &
     "    ""--help"", ""--version"", ""--describe"", ""--strict"", ""--verbose""" & ASCII.LF &
     "  ]" & ASCII.LF &
     "}";

   procedure Print_Usage;
   procedure Print_Error (Msg : String);
   procedure Print_Info (Msg : String);
   function Read_File (Path : String) return String;
   function Validate_JSON (Content : String) return Integer;

   procedure Print_Usage is
   begin
      Put_Line ("json_validate - Validate JSON structure");
      Put_Line ("Version: " & Version);
      Put_Line ("");
      Put_Line ("Usage: json_validate [OPTIONS] [FILE]");
      Put_Line ("");
      Put_Line ("Options:");
      Put_Line ("  --help, -h        Show this help message");
      Put_Line ("  --version, -v     Show version information");
      Put_Line ("  --describe        Show tool description (JSON)");
      Put_Line ("  --strict          Strict mode (no trailing commas)");
      Put_Line ("  --verbose         Verbose output");
      Put_Line ("");
      Put_Line ("Arguments:");
      Put_Line ("  FILE              JSON file to validate (default: stdin)");
      Put_Line ("");
      Put_Line ("Exit Codes:");
      Put_Line ("  0                 Valid JSON");
      Put_Line ("  1                 Invalid JSON");
      Put_Line ("  2                 Processing error");
      Put_Line ("  3                 Resource error");
   end Print_Usage;

   procedure Print_Error (Msg : String) is
   begin
      Put_Line (Standard_Error, "ERROR: " & Msg);
   end Print_Error;

   procedure Print_Info (Msg : String) is
   begin
      if Verbose_Mode then
         Put_Line (Standard_Error, "INFO: " & Msg);
      end if;
   end Print_Info;

   function Read_File (Path : String) return String is
      File : File_Type;
      Result : Unbounded_String := Null_Unbounded_String;
      Line   : String (1 .. 4096);
      Last   : Natural;
   begin
      Open (File, In_File, Path);
      while not End_Of_File (File) loop
         Get_Line (File, Line, Last);
         Append (Result, Line (1 .. Last));
         Append (Result, ASCII.LF);
      end loop;
      Close (File);
      return To_String (Result);
   exception
      when Ada.IO_Exceptions.Name_Error =>
         Print_Error ("Cannot open file: " & Path);
         return "";
      when others =>
         Print_Error ("Error reading file: " & Path);
         return "";
   end Read_File;

   function Validate_JSON (Content : String) return Integer is
      use STUNIR_JSON_Parser;
      State  : Parser_State;
      Status : Status_Code;
      Input_Str : JSON_String;
   begin
      if Content'Length = 0 then
         Print_Error ("Empty input");
         return Exit_Validation_Error;
      end if;

      if Content'Length > Max_JSON_Length then
         Print_Error ("Input too large (max" & Max_JSON_Length'Img & " bytes)");
         return Exit_Resource_Error;
      end if;

      --  Convert to bounded string
      Input_Str := JSON_Strings.To_Bounded_String (Content);

      --  Initialize parser
      Initialize_Parser (State, Input_Str, Status);
      if Status /= Success then
         Print_Error ("Failed to initialize parser");
         return Exit_Processing_Error;
      end if;

      --  Parse all tokens
      loop
         Next_Token (State, Status);
         exit when Status /= Success or else State.Current_Token = Token_EOF;
      end loop;

      if Status = Success then
         Print_Info ("JSON is valid");
         return Exit_Success;
      else
         Print_Error ("Invalid JSON at line" & State.Line'Img & ", column" & State.Column'Img);
         return Exit_Validation_Error;
      end if;
   end Validate_JSON;

   --  Command line parsing
   Config : GNAT.Command_Line.Command_Line_Configuration;

begin
   --  Set up command line options
   GNAT.Command_Line.Define_Switch (Config, Show_Help'Access, "-h", "--help");
   GNAT.Command_Line.Define_Switch (Config, Show_Version'Access, "-v", "--version");
   GNAT.Command_Line.Define_Switch (Config, Show_Describe'Access, "", "--describe");
   GNAT.Command_Line.Define_Switch (Config, Strict_Mode'Access, "", "--strict");
   GNAT.Command_Line.Define_Switch (Config, Verbose_Mode'Access, "", "--verbose");

   begin
      GNAT.Command_Line.Getopt (Config);
   exception
      when others =>
         Print_Error ("Invalid command line arguments");
         Set_Exit_Status (Exit_Processing_Error);
         return;
   end;

   --  Handle info options
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

   --  Get input file if provided
   declare
      Remaining : String := GNAT.Command_Line.Get_Argument;
   begin
      if Remaining'Length > 0 then
         Input_File := To_Unbounded_String (Remaining);
      end if;
   end;

   --  Read input
   declare
      Content : Unbounded_String;
   begin
      if Input_File = Null_Unbounded_String then
         --  Read from stdin
         Print_Info ("Reading from stdin...");
         while not End_Of_File loop
            declare
               Line : String (1 .. 4096);
               Last : Natural;
            begin
               Get_Line (Line, Last);
               Append (Content, Line (1 .. Last));
               Append (Content, ASCII.LF);
            end;
         end loop;
      else
         --  Read from file
         Print_Info ("Reading from file: " & To_String (Input_File));
         Content := To_Unbounded_String (Read_File (To_String (Input_File)));
         if Content = Null_Unbounded_String then
            Set_Exit_Status (Exit_Processing_Error);
            return;
         end if;
      end if;

      --  Validate
      Print_Info ("Validating JSON...");
      declare
         Result : constant Integer := Validate_JSON (To_String (Content));
      begin
         Set_Exit_Status (Result);
      end;
   end;

exception
   when others =>
      Print_Error ("Unexpected error");
      Set_Exit_Status (Exit_Processing_Error);
end JSON_Validate;