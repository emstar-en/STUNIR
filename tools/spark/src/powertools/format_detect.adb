--  format_detect - Detect extraction JSON format variant
--  Identifies which extraction format variant is being used
--  Phase 2 Powertool for STUNIR

pragma SPARK_Mode (Off);

with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;
with Ada.Strings.Fixed;

with GNAT.Command_Line;

with STUNIR_JSON_Parser;
with STUNIR_Types;

procedure Format_Detect is
   use Ada.Command_Line;
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;
   use STUNIR_JSON_Parser;
   use STUNIR_Types;

   --  Exit codes
   Exit_Success          : constant := 0;
   Exit_Validation_Error : constant := 1;
   Exit_Processing_Error : constant := 2;

   --  Format types
   type Extraction_Format is (
      Format_Unknown,
      Format_Direct_Functions,  --  {"functions": [...]}
      Format_Files_Array,       --  {"files": [{"functions": [...]}]}
      Format_Placeholder,       --  {"project": "...", "functions": [...]}
      Format_Legacy_V1          --  Old format
   );

   --  Configuration
   Input_File    : Unbounded_String := Null_Unbounded_String;
   Verbose_Mode  : Boolean := False;
   Show_Version  : Boolean := False;
   Show_Help     : Boolean := False;
   Show_Describe : Boolean := False;
   Output_Json   : Boolean := False;

   Version : constant String := "0.1.0-alpha";

   --  Description output
   Describe_Output : constant String :=
     "{" & ASCII.LF &
     "  ""tool"": ""format_detect""," & ASCII.LF &
     "  ""version"": ""0.1.0-alpha""," & ASCII.LF &
     "  ""description"": ""Detect extraction JSON format variant""," & ASCII.LF &
     "  ""inputs"": [{" & ASCII.LF &
     "    ""name"": ""extraction_json""," & ASCII.LF &
     "    ""type"": ""json""," & ASCII.LF &
     "    ""source"": [""stdin"", ""file""]," & ASCII.LF &
     "    ""required"": true" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  ""outputs"": [{" & ASCII.LF &
     "    ""name"": ""format_info""," & ASCII.LF &
     "    ""type"": ""string""," & ASCII.LF &
     "    ""source"": ""stdout""" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  ""complexity"": ""O(1)""," & ASCII.LF &
     "  ""options"": [" & ASCII.LF &
     "    ""--help"", ""--version"", ""--describe"", ""--json"", ""--verbose""" & ASCII.LF &
     "  ]" & ASCII.LF &
     "}";

   procedure Print_Usage;
   procedure Print_Error (Msg : String);
   procedure Print_Info (Msg : String);
   function Read_Input return String;
   function Detect_Format (Content : String) return Extraction_Format;
   function Format_Name (Fmt : Extraction_Format) return String;

   procedure Print_Usage is
   begin
      Put_Line ("format_detect - Detect extraction JSON format");
      Put_Line ("Version: " & Version);
      Put_Line ("");
      Put_Line ("Usage: format_detect [OPTIONS] [FILE]");
      Put_Line ("");
      Put_Line ("Options:");
      Put_Line ("  --help, -h        Show this help message");
      Put_Line ("  --version, -v     Show version information");
      Put_Line ("  --describe        Show tool description (JSON)");
      Put_Line ("  --json            Output JSON format info");
      Put_Line ("  --verbose         Verbose output");
      Put_Line ("");
      Put_Line ("Arguments:");
      Put_Line ("  FILE              JSON file (default: stdin)");
      Put_Line ("");
      Put_Line ("Exit Codes:");
      Put_Line ("  0                 Format detected successfully");
      Put_Line ("  1                 Invalid/unknown format");
      Put_Line ("  2                 Processing error");
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

   function Read_Input return String is
      Result : Unbounded_String := Null_Unbounded_String;
   begin
      if Input_File = Null_Unbounded_String then
         while not End_Of_File loop
            declare
               Line : String (1 .. 4096);
               Last : Natural;
            begin
               Get_Line (Line, Last);
               Append (Result, Line (1 .. Last));
               Append (Result, ASCII.LF);
            end;
         end loop;
      else
         declare
            File : File_Type;
            Line : String (1 .. 4096);
            Last : Natural;
         begin
            Open (File, In_File, To_String (Input_File));
            while not End_Of_File (File) loop
               Get_Line (File, Line, Last);
               Append (Result, Line (1 .. Last));
               Append (Result, ASCII.LF);
            end loop;
            Close (File);
         exception
            when others =>
               Print_Error ("Cannot read: " & To_String (Input_File));
               return "";
         end;
      end if;
      return To_String (Result);
   end Read_Input;

   function Detect_Format (Content : String) return Extraction_Format is
      State  : Parser_State;
      Status : Status_Code;
      Input_Str : JSON_String;
      Has_Files : Boolean := False;
      Has_Functions : Boolean := False;
      Has_Project : Boolean := False;
      Has_Version : Boolean := False;
   begin
      if Content'Length = 0 then
         return Format_Unknown;
      end if;

      Input_Str := STUNIR_Types.JSON_Strings.To_Bounded_String (Content);
      Initialize_Parser (State, Input_Str, Status);
      if Status /= STUNIR_Types.Success then
         return Format_Unknown;
      end if;

      --  Expect object start
      Next_Token (State, Status);
      if Status /= STUNIR_Types.Success or else State.Current_Token /= Token_Object_Start then
         return Format_Unknown;
      end if;

      --  Scan top-level keys
      loop
         Next_Token (State, Status);
         exit when Status /= STUNIR_Types.Success or else State.Current_Token = Token_Object_End;

         if State.Current_Token = Token_String then
            declare
               Key : constant String := To_String (State.Token_Value);
            begin
               if Key = "files" then
                  Has_Files := True;
                  Skip_Value (State, Status);
               elsif Key = "functions" then
                  Has_Functions := True;
                  Skip_Value (State, Status);
               elsif Key = "project" then
                  Has_Project := True;
                  Skip_Value (State, Status);
               elsif Key = "version" then
                  Has_Version := True;
                  Skip_Value (State, Status);
               else
                  Skip_Value (State, Status);
               end if;
            end;
         elsif State.Current_Token = Token_Comma then
            null;
         else
            exit;
         end if;

         exit when Status /= STUNIR_Types.Success;
      end loop;

      --  Determine format
      if Has_Files then
         return Format_Files_Array;
      elsif Has_Functions and then Has_Project then
         return Format_Placeholder;
      elsif Has_Functions then
         return Format_Direct_Functions;
      elsif Has_Version then
         return Format_Legacy_V1;
      else
         return Format_Unknown;
      end if;
   end Detect_Format;

   function Format_Name (Fmt : Extraction_Format) return String is
   begin
      case Fmt is
         when Format_Direct_Functions =>
            return "direct_functions";
         when Format_Files_Array =>
            return "files_array";
         when Format_Placeholder =>
            return "placeholder";
         when Format_Legacy_V1 =>
            return "legacy_v1";
         when Format_Unknown =>
            return "unknown";
      end case;
   end Format_Name;

   Config : GNAT.Command_Line.Command_Line_Configuration;

begin
   GNAT.Command_Line.Define_Switch (Config, Show_Help'Access, "-h", "--help");
   GNAT.Command_Line.Define_Switch (Config, Show_Version'Access, "-v", "--version");
   GNAT.Command_Line.Define_Switch (Config, Show_Describe'Access, "", "--describe");
   GNAT.Command_Line.Define_Switch (Config, Verbose_Mode'Access, "", "--verbose");
   GNAT.Command_Line.Define_Switch (Config, Output_Json'Access, "", "--json");

   begin
      GNAT.Command_Line.Getopt (Config);
   exception
      when others =>
         Print_Error ("Invalid arguments");
         Set_Exit_Status (Exit_Processing_Error);
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
      Arg : String := GNAT.Command_Line.Get_Argument;
   begin
      if Arg'Length > 0 then
         Input_File := To_Unbounded_String (Arg);
      end if;
   end;

   declare
      Content : constant String := Read_Input;
      Fmt     : constant Extraction_Format := Detect_Format (Content);
   begin
      if Content'Length = 0 then
         Print_Error ("Empty input");
         Set_Exit_Status (Exit_Processing_Error);
         return;
      end if;

      Print_Info ("Detected format: " & Format_Name (Fmt));

      if Output_Json then
         Put_Line ("{");
         Put_Line ("  ""format"": """ & Format_Name (Fmt) & """,");
         Put_Line ("  ""detected"": " & (if Fmt = Format_Unknown then "false" else "true"));
         Put_Line ("}");
      else
         Put_Line (Format_Name (Fmt));
      end if;

      Set_Exit_Status (if Fmt = Format_Unknown then Exit_Validation_Error else Exit_Success);
   end;

exception
   when others =>
      Print_Error ("Unexpected error");
      Set_Exit_Status (Exit_Processing_Error);
end Format_Detect;