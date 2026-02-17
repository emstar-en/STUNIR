--  json_formatter - Format JSON with indentation and whitespace
--  JSON beautification utility for STUNIR powertools
--  Phase 1 Utility for STUNIR

pragma SPARK_Mode (Off);

with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;
with Ada.Characters.Handling;

procedure JSON_Formatter is
   use Ada.Command_Line;
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;
   use Ada.Characters.Handling;

   --  Exit codes per powertools spec
   Exit_Success          : constant := 0;
   Exit_Validation_Error : constant := 1;
   Exit_Processing_Error : constant := 2;

   --  Configuration
   Indent_Level  : Natural := 2;
   Compact_Mode  : Boolean := False;
   Sort_Keys     : Boolean := False;
   Show_Version  : Boolean := False;
   Show_Help     : Boolean := False;
   Show_Describe : Boolean := False;

   Version : constant String := "0.1.0-alpha";

   --  Description output for --describe
   Describe_Output : constant String :=
     "{" & ASCII.LF &
     "  ""tool"": ""json_formatter""," & ASCII.LF &
     "  ""version"": ""0.1.0-alpha""," & ASCII.LF &
     "  ""description"": ""Format JSON with indentation and whitespace""," & ASCII.LF &
     "  ""inputs"": [{" & ASCII.LF &
     "    ""name"": ""json_input""," & ASCII.LF &
     "    ""type"": ""json""," & ASCII.LF &
     "    ""source"": ""stdin""," & ASCII.LF &
     "    ""required"": true" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  ""outputs"": [{" & ASCII.LF &
     "    ""name"": ""formatted_json""," & ASCII.LF &
     "    ""type"": ""json""," & ASCII.LF &
     "    ""source"": ""stdout""" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  ""complexity"": ""O(n)""," & ASCII.LF &
     "  ""options"": [" & ASCII.LF &
     "    ""--help"", ""--version"", ""--describe""," & ASCII.LF &
     "    ""--indent"", ""--compact"", ""--sort-keys""" & ASCII.LF &
     "  ]" & ASCII.LF &
     "}";

   procedure Print_Usage;
   procedure Print_Error (Msg : String);
   function Read_Stdin return String;
   function Format_JSON (Content : String) return String;
   function Is_Valid_JSON (Content : String) return Boolean;

   procedure Print_Usage is
   begin
      Put_Line ("json_formatter - Format JSON with indentation");
      Put_Line ("Version: " & Version);
      Put_Line ("");
      Put_Line ("Usage: json_formatter [OPTIONS]");
      Put_Line ("");
      Put_Line ("Options:");
      Put_Line ("  --help, -h        Show this help message");
      Put_Line ("  --version, -v     Show version information");
      Put_Line ("  --describe        Show tool description (JSON)");
      Put_Line ("  --indent N        Indent level (default: 2)");
      Put_Line ("  --compact         Remove all whitespace");
      Put_Line ("  --sort-keys       Sort object keys alphabetically");
      Put_Line ("");
      Put_Line ("Exit Codes:");
      Put_Line ("  0                 Success");
      Put_Line ("  1                 Invalid JSON input");
      Put_Line ("  2                 Processing error");
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
      while not End_Of_File (Standard_Input) loop
         Get_Line (Standard_Input, Line, Last);
         Append (Result, Line (1 .. Last));
      end loop;
      return To_String (Result);
   end Read_Stdin;

   function Is_Valid_JSON (Content : String) return Boolean is
      Depth : Integer := 0;
      In_String : Boolean := False;
      Escape_Next : Boolean := False;
   begin
      if Content'Length = 0 then
         return False;
      end if;

      for I in Content'Range loop
         if Escape_Next then
            Escape_Next := False;
         elsif Content (I) = '\' then
            Escape_Next := True;
         elsif Content (I) = '"' then
            In_String := not In_String;
         elsif not In_String then
            case Content (I) is
               when '{' | '[' =>
                  Depth := Depth + 1;
               when '}' | ']' =>
                  Depth := Depth - 1;
                  if Depth < 0 then
                     return False;
                  end if;
               when others =>
                  null;
            end case;
         end if;
      end loop;

      return Depth = 0 and not In_String;
   end Is_Valid_JSON;

   function Format_JSON (Content : String) return String is
      Result : Unbounded_String := Null_Unbounded_String;
      Depth  : Natural := 0;
      In_String : Boolean := False;
      Escape_Next : Boolean := False;
      Prev_Char : Character := ' ';
      Indent_Str : constant String (1 .. Indent_Level) := (others => ' ');
   begin
      for I in Content'Range loop
         if Escape_Next then
            Escape_Next := False;
            Append (Result, Content (I));
         elsif Content (I) = '\' then
            Escape_Next := True;
            Append (Result, Content (I));
         elsif Content (I) = '"' then
            In_String := not In_String;
            Append (Result, Content (I));
         elsif In_String then
            Append (Result, Content (I));
         elsif Compact_Mode then
            --  Skip whitespace in compact mode
            if not Is_Space (Content (I)) then
               Append (Result, Content (I));
            end if;
         else
            --  Format mode
            case Content (I) is
               when '{' | '[' =>
                  if Prev_Char /= ' ' and Prev_Char /= ':' and Prev_Char /= ',' then
                     Append (Result, ASCII.LF);
                     for J in 1 .. Depth loop
                        Append (Result, Indent_Str);
                     end loop;
                  end if;
                  Append (Result, Content (I));
                  Depth := Depth + 1;
                  Append (Result, ASCII.LF);
                  for J in 1 .. Depth loop
                     Append (Result, Indent_Str);
                  end loop;
               when '}' | ']' =>
                  Depth := Depth - 1;
                  Append (Result, ASCII.LF);
                  for J in 1 .. Depth loop
                     Append (Result, Indent_Str);
                  end loop;
                  Append (Result, Content (I));
               when ',' =>
                  Append (Result, Content (I));
                  Append (Result, ASCII.LF);
                  for J in 1 .. Depth loop
                     Append (Result, Indent_Str);
                  end loop;
               when ':' =>
                  Append (Result, Content (I));
                  Append (Result, ' ');
               when others =>
                  if not Is_Space (Content (I)) then
                     Append (Result, Content (I));
                  end if;
            end case;
         end if;
         Prev_Char := Content (I);
      end loop;

      return To_String (Result);
   end Format_JSON;

begin
   --  Parse command line arguments
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
         elsif Arg = "--compact" then
            Compact_Mode := True;
         elsif Arg = "--sort-keys" then
            Sort_Keys := True;
         elsif Arg'Length > 9 and then Arg (1 .. 9) = "--indent=" then
            Indent_Level := Natural'Value (Arg (10 .. Arg'Last));
         elsif Arg'Length > 8 and then Arg (1 .. 8) = "--indent" then
            --  Handle --indent N format
            if I < Argument_Count then
               Indent_Level := Natural'Value (Argument (I + 1));
            end if;
         end if;
      exception
         when others =>
            Print_Error ("Invalid argument: " & Arg);
            Set_Exit_Status (Exit_Processing_Error);
            return;
      end;
   end loop;

   --  Handle flags
   if Show_Help then
      Print_Usage;
      return;
   end if;

   if Show_Version then
      Put_Line ("json_formatter " & Version);
      return;
   end if;

   if Show_Describe then
      Put_Line (Describe_Output);
      return;
   end if;

   --  Read JSON from stdin
   declare
      Input : constant String := Read_Stdin;
   begin
      if Input'Length = 0 then
         Print_Error ("No input provided");
         Set_Exit_Status (Exit_Validation_Error);
         return;
      end if;

      if not Is_Valid_JSON (Input) then
         Print_Error ("Invalid JSON structure");
         Set_Exit_Status (Exit_Validation_Error);
         return;
      end if;

      declare
         Formatted : constant String := Format_JSON (Input);
      begin
         Put_Line (Formatted);
         Set_Exit_Status (Exit_Success);
      end;
   exception
      when others =>
         Print_Error ("Processing error");
         Set_Exit_Status (Exit_Processing_Error);
   end;

end JSON_Formatter;
