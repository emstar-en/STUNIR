--  json_path_parser - Parse dot-notation JSON paths into components
--  Path parsing utility for STUNIR powertools
--  Phase 1 Utility for STUNIR

pragma SPARK_Mode (Off);

with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;
with Ada.Strings.Fixed;

procedure JSON_Path_Parser is
   use Ada.Command_Line;
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;
   use Ada.Strings.Fixed;

   --  Exit codes per powertools spec
   Exit_Success          : constant := 0;
   Exit_Validation_Error : constant := 1;

   --  Configuration
   Input_Path    : Unbounded_String := Null_Unbounded_String;
   Show_Version  : Boolean := False;
   Show_Help     : Boolean := False;
   Show_Describe : Boolean := False;

   Version : constant String := "0.1.0-alpha";

   --  Description output for --describe
   Describe_Output : constant String :=
     "{" & ASCII.LF &
     "  ""tool"": ""json_path_parser""," & ASCII.LF &
     "  ""version"": ""0.1.0-alpha""," & ASCII.LF &
     "  ""description"": ""Parse dot-notation JSON paths into components""," & ASCII.LF &
     "  ""inputs"": [{" & ASCII.LF &
     "    ""name"": ""path""," & ASCII.LF &
     "    ""type"": ""string""," & ASCII.LF &
     "    ""source"": [""stdin"", ""argument""]," & ASCII.LF &
     "    ""required"": true" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  ""outputs"": [{" & ASCII.LF &
     "    ""name"": ""path_components""," & ASCII.LF &
     "    ""type"": ""json""," & ASCII.LF &
     "    ""source"": ""stdout""" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  ""complexity"": ""O(n)""," & ASCII.LF &
     "  ""options"": [" & ASCII.LF &
     "    ""--help"", ""--version"", ""--describe"", ""--path""" & ASCII.LF &
     "  ]" & ASCII.LF &
     "}";

   procedure Print_Usage;
   procedure Print_Error (Msg : String);
   function Read_Stdin return String;
   function Parse_Path (Path : String) return String;
   function Is_Valid_Index (S : String) return Boolean;

   procedure Print_Usage is
   begin
      Put_Line ("json_path_parser - Parse dot-notation JSON paths");
      Put_Line ("Version: " & Version);
      Put_Line ("");
      Put_Line ("Usage: json_path_parser [OPTIONS] [PATH]");
      Put_Line ("");
      Put_Line ("Options:");
      Put_Line ("  --help, -h        Show this help message");
      Put_Line ("  --version, -v     Show version information");
      Put_Line ("  --describe        Show tool description (JSON)");
      Put_Line ("  --path PATH       Path to parse (alternative to stdin)");
      Put_Line ("");
      Put_Line ("Path Syntax:");
      Put_Line ("  Dot notation:     a.b.c");
      Put_Line ("  Array indices:    a.0.b or a[0].b");
      Put_Line ("  Mixed:            a.b[1].c.d[2]");
      Put_Line ("");
      Put_Line ("Exit Codes:");
      Put_Line ("  0                 Success");
      Put_Line ("  1                 Invalid path syntax");
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
      if End_Of_File (Standard_Input) then
         return "";
      end if;
      Get_Line (Standard_Input, Line, Last);
      return Line (1 .. Last);
   end Read_Stdin;

   function Is_Valid_Index (S : String) return Boolean is
   begin
      if S'Length = 0 then
         return False;
      end if;
      for I in S'Range loop
         if S (I) < '0' or S (I) > '9' then
            return False;
         end if;
      end loop;
      return True;
   end Is_Valid_Index;

   function Parse_Path (Path : String) return String is
      Result : Unbounded_String := To_Unbounded_String ("[");
      Current : Unbounded_String := Null_Unbounded_String;
      In_Brackets : Boolean := False;
      I : Integer := Path'First;
   begin
      if Path'Length = 0 then
         return "[]";
      end if;

      while I <= Path'Last loop
         if In_Brackets then
            if Path (I) = ']' then
               --  End of bracket notation
               if Current = Null_Unbounded_String then
                  return ""; --  Error: empty brackets
               end if;
               if Length (Result) > 1 then
                  Append (Result, ",");
               end if;
               Append (Result, '"' & To_String (Current) & '"');
               Current := Null_Unbounded_String;
               In_Brackets := False;
            else
               Append (Current, Path (I));
            end if;
         else
            case Path (I) is
               when '.' =>
                  --  Dot separator
                  if Current /= Null_Unbounded_String then
                     if Length (Result) > 1 then
                        Append (Result, ",");
                     end if;
                     Append (Result, '"' & To_String (Current) & '"');
                     Current := Null_Unbounded_String;
                  elsif Length (Result) = 1 then
                     --  Empty component at start
                     return "";
                  end if;
               when '[' =>
                  --  Start of bracket notation
                  if Current /= Null_Unbounded_String then
                     if Length (Result) > 1 then
                        Append (Result, ",");
                     end if;
                     Append (Result, '"' & To_String (Current) & '"');
                     Current := Null_Unbounded_String;
                  end if;
                  In_Brackets := True;
               when others =>
                  Append (Current, Path (I));
            end case;
         end if;
         I := I + 1;
      end loop;

      --  Handle final component
      if Current /= Null_Unbounded_String then
         if Length (Result) > 1 then
            Append (Result, ",");
         end if;
         Append (Result, '"' & To_String (Current) & '"');
      elsif In_Brackets then
         --  Unclosed bracket
         return "";
      end if;

      Append (Result, "]");
      return To_String (Result);
   end Parse_Path;

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
         elsif Arg'Length > 7 and then Arg (1 .. 7) = "--path=" then
            Input_Path := To_Unbounded_String (Arg (8 .. Arg'Last));
         elsif Arg = "--path" then
            if I < Argument_Count then
               Input_Path := To_Unbounded_String (Argument (I + 1));
            end if;
         elsif Arg (1) /= '-' then
            Input_Path := To_Unbounded_String (Arg);
         end if;
      end;
   end loop;

   --  Handle flags
   if Show_Help then
      Print_Usage;
      return;
   end if;

   if Show_Version then
      Put_Line ("json_path_parser " & Version);
      return;
   end if;

   if Show_Describe then
      Put_Line (Describe_Output);
      return;
   end if;

   --  Get path from stdin or argument
   declare
      Path : Unbounded_String := Input_Path;
   begin
      if Path = Null_Unbounded_String then
         declare
            Stdin_Input : constant String := Read_Stdin;
         begin
            if Stdin_Input'Length = 0 then
               Print_Error ("No path provided");
               Set_Exit_Status (Exit_Validation_Error);
               return;
            end if;
            Path := To_Unbounded_String (Stdin_Input);
         end;
      end if;

      declare
         Parsed : constant String := Parse_Path (To_String (Path));
      begin
         if Parsed'Length = 0 then
            Print_Error ("Invalid path syntax");
            Set_Exit_Status (Exit_Validation_Error);
            return;
         end if;

         Put_Line (Parsed);
         Set_Exit_Status (Exit_Success);
      end;
   end;

end JSON_Path_Parser;
