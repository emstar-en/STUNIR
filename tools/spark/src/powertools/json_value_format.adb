--  json_value_format - Format extracted JSON values for output
--  Value formatting utility for STUNIR powertools
--  Phase 1 Utility for STUNIR

pragma SPARK_Mode (Off);

with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;
with Ada.Strings.Fixed;
with Ada.Characters.Handling;

procedure JSON_Value_Format is
   use Ada.Command_Line;
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;
   use Ada.Characters.Handling;

   --  Exit codes per powertools spec
   Exit_Success : constant := 0;

   --  Configuration
   Raw_Mode      : Boolean := False;
   Type_Annotate : Boolean := False;
   Compact_Mode  : Boolean := False;
   Show_Version  : Boolean := False;
   Show_Help     : Boolean := False;
   Show_Describe : Boolean := False;

   Version : constant String := "0.1.0-alpha";

   --  Description output for --describe
   Describe_Output : constant String :=
     "{" & ASCII.LF &
     "  ""tool"": ""json_value_format""," & ASCII.LF &
     "  ""version"": ""0.1.0-alpha""," & ASCII.LF &
     "  ""description"": ""Format extracted JSON values for output""," & ASCII.LF &
     "  ""inputs"": [{" & ASCII.LF &
     "    ""name"": ""json_value""," & ASCII.LF &
     "    ""type"": ""json""," & ASCII.LF &
     "    ""source"": ""stdin""," & ASCII.LF &
     "    ""required"": true" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  ""outputs"": [{" & ASCII.LF &
     "    ""name"": ""formatted_value""," & ASCII.LF &
     "    ""type"": ""text""," & ASCII.LF &
     "    ""source"": ""stdout""" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  ""complexity"": ""O(n)""," & ASCII.LF &
     "  ""options"": [" & ASCII.LF &
     "    ""--help"", ""--version"", ""--describe""," & ASCII.LF &
     "    ""--raw"", ""--type"", ""--compact""" & ASCII.LF &
     "  ]" & ASCII.LF &
     "}";

   procedure Print_Usage;
   function Read_Stdin return String;
   function Detect_Type (Value : String) return String;
   function Format_Value (Value : String) return String;

   procedure Print_Usage is
   begin
      Put_Line ("json_value_format - Format JSON values for output");
      Put_Line ("Version: " & Version);
      Put_Line ("");
      Put_Line ("Usage: json_value_format [OPTIONS]");
      Put_Line ("");
      Put_Line ("Options:");
      Put_Line ("  --help, -h        Show this help message");
      Put_Line ("  --version, -v     Show version information");
      Put_Line ("  --describe        Show tool description (JSON)");
      Put_Line ("  --raw             Remove quotes from strings");
      Put_Line ("  --type            Add type annotation");
      Put_Line ("  --compact         Minimize whitespace");
   end Print_Usage;

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

   function Detect_Type (Value : String) return String is
      Trimmed : constant String := Ada.Strings.Fixed.Trim (Value, Ada.Strings.Both);
   begin
      if Trimmed'Length = 0 then
         return "unknown";
      end if;

      if Trimmed (Trimmed'First) = '"' and then Trimmed (Trimmed'Last) = '"' then
         return "string";
      elsif Trimmed = "true" or Trimmed = "false" then
         return "boolean";
      elsif Trimmed = "null" then
         return "null";
      elsif Trimmed (Trimmed'First) = '{' then
         return "object";
      elsif Trimmed (Trimmed'First) = '[' then
         return "array";
      else
         --  Check if it's a number
         declare
            All_Digits : Boolean := True;
            Has_Dot    : Boolean := False;
         begin
            for I in Trimmed'Range loop
               if Trimmed (I) = '.' then
                  Has_Dot := True;
               elsif not Is_Digit (Trimmed (I)) and Trimmed (I) /= '-' then
                  All_Digits := False;
                  exit;
               end if;
            end loop;
            if All_Digits then
               if Has_Dot then
                  return "number";
               else
                  return "integer";
               end if;
            end if;
         end;
         return "unknown";
      end if;
   end Detect_Type;

   function Format_Value (Value : String) return String is
      Trimmed : constant String := Ada.Strings.Fixed.Trim (Value, Ada.Strings.Both);
   begin
      if Raw_Mode then
         --  Remove quotes from strings
         if Trimmed'Length >= 2 and then
            Trimmed (Trimmed'First) = '"' and then
            Trimmed (Trimmed'Last) = '"' then
            return Trimmed (Trimmed'First + 1 .. Trimmed'Last - 1);
         end if;
         return Trimmed;
      elsif Compact_Mode then
         return Trimmed;
      elsif Type_Annotate then
         return Trimmed & " (" & Detect_Type (Trimmed) & ")";
      else
         return Trimmed;
      end if;
   end Format_Value;

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
         elsif Arg = "--raw" then
            Raw_Mode := True;
         elsif Arg = "--type" then
            Type_Annotate := True;
         elsif Arg = "--compact" then
            Compact_Mode := True;
         end if;
      end;
   end loop;

   --  Handle flags
   if Show_Help then
      Print_Usage;
      return;
   end if;

   if Show_Version then
      Put_Line ("json_value_format " & Version);
      return;
   end if;

   if Show_Describe then
      Put_Line (Describe_Output);
      return;
   end if;

   --  Read and format value
   declare
      Input : constant String := Read_Stdin;
   begin
      if Input'Length = 0 then
         Put_Line ("");
      else
         Put_Line (Format_Value (Input));
      end if;
      Set_Exit_Status (Exit_Success);
   end;

end JSON_Value_Format;
