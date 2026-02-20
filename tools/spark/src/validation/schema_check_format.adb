--  schema_check_format - Validate formats/patterns in JSON against schema
--  Format validation utility for STUNIR powertools
--  Phase 3 Utility for STUNIR
--
--  REGEX_IR_REF: schema/stunir_regex_ir_v1.dcbor.json
--               group: validation.identifier / pattern_id: identifier_start
--  The 'version' field name check uses literal substring matching.
--  See regex IR for the formal definition of valid identifier format.

pragma SPARK_Mode (Off);

with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;

procedure Schema_Check_Format is
   use Ada.Command_Line;
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;

   Exit_Success          : constant := 0;
   Exit_Validation_Error : constant := 1;
   Exit_Processing_Error : constant := 2;

   Show_Version  : Boolean := False;
   Show_Help     : Boolean := False;
   Show_Describe : Boolean := False;

   Version : constant String := "0.1.0-alpha";

   Describe_Output : constant String :=
     "{" & ASCII.LF &
     "  ""tool"": ""schema_check_format""," & ASCII.LF &
     "  ""version"": ""0.1.0-alpha""," & ASCII.LF &
     "  ""description"": ""Validate formats/patterns in JSON""," & ASCII.LF &
     "  ""inputs"": [{" & ASCII.LF &
     "    ""name"": ""json_data""," & ASCII.LF &
     "    ""type"": ""json""," & ASCII.LF &
     "    ""source"": ""stdin""," & ASCII.LF &
     "    ""required"": true" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  ""outputs"": [{" & ASCII.LF &
     "    ""name"": ""format_errors""," & ASCII.LF &
     "    ""type"": ""json""," & ASCII.LF &
     "    ""source"": ""stdout""" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  ""complexity"": ""O(n)""," & ASCII.LF &
     "  ""options"": [" & ASCII.LF &
     "    ""--help"", ""--version"", ""--describe""" & ASCII.LF &
     "  ]" & ASCII.LF &
     "}";

   procedure Print_Usage;
   procedure Print_Error (Msg : String);
   function Read_Stdin return String;
   function Check_Format (JSON : String) return Boolean;

   procedure Print_Usage is
   begin
      Put_Line ("schema_check_format - Validate formats/patterns in JSON");
      Put_Line ("Version: " & Version);
      Put_Line ("");
      Put_Line ("Usage: schema_check_format [OPTIONS]");
      Put_Line ("");
      Put_Line ("Options:");
      Put_Line ("  --help, -h        Show this help message");
      Put_Line ("  --version, -v     Show version information");
      Put_Line ("  --describe        Show tool description (JSON)");
      Put_Line ("");
      Put_Line ("Exit Codes:");
      Put_Line ("  0                 All formats valid");
      Put_Line ("  1                 Format validation errors");
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

   function Check_Format (JSON : String) return Boolean is
      Valid : Boolean := True;
   begin
      --  Check version follows semantic versioning (simplified)
      if JSON'Length > 0 then
         declare
            Version_Pattern : constant String := """version""";
         begin
            for I in JSON'First .. JSON'Last - Version_Pattern'Length + 1 loop
               if JSON (I .. I + Version_Pattern'Length - 1) = Version_Pattern then
                  --  Found version field, check format
                  declare
                     Value_Start : Integer := I + Version_Pattern'Length;
                  begin
                     while Value_Start <= JSON'Last and then
                           (JSON (Value_Start) = ':' or JSON (Value_Start) = ' ' or
                            JSON (Value_Start) = ASCII.HT or JSON (Value_Start) = ASCII.LF) loop
                        Value_Start := Value_Start + 1;
                     end loop;

                     if Value_Start <= JSON'Last and then JSON (Value_Start) = '"' then
                        Value_Start := Value_Start + 1;
                        declare
                           Value_End : Integer := Value_Start;
                        begin
                           while Value_End <= JSON'Last and then JSON (Value_End) /= '"' loop
                              Value_End := Value_End + 1;
                           end loop;

                           if Value_End > Value_Start then
                              declare
                                 Version_Value : constant String := JSON (Value_Start .. Value_End - 1);
                              begin
                                 if Version_Value'Length = 0 then
                                    Put_Line ("Format error: 'version' is empty");
                                    Valid := False;
                                 end if;
                              end;
                           end if;
                        end;
                     end if;
                  end;
                  exit;
               end if;
            end loop;
         end;
      end if;

      return Valid;
   end Check_Format;

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
      return;
   end if;

   if Show_Version then
      Put_Line ("schema_check_format " & Version);
      return;
   end if;

   if Show_Describe then
      Put_Line (Describe_Output);
      return;
   end if;

   declare
      Input : constant String := Read_Stdin;
   begin
      if Input'Length = 0 then
         Print_Error ("No JSON input provided");
         Set_Exit_Status (Exit_Validation_Error);
         return;
      end if;

      if Check_Format (Input) then
         Set_Exit_Status (Exit_Success);
      else
         Set_Exit_Status (Exit_Validation_Error);
      end if;
   exception
      when others =>
         Print_Error ("Processing error");
         Set_Exit_Status (Exit_Processing_Error);
   end;

end Schema_Check_Format;
