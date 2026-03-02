--  extraction_to_spec - Convert extraction JSON to STUNIR spec format
--  Normalizes various extraction formats into unified spec format
--  Phase 2 Powertool for STUNIR

pragma SPARK_Mode (Off);

with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;
with Ada.Strings.Fixed;

with GNAT.Command_Line;
with GNAT.Strings;

with STUNIR_JSON_Parser;
with STUNIR_Types;

procedure Extraction_To_Spec is
   use Ada.Command_Line;
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;
   use GNAT.Strings;
   use STUNIR_JSON_Parser;
   use STUNIR_Types;

   --  Exit codes
   Exit_Success          : constant := 0;
   Exit_Validation_Error : constant := 1;
   Exit_Processing_Error : constant := 2;

   --  Configuration
   Input_File    : Unbounded_String := Null_Unbounded_String;
   Output_File   : aliased GNAT.Strings.String_Access := new String'("");
   Source_Lang   : aliased GNAT.Strings.String_Access := new String'("c");
   Verbose_Mode  : aliased Boolean := False;
   Show_Version  : aliased Boolean := False;
   Show_Help     : aliased Boolean := False;
   Show_Describe : aliased Boolean := False;
   Pretty_Print  : aliased Boolean := False;

   Version : constant String := "0.1.0-alpha";

   --  Description output
   Describe_Output : constant String :=
     "{" & ASCII.LF &
     "  ""tool"": ""extraction_to_spec""," & ASCII.LF &
     "  ""version"": ""0.1.0-alpha""," & ASCII.LF &
     "  ""description"": ""Convert extraction JSON to STUNIR spec format""," & ASCII.LF &
     "  ""inputs"": [{" & ASCII.LF &
     "    ""name"": ""extraction_json""," & ASCII.LF &
     "    ""type"": ""json""," & ASCII.LF &
     "    ""source"": [""stdin"", ""file""]," & ASCII.LF &
     "    ""required"": true" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  ""outputs"": [{" & ASCII.LF &
     "    ""name"": ""spec_json""," & ASCII.LF &
     "    ""type"": ""json""," & ASCII.LF &
     "    ""source"": ""stdout""" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  ""complexity"": ""O(n)""," & ASCII.LF &
     "  ""options"": [" & ASCII.LF &
     "    ""--help"", ""--version"", ""--describe"", ""--lang"", ""--output"", ""--pretty"", ""--verbose""" & ASCII.LF &
     "  ]" & ASCII.LF &
     "}";

   procedure Print_Usage;
   procedure Print_Error (Msg : String);
   procedure Print_Info (Msg : String);
   function Read_Input return String;
   procedure Write_Output (Content : String);
   function Convert_To_Spec (Content : String) return String;

   procedure Print_Usage is
   begin
      Put_Line ("extraction_to_spec - Convert extraction to STUNIR spec");
      Put_Line ("Version: " & Version);
      Put_Line ("");
      Put_Line ("Usage: extraction_to_spec [OPTIONS] [INPUT]");
      Put_Line ("");
      Put_Line ("Options:");
      Put_Line ("  --help, -h        Show this help message");
      Put_Line ("  --version, -v     Show version information");
      Put_Line ("  --describe        Show tool description (JSON)");
      Put_Line ("  --lang LANG       Source language (default: c)");
      Put_Line ("  --output FILE     Output file (default: stdout)");
      Put_Line ("  --pretty          Pretty-print output");
      Put_Line ("  --verbose         Verbose output");
      Put_Line ("");
      Put_Line ("Arguments:");
      Put_Line ("  INPUT             Extraction JSON file (default: stdin)");
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

   procedure Write_Output (Content : String) is
   begin
      if Output_File = null or else Output_File.all = "" then
         Put_Line (Content);
      else
         declare
            File : File_Type;
         begin
            Create (File, Out_File, Output_File.all);
            Put (File, Content);
            Close (File);
         exception
            when others =>
               Print_Error ("Cannot write: " & Output_File.all);
         end;
      end if;
   end Write_Output;

   function Escape_Json_String (S : String) return String is
      Result : Unbounded_String := Null_Unbounded_String;
   begin
      for I in S'Range loop
         case S (I) is
            when '"' =>
               Append (Result, '\');
               Append (Result, '"');
            when '\' =>
               Append (Result, '\');
               Append (Result, '\');
            when ASCII.BS =>
               Append (Result, '\');
               Append (Result, 'b');
            when ASCII.FF =>
               Append (Result, '\');
               Append (Result, 'f');
            when ASCII.LF =>
               Append (Result, '\');
               Append (Result, 'n');
            when ASCII.CR =>
               Append (Result, '\');
               Append (Result, 'r');
            when ASCII.HT =>
               Append (Result, '\');
               Append (Result, 't');
            when others =>
               if Character'Pos (S (I)) < 32 then
                  Append (Result, '\');
                  Append (Result, 'u');
                  Append (Result, '0');
                  Append (Result, '0');
                  declare
                     Hex_Digit : constant array (0 .. 15) of Character := "0123456789ABCDEF";
                     Val : constant Integer := Character'Pos (S (I));
                  begin
                     Append (Result, Hex_Digit (Val / 16));
                     Append (Result, Hex_Digit (Val mod 16));
                  end;
               else
                  Append (Result, S (I));
               end if;
         end case;
      end loop;
      return To_String (Result);
   end Escape_Json_String;

   function Convert_To_Spec (Content : String) return String is
      use STUNIR_JSON_Parser;
      use STUNIR_Types;
      State  : Parser_State;
      Status : Status_Code;
      Input_Str : JSON_String;

      Result : Unbounded_String := Null_Unbounded_String;
      Functions_Array : Unbounded_String := Null_Unbounded_String;
      In_Functions : Boolean := False;
      Function_Count : Natural := 0;

      Indent : constant String := (if Pretty_Print then "  " else "");
      Newline : constant String := (if Pretty_Print then "" & ASCII.LF else "");
   begin
      if Content'Length = 0 then
         return "{}";
      end if;

      Input_Str := JSON_Strings.To_Bounded_String (Content);
      Initialize_Parser (State, Input_Str, Status);
      if Status /= STUNIR_Types.Success then
         return "{}";
      end if;

      --  Build spec output
      Append (Result, "{");
      Append (Result, Newline);
      Append (Result, Indent & """spec_version"": ""1.0.0"",");
      Append (Result, Newline);
      Append (Result, Indent & """source_language"": """ & Source_Lang.all & """,");
      Append (Result, Newline);
      Append (Result, Indent & """functions"": [");
      Append (Result, Newline);

      --  Parse and extract functions
      loop
         Next_Token (State, Status);
         exit when Status /= STUNIR_Types.Success or else State.Current_Token = Token_EOF;

         if State.Current_Token = Token_String then
            declare
               Key : constant String := JSON_Strings.To_String (State.Token_Value);
            begin
               if Key = "functions" then
                  Expect_Token (State, Token_Colon, Status);
                  if Status = STUNIR_Types.Success then
                     Expect_Token (State, Token_Array_Start, Status);
                     if Status = STUNIR_Types.Success then
                        In_Functions := True;
                     end if;
                  end if;
               end if;
            end;
         elsif State.Current_Token = Token_Object_Start and then In_Functions then
            --  Found a function object
            if Function_Count > 0 then
               Append (Result, ",");
               Append (Result, Newline);
            end if;
            Function_Count := Function_Count + 1;

            Append (Result, Indent & Indent & "{");
            Append (Result, Newline);

            --  Parse function object
            declare
               First_Member : Boolean := True;
            begin
               loop
                  Next_Token (State, Status);
                  exit when Status /= STUNIR_Types.Success or else State.Current_Token = Token_Object_End;

                  if State.Current_Token = Token_String then
                     declare
                        Member_Key : constant String := JSON_Strings.To_String (State.Token_Value);
                     begin
                        Expect_Token (State, Token_Colon, Status);
                        exit when Status /= STUNIR_Types.Success;

                        Next_Token (State, Status);
                        exit when Status /= STUNIR_Types.Success;

                        if not First_Member then
                           Append (Result, ",");
                           Append (Result, Newline);
                        end if;
                        First_Member := False;

                        Append (Result, Indent & Indent & Indent & """" & Member_Key & """: ");

                        case State.Current_Token is
                           when Token_String =>
                              Append (Result, """" & Escape_Json_String (JSON_Strings.To_String (State.Token_Value)) & """");
                           when Token_Number | Token_True | Token_False | Token_Null =>
                              Append (Result, JSON_Strings.To_String (State.Token_Value));
                           when Token_Array_Start =>
                              Append (Result, "[");
                              declare
                                 Depth : Natural := 1;
                                 Need_Comma : Boolean := False;
                              begin
                                 loop
                                    Next_Token (State, Status);
                                    exit when Status /= STUNIR_Types.Success or else Depth = 0;
                                    case State.Current_Token is
                                       when Token_Array_Start =>
                                          Depth := Depth + 1;
                                          if Need_Comma then
                                             Append (Result, ",");
                                          end if;
                                          Append (Result, "[");
                                          Need_Comma := False;
                                       when Token_Array_End =>
                                          Depth := Depth - 1;
                                          if Depth > 0 then
                                             Append (Result, "]");
                                             Need_Comma := True;
                                          end if;
                                       when Token_Object_Start =>
                                          if Need_Comma then
                                             Append (Result, ",");
                                          end if;
                                          Append (Result, "{");
                                          Need_Comma := False;
                                          declare
                                             Obj_Depth : Natural := 1;
                                             Obj_Need_Comma : Boolean := False;
                                          begin
                                             while Obj_Depth > 0 loop
                                                Next_Token (State, Status);
                                                exit when Status /= STUNIR_Types.Success;
                                                case State.Current_Token is
                                                   when Token_Object_Start =>
                                                      Obj_Depth := Obj_Depth + 1;
                                                      if Obj_Need_Comma then
                                                         Append (Result, ",");
                                                      end if;
                                                      Append (Result, "{");
                                                      Obj_Need_Comma := False;
                                                   when Token_Object_End =>
                                                      Obj_Depth := Obj_Depth - 1;
                                                      if Obj_Depth > 0 then
                                                         Append (Result, "}");
                                                         Obj_Need_Comma := True;
                                                      end if;
                                                   when Token_String =>
                                                      if Obj_Need_Comma then
                                                         Append (Result, ",");
                                                      end if;
                                                      Append (Result, """" & Escape_Json_String (JSON_Strings.To_String (State.Token_Value)) & """");
                                                      Obj_Need_Comma := True;
                                                   when Token_Colon =>
                                                      Append (Result, ":");
                                                      Obj_Need_Comma := False;
                                                   when Token_Number | Token_True | Token_False | Token_Null =>
                                                      Append (Result, JSON_Strings.To_String (State.Token_Value));
                                                      Obj_Need_Comma := True;
                                                   when Token_Comma =>
                                                      Obj_Need_Comma := True;
                                                   when others =>
                                                      null;
                                                end case;
                                             end loop;
                                          end;
                                          Append (Result, "}");
                                          Need_Comma := True;
                                       when Token_String =>
                                          if Need_Comma then
                                             Append (Result, ",");
                                          end if;
                                          declare
                                             Array_String : constant String := JSON_Strings.To_String (State.Token_Value);
                                          begin
                                             Append (Result, """" & Escape_Json_String (Array_String) & """");
                                          end;
                                          Need_Comma := True;
                                       when Token_Number | Token_True | Token_False | Token_Null =>
                                          if Need_Comma then
                                             Append (Result, ",");
                                          end if;
                                          Append (Result, JSON_Strings.To_String (State.Token_Value));
                                          Need_Comma := True;
                                       when Token_Comma =>
                                          Need_Comma := True;
                                       when others =>
                                          null;
                                    end case;
                                 end loop;
                              end;
                              Append (Result, "]");
                           when others =>
                              Skip_Value (State, Status);
                              Append (Result, "null");
                        end case;
                     end;
                  elsif State.Current_Token = Token_Comma then
                     null;
                  else
                     exit;
                  end if;
               end loop;
            end;

            Append (Result, Newline);
            Append (Result, Indent & Indent & "}");
         elsif State.Current_Token = Token_Array_End and then In_Functions then
            In_Functions := False;
         end if;
      end loop;

      Append (Result, Newline);
      Append (Result, Indent & "]");
      Append (Result, Newline);
      Append (Result, "}");

      Print_Info ("Converted" & Function_Count'Img & " functions");

      return To_String (Result);
   end Convert_To_Spec;

   Config : GNAT.Command_Line.Command_Line_Configuration;

begin
   GNAT.Command_Line.Define_Switch (Config, Show_Help'Access, "-h", "--help");
   GNAT.Command_Line.Define_Switch (Config, Show_Version'Access, "-v", "--version");
   GNAT.Command_Line.Define_Switch (Config, Show_Describe'Access, "", "--describe");
   GNAT.Command_Line.Define_Switch (Config, Verbose_Mode'Access, "", "--verbose");
   GNAT.Command_Line.Define_Switch (Config, Pretty_Print'Access, "", "--pretty");
   GNAT.Command_Line.Define_Switch (Config, Source_Lang'Access, "", "--lang=");
   GNAT.Command_Line.Define_Switch (Config, Output_File'Access, "-o:", "--output=");

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
      Result  : constant String := Convert_To_Spec (Content);
   begin
      if Content'Length = 0 then
         Print_Error ("Empty input");
         Set_Exit_Status (Exit_Processing_Error);
         return;
      end if;

      Write_Output (Result);
      Set_Exit_Status (Exit_Success);
   end;

exception
   when others =>
      Print_Error ("Unexpected error");
      Set_Exit_Status (Exit_Processing_Error);
end Extraction_To_Spec;
