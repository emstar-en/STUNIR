--  json_path_eval - Evaluate JSONPath expressions
--  JSONPath evaluation utility for STUNIR powertools
--  Phase 1 Utility for STUNIR

pragma SPARK_Mode (Off);

with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;

procedure JSON_Path_Eval is
   use Ada.Command_Line;
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;

   --  Exit codes per powertools spec
   Exit_Success          : constant := 0;
   Exit_Validation_Error : constant := 1;
   Exit_Processing_Error : constant := 2;

   --  Configuration
   Path_Expr     : Unbounded_String := Null_Unbounded_String;
   Show_Version  : Boolean := False;
   Show_Help     : Boolean := False;
   Show_Describe : Boolean := False;

   Version : constant String := "0.1.0-alpha";

   --  Description output for --describe
   Describe_Output : constant String :=
     "{" & ASCII.LF &
     "  ""tool"": ""json_path_eval""," & ASCII.LF &
     "  ""version"": ""0.1.0-alpha""," & ASCII.LF &
     "  ""description"": ""Evaluate JSONPath expressions on JSON input""," & ASCII.LF &
     "  ""inputs"": [{" & ASCII.LF &
     "    ""name"": ""json_data""," & ASCII.LF &
     "    ""type"": ""json""," & ASCII.LF &
     "    ""source"": ""stdin""," & ASCII.LF &
     "    ""required"": true" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  ""outputs"": [{" & ASCII.LF &
     "    ""name"": ""path_result""," & ASCII.LF &
     "    ""type"": ""json""," & ASCII.LF &
     "    ""source"": ""stdout""" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  ""complexity"": ""O(n)""," & ASCII.LF &
     "  ""options"": [" & ASCII.LF &
     "    ""--help"", ""--version"", ""--describe""," & ASCII.LF &
     "    ""--path""" & ASCII.LF &
     "  ]" & ASCII.LF &
     "}";

   procedure Print_Usage;
   procedure Print_Error (Msg : String);
   function Read_Stdin return String;
   function Eval_Path (JSON : String; Path : String) return String;

   procedure Print_Usage is
   begin
      Put_Line ("json_path_eval - Evaluate JSONPath expressions");
      Put_Line ("Version: " & Version);
      Put_Line ("");
      Put_Line ("Usage: json_path_eval --path=EXPR [OPTIONS]");
      Put_Line ("");
      Put_Line ("Options:");
      Put_Line ("  --help, -h        Show this help message");
      Put_Line ("  --version, -v     Show version information");
      Put_Line ("  --describe        Show tool description (JSON)");
      Put_Line ("  --path EXPR       JSONPath expression to evaluate");
      Put_Line ("");
      Put_Line ("JSONPath Syntax:");
      Put_Line ("  $                 Root element");
      Put_Line ("  .property         Child property");
      Put_Line ("  [index]           Array index");
      Put_Line ("  [*]               All array elements");
      Put_Line ("");
      Put_Line ("Exit Codes:");
      Put_Line ("  0                 Success");
      Put_Line ("  1                 Invalid path expression");
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

   function Eval_Path (JSON : String; Path : String) return String is
      --  Simplified JSONPath evaluation
      --  Supports: $.property, $[index], $.property[index]
      Result : Unbounded_String := Null_Unbounded_String;
      Current_Pos : Integer := JSON'First;
      In_String : Boolean := False;
   begin
      if Path'Length = 0 or Path (Path'First) /= '$' then
         return "";
      end if;

      if Path = "$" then
         return JSON;
      end if;

      --  Parse path components
      declare
         Path_Pos : Integer := Path'First + 1;
      begin
         while Path_Pos <= Path'Last loop
            if Path (Path_Pos) = '.' then
               --  Property access
               Path_Pos := Path_Pos + 1;
               declare
                  Prop_End : Integer := Path_Pos;
               begin
                  while Prop_End <= Path'Last and then
                        Path (Prop_End) /= '.' and then
                        Path (Prop_End) /= '[' loop
                     Prop_End := Prop_End + 1;
                  end loop;

                  declare
                     Property : constant String := Path (Path_Pos .. Prop_End - 1);
                     Search_Str : constant String := """" & Property & """:";
                     Found_Pos : Integer := 0;
                  begin
                     --  Search for property in JSON
                     for I in Current_Pos .. JSON'Last - Search_Str'Length + 1 loop
                        if JSON (I .. I + Search_Str'Length - 1) = Search_Str then
                           Found_Pos := I + Search_Str'Length;
                           exit;
                        end if;
                     end loop;

                     if Found_Pos = 0 then
                        return "null";
                     end if;

                     --  Find value end
                     Current_Pos := Found_Pos;
                     declare
                        Depth : Natural := 0;
                        In_Str : Boolean := False;
                     begin
                        while Current_Pos <= JSON'Last loop
                           if JSON (Current_Pos) = '"' and then
                              (Current_Pos = JSON'First or else JSON (Current_Pos - 1) /= '\') then
                              In_Str := not In_Str;
                           elsif not In_Str then
                              if JSON (Current_Pos) = '{' or JSON (Current_Pos) = '[' then
                                 Depth := Depth + 1;
                              elsif JSON (Current_Pos) = '}' or JSON (Current_Pos) = ']' then
                                 if Depth = 0 then
                                    exit;
                                 end if;
                                 Depth := Depth - 1;
                              elsif Depth = 0 and (JSON (Current_Pos) = ',' or
                                                    JSON (Current_Pos) = '}') then
                                 exit;
                              end if;
                           end if;
                           Current_Pos := Current_Pos + 1;
                        end loop;
                     end;

                     Result := To_Unbounded_String (JSON (Found_Pos .. Current_Pos - 1));
                  end;

                  Path_Pos := Prop_End;
               end;
            elsif Path (Path_Pos) = '[' then
               --  Array index access
               Path_Pos := Path_Pos + 1;
               declare
                  Index_End : Integer := Path_Pos;
               begin
                  while Index_End <= Path'Last and then Path (Index_End) /= ']' loop
                     Index_End := Index_End + 1;
                  end loop;

                  if Index_End > Path'Last then
                     return "";
                  end if;

                  declare
                     Index_Str : constant String := Path (Path_Pos .. Index_End - 1);
                     Index_Val : Integer := 0;
                     Array_Pos : Integer := Current_Pos;
                     Depth : Natural := 0;
                     In_Str : Boolean := False;
                     Current_Index : Integer := 0;
                     Element_Start : Integer := 0;
                  begin
                     --  Parse index
                     begin
                        Index_Val := Integer'Value (Index_Str);
                     exception
                        when others =>
                           return "";
                     end;

                     --  Find array start
                     while Array_Pos <= JSON'Last and then JSON (Array_Pos) /= '[' loop
                        Array_Pos := Array_Pos + 1;
                     end loop;

                     if Array_Pos > JSON'Last then
                        return "null";
                     end if;

                     Array_Pos := Array_Pos + 1;
                     Element_Start := Array_Pos;

                     --  Find element at index
                     while Array_Pos <= JSON'Last loop
                        if JSON (Array_Pos) = '"' and then
                           (Array_Pos = JSON'First or else JSON (Array_Pos - 1) /= '\') then
                           In_Str := not In_Str;
                        elsif not In_Str then
                           if JSON (Array_Pos) = '{' or JSON (Array_Pos) = '[' then
                              Depth := Depth + 1;
                           elsif JSON (Array_Pos) = '}' or JSON (Array_Pos) = ']' then
                              if Depth = 0 then
                                 if Current_Index = Index_Val then
                                    Result := To_Unbounded_String (JSON (Element_Start .. Array_Pos - 1));
                                    exit;
                                 end if;
                                 exit;
                              end if;
                              Depth := Depth - 1;
                           elsif Depth = 0 and JSON (Array_Pos) = ',' then
                              if Current_Index = Index_Val then
                                 Result := To_Unbounded_String (JSON (Element_Start .. Array_Pos - 1));
                                 exit;
                              end if;
                              Current_Index := Current_Index + 1;
                              Element_Start := Array_Pos + 1;
                           end if;
                        end if;
                        Array_Pos := Array_Pos + 1;
                     end loop;
                  end;

                  Path_Pos := Index_End + 1;
               end;
            else
               Path_Pos := Path_Pos + 1;
            end if;
         end loop;
      end;

      declare
         Res : constant String := To_String (Result);
      begin
         if Res'Length = 0 then
            return "null";
         else
            return Res;
         end if;
      end;
   end Eval_Path;

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
            Path_Expr := To_Unbounded_String (Arg (8 .. Arg'Last));
         elsif Arg = "--path" then
            if I < Argument_Count then
               Path_Expr := To_Unbounded_String (Argument (I + 1));
            end if;
         end if;
      end;
   end loop;

   --  Handle flags
   if Show_Help then
      Print_Usage;
      return;
   end if;

   if Show_Version then
      Put_Line ("json_path_eval " & Version);
      return;
   end if;

   if Show_Describe then
      Put_Line (Describe_Output);
      return;
   end if;

   --  Validate path expression
   if Length (Path_Expr) = 0 then
      Print_Error ("No path expression provided. Use --path=EXPR");
      Set_Exit_Status (Exit_Validation_Error);
      return;
   end if;

   --  Read JSON and evaluate path
   declare
      JSON_Input : constant String := Read_Stdin;
   begin
      if JSON_Input'Length = 0 then
         Print_Error ("No JSON input provided");
         Set_Exit_Status (Exit_Validation_Error);
         return;
      end if;

      declare
         Result : constant String := Eval_Path (JSON_Input, To_String (Path_Expr));
      begin
         if Result'Length = 0 then
            Print_Error ("Invalid path expression or path not found");
            Set_Exit_Status (Exit_Validation_Error);
            return;
         end if;
         Put_Line (Result);
         Set_Exit_Status (Exit_Success);
      end;
   exception
      when others =>
         Print_Error ("Processing error");
         Set_Exit_Status (Exit_Processing_Error);
   end;

end JSON_Path_Eval;
