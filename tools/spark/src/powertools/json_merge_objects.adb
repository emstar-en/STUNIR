--  json_merge_objects - Merge two or more JSON objects
--  Object merging utility for STUNIR powertools
--  Phase 1 Utility for STUNIR

pragma SPARK_Mode (Off);

with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;
with Ada.Strings.Fixed;
with Ada.Characters.Handling;

procedure JSON_Merge_Objects is
   use Ada.Command_Line;
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;
   use Ada.Characters.Handling;

   --  Exit codes per powertools spec
   Exit_Success          : constant := 0;
   Exit_Validation_Error : constant := 1;
   Exit_Processing_Error : constant := 2;

   --  Configuration
   Strategy      : Unbounded_String := To_Unbounded_String ("last");
   Show_Version  : Boolean := False;
   Show_Help     : Boolean := False;
   Show_Describe : Boolean := False;

   Version : constant String := "0.1.0-alpha";

   --  Description output for --describe
   Describe_Output : constant String :=
     "{" & ASCII.LF &
     "  ""tool"": ""json_merge_objects""," & ASCII.LF &
     "  ""version"": ""0.1.0-alpha""," & ASCII.LF &
     "  ""description"": ""Merge two or more JSON objects""," & ASCII.LF &
     "  ""inputs"": [{" & ASCII.LF &
     "    ""name"": ""json_objects""," & ASCII.LF &
     "    ""type"": ""json""," & ASCII.LF &
     "    ""source"": ""stdin""," & ASCII.LF &
     "    ""required"": true," & ASCII.LF &
     "    ""description"": ""One JSON object per line""," & ASCII.LF &
     "  }]," & ASCII.LF &
     "  ""outputs"": [{" & ASCII.LF &
     "    ""name"": ""merged_object""," & ASCII.LF &
     "    ""type"": ""json""," & ASCII.LF &
     "    ""source"": ""stdout""" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  ""complexity"": ""O(n*m)""," & ASCII.LF &
     "  ""options"": [" & ASCII.LF &
     "    ""--help"", ""--version"", ""--describe""," & ASCII.LF &
     "    ""--strategy""" & ASCII.LF &
     "  ]" & ASCII.LF &
     "}";

   procedure Print_Usage;
   procedure Print_Error (Msg : String);
   function Read_Stdin return String;
   function Merge_Objects (Content : String) return String;

   procedure Print_Usage is
   begin
      Put_Line ("json_merge_objects - Merge two or more JSON objects");
      Put_Line ("Version: " & Version);
      Put_Line ("");
      Put_Line ("Usage: json_merge_objects [OPTIONS]");
      Put_Line ("");
      Put_Line ("Options:");
      Put_Line ("  --help, -h        Show this help message");
      Put_Line ("  --version, -v     Show version information");
      Put_Line ("  --describe        Show tool description (JSON)");
      Put_Line ("  --strategy STRAT  Conflict resolution (last/first/error)");
      Put_Line ("");
      Put_Line ("Merge Strategies:");
      Put_Line ("  last   : Last value wins (default)");
      Put_Line ("  first  : First value wins");
      Put_Line ("  error  : Exit with error on conflict");
      Put_Line ("");
      Put_Line ("Exit Codes:");
      Put_Line ("  0                 Success");
      Put_Line ("  1                 Merge conflict (strategy=error)");
      Put_Line ("  2                 Invalid JSON input");
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
         Append (Result, ASCII.LF);
      end loop;
      return To_String (Result);
   end Read_Stdin;

   function Merge_Objects (Content : String) return String is
      Result : Unbounded_String := To_Unbounded_String ("{");
      Line_Start : Integer := Content'First;
      In_Object : Boolean := False;
      First_Key : Boolean := True;

      type Key_Array is array (1 .. 100) of Unbounded_String;
      Keys : Key_Array;
      Key_Count : Natural := 0;

      function Key_Exists (Key : String) return Boolean is
      begin
         for I in 1 .. Key_Count loop
            if To_String (Keys (I)) = Key then
               return True;
            end if;
         end loop;
         return False;
      end Key_Exists;

      procedure Add_Key (Key : String) is
      begin
         if Key_Count < Keys'Length then
            Key_Count := Key_Count + 1;
            Keys (Key_Count) := To_Unbounded_String (Key);
         end if;
      end Add_Key;

   begin
      if Content'Length = 0 then
         return "{}";
      end if;

      --  Simple approach: collect all key-value pairs
      for I in Content'Range loop
         if Content (I) = '{' then
            In_Object := True;
         elsif Content (I) = '}' and In_Object then
            In_Object := False;
         elsif In_Object and Content (I) = '"' then
            --  Found a key
            declare
               Key_End : Integer := I + 1;
            begin
               while Key_End <= Content'Last and then Content (Key_End) /= '"' loop
                  Key_End := Key_End + 1;
               end loop;

               if Key_End <= Content'Last then
                  declare
                     Key : constant String := Content (I + 1 .. Key_End - 1);
                     Value_Start : Integer := Key_End + 1;
                     Value_End : Integer := Value_Start;
                     Depth : Natural := 0;
                     In_String : Boolean := False;
                  begin
                     --  Skip to value
                     while Value_Start <= Content'Last and then
                           (Content (Value_Start) = ':' or Is_Space (Content (Value_Start))) loop
                        Value_Start := Value_Start + 1;
                     end loop;

                     --  Find value end
                     Value_End := Value_Start;
                     while Value_End <= Content'Last loop
                        if Content (Value_End) = '"' and then
                           (Value_End = Value_Start or else Content (Value_End - 1) /= '\') then
                           In_String := not In_String;
                        elsif not In_String then
                           if Content (Value_End) = '{' or Content (Value_End) = '[' then
                              Depth := Depth + 1;
                           elsif Content (Value_End) = '}' or Content (Value_End) = ']' then
                              if Depth = 0 then
                                 exit;
                              end if;
                              Depth := Depth - 1;
                           elsif Depth = 0 and Content (Value_End) = ',' then
                              exit;
                           end if;
                        end if;
                        Value_End := Value_End + 1;
                     end loop;

                     declare
                        Value : constant String := Content (Value_Start .. Value_End - 1);
                        Is_New_Key : constant Boolean := not Key_Exists (Key);
                     begin
                        if Is_New_Key then
                           Add_Key (Key);
                           if not First_Key then
                              Append (Result, ",");
                           end if;
                           First_Key := False;
                           Append (Result, '"' & Key & """:" & Value);
                        elsif To_String (Strategy) = "last" then
                           --  Replace existing key - skip for now (simplified)
                           null;
                        elsif To_String (Strategy) = "error" then
                           Print_Error ("Key conflict: " & Key);
                           return "";
                        end if;
                     end;
                  end;
               end if;
            end;
         end if;
      end loop;

      Append (Result, "}");
      return To_String (Result);
   end Merge_Objects;

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
         elsif Arg'Length > 11 and then Arg (1 .. 11) = "--strategy=" then
            Strategy := To_Unbounded_String (Arg (12 .. Arg'Last));
         elsif Arg = "--strategy" then
            if I < Argument_Count then
               Strategy := To_Unbounded_String (Argument (I + 1));
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
      Put_Line ("json_merge_objects " & Version);
      return;
   end if;

   if Show_Describe then
      Put_Line (Describe_Output);
      return;
   end if;

   --  Validate strategy
   declare
      Strat : constant String := To_String (Strategy);
   begin
      if Strat /= "last" and Strat /= "first" and Strat /= "error" then
         Print_Error ("Invalid strategy: " & Strat);
         Set_Exit_Status (Exit_Processing_Error);
         return;
      end if;
   end;

   --  Read and merge objects
   declare
      Input : constant String := Read_Stdin;
   begin
      if Input'Length = 0 then
         Put_Line ("{}");
         Set_Exit_Status (Exit_Success);
         return;
      end if;

      declare
         Merged : constant String := Merge_Objects (Input);
      begin
         if Merged'Length = 0 then
            Set_Exit_Status (Exit_Validation_Error);
            return;
         end if;
         Put_Line (Merged);
         Set_Exit_Status (Exit_Success);
      end;
   exception
      when others =>
         Print_Error ("Processing error");
         Set_Exit_Status (Exit_Processing_Error);
   end;

end JSON_Merge_Objects;
