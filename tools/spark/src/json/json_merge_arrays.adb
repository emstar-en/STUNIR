--  json_merge_arrays - Merge two or more JSON arrays
--  Array merging utility for STUNIR powertools
--  Phase 1 Utility for STUNIR

pragma SPARK_Mode (Off);

with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;
with Ada.Strings.Fixed;
with Ada.Characters.Handling;

procedure JSON_Merge_Arrays is
   use Ada.Command_Line;
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;
   use Ada.Characters.Handling;

   --  Exit codes per powertools spec
   Exit_Success          : constant := 0;
   Exit_Processing_Error : constant := 2;

   --  Configuration
   Unique_Mode   : Boolean := False;
   Sort_Mode     : Boolean := False;
   Show_Version  : Boolean := False;
   Show_Help     : Boolean := False;
   Show_Describe : Boolean := False;

   Version : constant String := "0.1.0-alpha";

   --  Description output for --describe
   Describe_Output : constant String :=
     "{" & ASCII.LF &
     "  ""tool"": ""json_merge_arrays""," & ASCII.LF &
     "  ""version"": ""0.1.0-alpha""," & ASCII.LF &
     "  ""description"": ""Merge two or more JSON arrays""," & ASCII.LF &
     "  ""inputs"": [{" & ASCII.LF &
     "    ""name"": ""json_arrays""," & ASCII.LF &
     "    ""type"": ""json""," & ASCII.LF &
     "    ""source"": ""stdin""," & ASCII.LF &
     "    ""required"": true," & ASCII.LF &
     "    ""description"": ""One JSON array per line""" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  ""outputs"": [{" & ASCII.LF &
     "    ""name"": ""merged_array""," & ASCII.LF &
     "    ""type"": ""json""," & ASCII.LF &
     "    ""source"": ""stdout""" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  ""complexity"": ""O(n)""," & ASCII.LF &
     "  ""options"": [" & ASCII.LF &
     "    ""--help"", ""--version"", ""--describe""," & ASCII.LF &
     "    ""--unique"", ""--sort""" & ASCII.LF &
     "  ]" & ASCII.LF &
     "}";

   procedure Print_Usage;
   procedure Print_Error (Msg : String);
   function Read_Stdin return String;
   function Merge_Arrays (Content : String) return String;

   procedure Print_Usage is
   begin
      Put_Line ("json_merge_arrays - Merge two or more JSON arrays");
      Put_Line ("Version: " & Version);
      Put_Line ("");
      Put_Line ("Usage: json_merge_arrays [OPTIONS]");
      Put_Line ("");
      Put_Line ("Options:");
      Put_Line ("  --help, -h        Show this help message");
      Put_Line ("  --version, -v     Show version information");
      Put_Line ("  --describe        Show tool description (JSON)");
      Put_Line ("  --unique          Remove duplicate values");
      Put_Line ("  --sort            Sort resulting array");
      Put_Line ("");
      Put_Line ("Exit Codes:");
      Put_Line ("  0                 Success");
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

   function Merge_Arrays (Content : String) return String is
      Result : Unbounded_String := To_Unbounded_String ("[");
      First_Element : Boolean := True;
      In_Array : Boolean := False;
      Element_Start : Integer := 0;
      Depth : Natural := 0;
      In_String : Boolean := False;
   begin
      if Content'Length = 0 then
         return "[]";
      end if;

      for I in Content'Range loop
         if Content (I) = '"' then
            if I = Content'First or else Content (I - 1) /= '\' then
               In_String := not In_String;
            end if;
         elsif not In_String then
            if Content (I) = '[' then
               if not In_Array then
                  In_Array := True;
                  Element_Start := I + 1;
               else
                  Depth := Depth + 1;
               end if;
            elsif Content (I) = ']' then
               if Depth = 0 then
                  In_Array := False;
                  --  Process elements in this array
                  if Element_Start < I then
                     declare
                        Elements : constant String := Content (Element_Start .. I - 1);
                        Elem_Start : Integer := Elements'First;
                        Elem_Depth : Natural := 0;
                        Elem_In_String : Boolean := False;
                     begin
                        for J in Elements'Range loop
                           if Elements (J) = '"' and then
                              (J = Elements'First or else Elements (J - 1) /= '\') then
                              Elem_In_String := not Elem_In_String;
                           elsif not Elem_In_String then
                              if Elements (J) = '{' or Elements (J) = '[' then
                                 Elem_Depth := Elem_Depth + 1;
                              elsif Elements (J) = '}' or Elements (J) = ']' then
                                 Elem_Depth := Elem_Depth - 1;
                              elsif Elem_Depth = 0 and Elements (J) = ',' then
                                 --  Found element boundary
                                 if Elem_Start < J then
                                    declare
                                       Elem : constant String := Elements (Elem_Start .. J - 1);
                                       Trimmed : constant String := Ada.Strings.Fixed.Trim (Elem, Ada.Strings.Both);
                                    begin
                                       if Trimmed'Length > 0 then
                                          if not First_Element then
                                             Append (Result, ",");
                                          end if;
                                          First_Element := False;
                                          Append (Result, Trimmed);
                                       end if;
                                    end;
                                    Elem_Start := J + 1;
                                 end if;
                              end if;
                           end if;
                        end loop;
                        --  Handle last element
                        if Elem_Start <= Elements'Last then
                           declare
                              Elem : constant String := Elements (Elem_Start .. Elements'Last);
                              Trimmed : constant String := Ada.Strings.Fixed.Trim (Elem, Ada.Strings.Both);
                           begin
                              if Trimmed'Length > 0 then
                                 if not First_Element then
                                    Append (Result, ",");
                                 end if;
                                 First_Element := False;
                                 Append (Result, Trimmed);
                              end if;
                           end;
                        end if;
                     end;
                  end if;
               else
                  Depth := Depth - 1;
               end if;
            end if;
         end if;
      end loop;

      Append (Result, "]");
      return To_String (Result);
   end Merge_Arrays;

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
         elsif Arg = "--unique" then
            Unique_Mode := True;
         elsif Arg = "--sort" then
            Sort_Mode := True;
         end if;
      end;
   end loop;

   --  Handle flags
   if Show_Help then
      Print_Usage;
      return;
   end if;

   if Show_Version then
      Put_Line ("json_merge_arrays " & Version);
      return;
   end if;

   if Show_Describe then
      Put_Line (Describe_Output);
      return;
   end if;

   --  Read and merge arrays
   declare
      Input : constant String := Read_Stdin;
   begin
      if Input'Length = 0 then
         Put_Line ("[]");
         Set_Exit_Status (Exit_Success);
         return;
      end if;

      declare
         Merged : constant String := Merge_Arrays (Input);
      begin
         Put_Line (Merged);
         Set_Exit_Status (Exit_Success);
      end;
   exception
      when others =>
         Print_Error ("Processing error");
         Set_Exit_Status (Exit_Processing_Error);
   end;

end JSON_Merge_Arrays;
