--  func_dedup - Deduplicate functions by signature hash
--  Removes duplicate functions from extraction data
--  Phase 1 Powertool for STUNIR

pragma SPARK_Mode (Off);

with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;
with Ada.Strings.Fixed;
with Ada.Containers.Indefinite_Hashed_Maps;
with Ada.Strings.Hash;

with GNAT.Command_Line;
with GNAT.Strings;

with STUNIR_JSON_Parser;
with STUNIR_Types;

procedure Func_Dedup is
   use Ada.Command_Line;
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;
   use STUNIR_JSON_Parser;
   use STUNIR_Types;

   --  Exit codes
   Exit_Success          : constant := 0;
   Exit_Validation_Error : constant := 1;
   Exit_Processing_Error : constant := 2;

   --  Configuration
   Input_File    : Unbounded_String := Null_Unbounded_String;
   Output_File   : aliased GNAT.Strings.String_Access := null;
   Key_Field     : aliased GNAT.Strings.String_Access := new String'("name");
   Verbose_Mode  : aliased Boolean := False;
   Show_Version  : aliased Boolean := False;
   Show_Help     : aliased Boolean := False;
   Show_Describe : aliased Boolean := False;
   Keep_First    : aliased Boolean := True;

   Version : constant String := "0.1.0-alpha";

   --  Description output
   Describe_Output : constant String :=
     "{" & ASCII.LF &
     "  ""tool"": ""func_dedup""," & ASCII.LF &
     "  ""version"": ""0.1.0-alpha""," & ASCII.LF &
     "  ""description"": ""Deduplicate functions by signature key""," & ASCII.LF &
     "  ""inputs"": [{" & ASCII.LF &
     "    ""name"": ""functions_array""," & ASCII.LF &
     "    ""type"": ""json""," & ASCII.LF &
     "    ""source"": [""stdin"", ""file""]," & ASCII.LF &
     "    ""required"": true" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  ""outputs"": [{" & ASCII.LF &
     "    ""name"": ""deduplicated_functions""," & ASCII.LF &
     "    ""type"": ""json""," & ASCII.LF &
     "    ""source"": ""stdout""" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  ""complexity"": ""O(n)""," & ASCII.LF &
     "  ""options"": [" & ASCII.LF &
     "    ""--help"", ""--version"", ""--describe"", ""--key"", ""--output"", ""--last"", ""--verbose""" & ASCII.LF &
     "  ]" & ASCII.LF &
     "}";


   package String_Maps is new Ada.Containers.Indefinite_Hashed_Maps
     (Key_Type        => String,
      Element_Type    => Natural,
      Hash            => Ada.Strings.Hash,
      Equivalent_Keys => "=");
   procedure Print_Usage;
   procedure Print_Error (Msg : String);
   procedure Print_Info (Msg : String);
   function Read_Input return String;
   procedure Write_Output (Content : String);
   function Deduplicate_Functions (Content : String) return String;

   procedure Print_Usage is
   begin
      Put_Line ("func_dedup - Deduplicate functions by key");
      Put_Line ("Version: " & Version);
      Put_Line ("");
      Put_Line ("Usage: func_dedup [OPTIONS] [INPUT]");
      Put_Line ("");
      Put_Line ("Options:");
      Put_Line ("  --help, -h        Show this help message");
      Put_Line ("  --version, -v     Show version information");
      Put_Line ("  --describe        Show tool description (JSON)");
      Put_Line ("  --key FIELD       Key field for deduplication (default: name)");
      Put_Line ("  --output FILE     Output file (default: stdout)");
      Put_Line ("  --last            Keep last occurrence instead of first");
      Put_Line ("  --verbose         Verbose output");
      Put_Line ("");
      Put_Line ("Arguments:");
      Put_Line ("  INPUT             JSON file with functions array (default: stdin)");
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
      if Output_File = null then
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

   function Deduplicate_Functions (Content : String) return String is
      use STUNIR_JSON_Parser;
      State  : Parser_State;
      Status : Status_Code;
      Input_Str : STUNIR_Types.JSON_String;

      Seen : String_Maps.Map;
      Result_Array : Unbounded_String := Null_Unbounded_String;
      In_Functions_Array : Boolean := False;
      Current_Object : Unbounded_String := Null_Unbounded_String;
      Current_Key : Unbounded_String := Null_Unbounded_String;
      Object_Depth : Natural := 0;
      Functions_Count : Natural := 0;
      Duplicates_Count : Natural := 0;
   begin
      if Content'Length = 0 then
         return "[]";
      end if;

      Input_Str := STUNIR_Types.JSON_Strings.To_Bounded_String (Content);
      Initialize_Parser (State, Input_Str, Status);
      if Status /= STUNIR_Types.Success then
         return "[]";
      end if;

      Append (Result_Array, "[");

      loop
         Next_Token (State, Status);
         exit when Status /= STUNIR_Types.Success or else State.Current_Token = STUNIR_Types.Token_EOF;

         if State.Current_Token = Token_String then
            declare
               Val : constant String := JSON_Strings.To_String (State.Token_Value);
            begin
               if Val = "functions" then
                  Expect_Token (State, Token_Colon, Status);
                  if Status = STUNIR_Types.Success then
                     Expect_Token (State, Token_Array_Start, Status);
                     if Status = STUNIR_Types.Success then
                        In_Functions_Array := True;
                     end if;
                  end if;
               elsif In_Functions_Array and then Object_Depth = 1 and then
                     Val = Key_Field.all then
                  Expect_Token (State, Token_Colon, Status);
                  if Status = STUNIR_Types.Success then
                     Next_Token (State, Status);
                     if Status = STUNIR_Types.Success and then State.Current_Token = Token_String then
                        Current_Key := State.Token_Value;
                     end if;
                  end if;
               end if;
            end;
         elsif State.Current_Token = Token_Object_Start and then In_Functions_Array then
            Object_Depth := Object_Depth + 1;
            if Object_Depth = 1 then
               Current_Object := To_Unbounded_String ("{");
               Current_Key := Null_Unbounded_String;
            else
               Append (Current_Object, "{");
            end if;
         elsif State.Current_Token = Token_Object_End and then In_Functions_Array then
            Object_Depth := Object_Depth - 1;
            if Object_Depth = 0 then
               Append (Current_Object, "}");
               Functions_Count := Functions_Count + 1;

               declare
                  Key_Str : constant Unbounded_String := Current_Key;
               begin
                  if Length (Key_Str) = 0 then
                     --  No key, always include
                     if Length (Result_Array) > 1 then
                        Append (Result_Array, ",");
                     end if;
                     Append (Result_Array, Current_Object);
                  elsif String_Maps.Contains (Seen, Key_Str) then
                     Duplicates_Count := Duplicates_Count + 1;
                     if not Keep_First then
                        --  Replace with last occurrence
                        String_Maps.Replace (Seen, Key_Str, Current_Object);
                     end if;
                  else
                     String_Maps.Insert (Seen, Key_Str, Current_Object);
                  end if;
               end;

               Current_Object := Null_Unbounded_String;
            else
               Append (Current_Object, "}");
            end if;
         elsif In_Functions_Array and then Object_Depth > 0 then
            --  Accumulate object content
            case State.Current_Token is
               when Token_String =>
                  Append (Current_Object, """" & To_String (State.Token_Value) & """");
               when Token_Number | Token_True | Token_False | Token_Null =>
                  Append (Current_Object, To_String (State.Token_Value));
               when Token_Object_Start =>
                  Append (Current_Object, "{");
               when Token_Object_End =>
                  Append (Current_Object, "}");
               when Token_Array_Start =>
                  Append (Current_Object, "[");
               when Token_Array_End =>
                  Append (Current_Object, "]");
               when Token_Colon =>
                  Append (Current_Object, ":");
               when Token_Comma =>
                  Append (Current_Object, ",");
               when others =>
                  null;
            end case;
         end if;
      end loop;

      --  Build output from seen map
      for Cursor in String_Maps.Iterate (Seen) loop
         if Length (Result_Array) > 1 then
            Append (Result_Array, ",");
         end if;
         Append (Result_Array, String_Maps.Element (Cursor));
      end loop;

      Append (Result_Array, "]");

      Print_Info ("Processed" & Functions_Count'Img & " functions, removed" & Duplicates_Count'Img & " duplicates");

      return To_String (Result_Array);
   end Deduplicate_Functions;

   Config : GNAT.Command_Line.Command_Line_Configuration;

begin
   GNAT.Command_Line.Define_Switch (Config, Show_Help'Access, "-h", "--help");
   GNAT.Command_Line.Define_Switch (Config, Show_Version'Access, "-v", "--version");
   GNAT.Command_Line.Define_Switch (Config, Show_Describe'Access, "", "--describe");
   GNAT.Command_Line.Define_Switch (Config, Verbose_Mode'Access, "", "--verbose");
   GNAT.Command_Line.Define_Switch (Config, Keep_First'Access, "", "--last", "", GNAT.Command_Line.Enable_Abbr);
   GNAT.Command_Line.Define_Switch (Config, Key_Field'Access, "", "--key=");
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
      Result  : constant String := Deduplicate_Functions (Content);
   begin
      Write_Output (Result);
      Set_Exit_Status (Exit_Success);
   end;

exception
   when others =>
      Print_Error ("Unexpected error");
      Set_Exit_Status (Exit_Processing_Error);
end Func_Dedup;
