--  json_extract - Extract values from JSON by path
--  Extracts fields from JSON using dot-notation paths
--  Phase 1 Powertool for STUNIR

pragma SPARK_Mode (Off);

with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;
with Ada.Strings.Fixed;
with Ada.IO_Exceptions;

with GNAT.Command_Line;

with STUNIR_JSON_Parser;
with STUNIR_Types;

procedure JSON_Extract is
   use Ada.Command_Line;
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;
   use Ada.Strings.Fixed;
   use STUNIR_Types;

   --  Exit codes per powertools spec
   Exit_Success          : constant := 0;
   Exit_Validation_Error : constant := 1;
   Exit_Processing_Error : constant := 2;
   Exit_Resource_Error   : constant := 3;

   --  Configuration
   Input_File    : Unbounded_String := Null_Unbounded_String;
   Extract_Path  : Unbounded_String := Null_Unbounded_String;
   Default_Value : Unbounded_String := Null_Unbounded_String;
   Verbose_Mode  : Boolean := False;
   Show_Version  : Boolean := False;
   Show_Help     : Boolean := False;
   Show_Describe : Boolean := False;
   Raw_Output    : Boolean := False;

   Version : constant String := "1.0.0";

   --  Description output for --describe
   Describe_Output : constant String :=
     "{" & ASCII.LF &
     "  ""tool"": ""json_extract""," & ASCII.LF &
     "  ""version"": ""1.0.0""," & ASCII.LF &
     "  ""description"": ""Extract values from JSON by path""," & ASCII.LF &
     "  ""inputs"": [{" & ASCII.LF &
     "    ""name"": ""json_input""," & ASCII.LF &
     "    ""type"": ""json""," & ASCII.LF &
     "    ""source"": [""stdin"", ""file""]," & ASCII.LF &
     "    ""required"": true" & ASCII.LF &
     "  }, {" & ASCII.LF &
     "    ""name"": ""path""," & ASCII.LF &
     "    ""type"": ""string""," & ASCII.LF &
     "    ""description"": ""Dot-notation path (e.g., 'functions.0.name')""," & ASCII.LF &
     "    ""required"": true" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  ""outputs"": [{" & ASCII.LF &
     "    ""name"": ""extracted_value""," & ASCII.LF &
     "    ""type"": ""json""," & ASCII.LF &
     "    ""source"": ""stdout""" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  ""complexity"": ""O(n)""," & ASCII.LF &
     "  ""options"": [" & ASCII.LF &
     "    ""--help"", ""--version"", ""--describe"", ""--path"", ""--default"", ""--raw"", ""--verbose""" & ASCII.LF &
     "  ]" & ASCII.LF &
     "}";

   subtype Path_String is String (1 .. 256);
   type Path_Component is record
      Name  : Path_String;
      Len   : Natural;
      Index : Integer;  --  -1 for object keys, >=0 for array indices
   end record;
   type Path_Components is array (1 .. 32) of Path_Component;

   procedure Print_Usage;
   procedure Print_Error (Msg : String);
   procedure Print_Info (Msg : String);
   function Read_Input return String;
   procedure Parse_Path (Path : String; Components : out Path_Components; Count : out Natural);
   function Extract_Value (Content : String; Components : Path_Components; Count : Natural) return String;

   procedure Print_Usage is
   begin
      Put_Line ("json_extract - Extract values from JSON by path");
      Put_Line ("Version: " & Version);
      Put_Line ("");
      Put_Line ("Usage: json_extract [OPTIONS] -p PATH [FILE]");
      Put_Line ("");
      Put_Line ("Options:");
      Put_Line ("  --help, -h        Show this help message");
      Put_Line ("  --version, -v     Show version information");
      Put_Line ("  --describe        Show tool description (JSON)");
      Put_Line ("  --path, -p PATH   Path to extract (e.g., 'functions.0.name')");
      Put_Line ("  --default VALUE   Default value if path not found");
      Put_Line ("  --raw             Output raw strings without quotes");
      Put_Line ("  --verbose         Verbose output");
      Put_Line ("");
      Put_Line ("Arguments:");
      Put_Line ("  FILE              JSON file (default: stdin)");
      Put_Line ("");
      Put_Line ("Examples:");
      Put_Line ("  echo '{""a"":{""b"":1}}' | json_extract -p a.b");
      Put_Line ("  json_extract -p functions.0.name extraction.json");
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
               Print_Error ("Cannot read file: " & To_String (Input_File));
               return "";
         end;
      end if;
      return To_String (Result);
   end Read_Input;

   procedure Parse_Path (Path : String; Components : out Path_Components; Count : out Natural) is
      Start : Positive := Path'First;
      Idx   : Natural := 0;
   begin
      Count := 0;
      for I in Path'Range loop
         if Path (I) = '.' then
            Idx := Idx + 1;
            exit when Idx > Components'Last;
            declare
               Comp : constant String := Path (Start .. I - 1);
            begin
               Components (Idx).Len := Comp'Length;
               Components (Idx).Name (1 .. Comp'Length) := Comp;
               --  Check if it's an array index
               begin
                  Components (Idx).Index := Integer'Value (Comp);
               exception
                  when others =>
                     Components (Idx).Index := -1;
               end;
            end;
            Start := I + 1;
         end if;
      end loop;

      --  Last component
      if Start <= Path'Last then
         Idx := Idx + 1;
         if Idx <= Components'Last then
            declare
               Comp : constant String := Path (Start .. Path'Last);
            begin
               Components (Idx).Len := Comp'Length;
               Components (Idx).Name (1 .. Comp'Length) := Comp;
               begin
                  Components (Idx).Index := Integer'Value (Comp);
               exception
                  when others =>
                     Components (Idx).Index := -1;
               end;
            end;
         end if;
      end if;
      Count := Idx;
   end Parse_Path;

   function Extract_Value (Content : String; Components : Path_Components; Count : Natural) return String is
      use STUNIR_JSON_Parser;
      State  : Parser_State;
      Status : Status_Code;
      Input_Str : JSON_String;
      Current_Comp : Positive := 1;
      In_Target : Boolean := False;
      Result : Unbounded_String := Null_Unbounded_String;
   begin
      if Content'Length = 0 or else Count = 0 then
         return "";
      end if;

      Input_Str := JSON_Strings.To_Bounded_String (Content);
      Initialize_Parser (State, Input_Str, Status);
      if Status /= Success then
         return "";
      end if;

      --  Navigate to target
      loop
         Next_Token (State, Status);
         exit when Status /= Success or else State.Current_Token = Token_EOF;

         if Current_Comp > Count then
            --  We've reached the target - capture value
            if State.Current_Token = Token_String then
               declare
                  Val : constant String := JSON_Strings.To_String (State.Token_Value);
               begin
                  if Raw_Output then
                     return Val;
                  else
                     return """" & Val & """";
                  end if;
               end;
            elsif State.Current_Token in Token_Number | Token_True | Token_False | Token_Null then
               return JSON_Strings.To_String (State.Token_Value);
            elsif State.Current_Token in Token_Object_Start | Token_Array_Start then
               --  Capture complex value
               declare
                  Depth : Natural := 1;
               begin
                  Append (Result, (if State.Current_Token = Token_Object_Start then "{" else "["));
                  loop
                     Next_Token (State, Status);
                     exit when Status /= Success or else Depth = 0;
                     case State.Current_Token is
                        when Token_Object_Start | Token_Array_Start =>
                           Depth := Depth + 1;
                           Append (Result, (if State.Current_Token = Token_Object_Start then "{" else "["));
                        when Token_Object_End | Token_Array_End =>
                           Depth := Depth - 1;
                           if Depth > 0 then
                              Append (Result, (if State.Current_Token = Token_Object_End then "}" else "]"));
                           end if;
                        when Token_String =>
                           Append (Result, """" & JSON_Strings.To_String (State.Token_Value) & """");
                        when others =>
                           Append (Result, JSON_Strings.To_String (State.Token_Value));
                     end case;
                     if Depth > 0 and then State.Current_Token /= Token_Object_Start and then
                        State.Current_Token /= Token_Array_Start then
                        --  Add separator detection would go here
                        null;
                     end if;
                  end loop;
                  return To_String (Result);
               end;
            end if;
         end if;

         --  Navigate based on current token
         if State.Current_Token = Token_String and then Components (Current_Comp).Index = -1 then
            declare
               Key : constant String := JSON_Strings.To_String (State.Token_Value);
               Comp_Name : constant String := Components (Current_Comp).Name (1 .. Components (Current_Comp).Len);
            begin
               if Key = Comp_Name then
                  --  Found matching key, expect colon and value
                  Expect_Token (State, Token_Colon, Status);
                  if Status = Success then
                     Current_Comp := Current_Comp + 1;
                  end if;
               end if;
            end;
         elsif State.Current_Token = Token_Array_Start and then Components (Current_Comp).Index >= 0 then
            declare
               Target_Idx : constant Integer := Components (Current_Comp).Index;
               Current_Idx : Integer := 0;
            begin
               loop
                  Next_Token (State, Status);
                  exit when Status /= Success or else State.Current_Token = Token_Array_End;

                  if Current_Idx = Target_Idx then
                     Current_Comp := Current_Comp + 1;
                     exit;
                  end if;

                  --  Skip value
                  Skip_Value (State, Status);
                  exit when Status /= Success;

                  if State.Current_Token = Token_Comma then
                     Current_Idx := Current_Idx + 1;
                  end if;
               end loop;
            end;
         end if;
      end loop;

      return "";
   end Extract_Value;

   Config : GNAT.Command_Line.Command_Line_Configuration;

begin
   GNAT.Command_Line.Define_Switch (Config, Show_Help'Access, "-h", "--help");
   GNAT.Command_Line.Define_Switch (Config, Show_Version'Access, "-v", "--version");
   GNAT.Command_Line.Define_Switch (Config, Show_Describe'Access, "", "--describe");
   GNAT.Command_Line.Define_Switch (Config, Verbose_Mode'Access, "", "--verbose");
   GNAT.Command_Line.Define_Switch (Config, Raw_Output'Access, "", "--raw");
   GNAT.Command_Line.Define_Switch (Config, Extract_Path'Access, "-p:", "--path=");
   GNAT.Command_Line.Define_Switch (Config, Default_Value'Access, "", "--default=");

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

   if Extract_Path = Null_Unbounded_String then
      Print_Error ("Path required (--path or -p)");
      Set_Exit_Status (Exit_Validation_Error);
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
      Components : Path_Components;
      Comp_Count : Natural;
      Result : String (1 .. 4096);
      Result_Len : Natural := 0;
   begin
      if Content'Length = 0 then
         Print_Error ("Empty input");
         Set_Exit_Status (Exit_Processing_Error);
         return;
      end if;

      Parse_Path (To_String (Extract_Path), Components, Comp_Count);
      Print_Info ("Extracting path with" & Comp_Count'Img & " components");

      declare
         Extracted : constant String := Extract_Value (Content, Components, Comp_Count);
      begin
         if Extracted'Length = 0 then
            if Default_Value /= Null_Unbounded_String then
               Put_Line (To_String (Default_Value));
               Set_Exit_Status (Exit_Success);
            else
               Print_Error ("Path not found: " & To_String (Extract_Path));
               Set_Exit_Status (Exit_Validation_Error);
            end if;
         else
            Put_Line (Extracted);
            Set_Exit_Status (Exit_Success);
         end if;
      end;
   end;

exception
   when others =>
      Print_Error ("Unexpected error");
      Set_Exit_Status (Exit_Processing_Error);
end JSON_Extract;