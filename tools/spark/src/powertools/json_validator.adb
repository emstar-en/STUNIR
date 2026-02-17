--  json_validator - Minimal JSON validator
--
--  This utility provides basic JSON validation functionality
--  that can be called by other powertools.
--

pragma SPARK_Mode (Off);

with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;

procedure Json_Validator is
   use Ada.Command_Line;
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;

   --  Exit codes
   Exit_Success : constant := 0;
   Exit_Error   : constant := 1;

   --  Configuration
   Show_Version  : Boolean := False;
   Show_Help     : Boolean := False;
   Show_Describe : Boolean := False;

   Version : constant String := "0.1.0-alpha";

   Describe_Output : constant String :=
     "{" & ASCII.LF &
     "  ""tool"": ""json_validator""," & ASCII.LF &
     "  ""version"": ""0.1.0-alpha""," & ASCII.LF &
     "  ""description"": ""Validate JSON syntax""," & ASCII.LF &
     "  ""inputs"": [{" & ASCII.LF &
     "    ""name"": ""json""," & ASCII.LF &
     "    ""type"": ""string""," & ASCII.LF &
     "    ""source"": [""stdin""]," & ASCII.LF &
     "    ""required"": true" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  ""outputs"": [{" & ASCII.LF &
     "    ""name"": ""valid""," & ASCII.LF &
     "    ""type"": ""boolean""," & ASCII.LF &
     "    ""source"": ""stdout""" & ASCII.LF &
     "  }]" & ASCII.LF &
     "}";

   procedure Print_Usage is
   begin
      Put_Line ("json_validator - Validate JSON syntax");
      Put_Line ("Version: " & Version);
      Put_Line ("");
      Put_Line ("Usage: json_validator [OPTIONS] < json.txt");
      Put_Line ("");
      Put_Line ("Options:");
      Put_Line ("  --help, -h        Show this help message");
      Put_Line ("  --version, -v     Show version information");
      Put_Line ("  --describe        Show tool description (JSON)");
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
      while not End_Of_File loop
         Get_Line (Line, Last);
         Append (Result, Line (1 .. Last));
      end loop;
      return To_String (Result);
   end Read_Stdin;

   --  Error reporting
   procedure Report_Error (Message : String) is
   begin
      Put_Line (Standard_Error, "ERROR: " & Message);
      Set_Exit_Status (Exit_Error);
   exception
      when others => null;  --  Don't fail on error in error handler
   end Report_Error;

   function Is_Valid_JSON (Content : String) return Boolean is
      Stack_Size : Integer := 0;
      Max_Stack  : constant := 256;
      Stack      : array (1 .. Max_Stack) of Character;
      In_String  : Boolean := False;
      Escaped    : Boolean := False;
   begin
      if Content'Length = 0 then
         return False;
      end if;

      for C of Content loop
         if In_String then
            if Escaped then
               Escaped := False;
            elsif C = '\' then
               Escaped := True;
            elsif C = '"' then
               In_String := False;
            end if;
         else
            case C is
               when '"' =>
                  In_String := True;
               when '{' | '[' =>
                  if Stack_Size >= Max_Stack then
                     return False;
                  end if;
                  Stack_Size := Stack_Size + 1;
                  Stack (Stack_Size) := C;
               when '}' =>
                  if Stack_Size = 0 or else Stack (Stack_Size) /= '{' then
                     return False;
                  end if;
                  Stack_Size := Stack_Size - 1;
               when ']' =>
                  if Stack_Size = 0 or else Stack (Stack_Size) /= '[' then
                     return False;
                  end if;
                  Stack_Size := Stack_Size - 1;
               when others =>
                  null;
            end case;
         end if;
      end loop;

      return Stack_Size = 0 and not In_String;
   end Is_Valid_JSON;

begin
   --  Parse arguments
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
      Content : constant String := Read_Stdin;
   begin
      if Is_Valid_JSON (Content) then
         Put_Line ("true");
         Set_Exit_Status (Exit_Success);
      else
         Put_Line ("false");
         Set_Exit_Status (Exit_Error);
      end if;
   end;

exception
   when others =>
      Set_Exit_Status (Exit_Error);
end Json_Validator;