--  json_read - Read and validate JSON from file or stdin
--  Phase 1 Powertool for STUNIR

pragma SPARK_Mode (Off);

with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;
with Ada.Exceptions;

procedure Json_Read is
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
     "  ""tool"": ""json_read""," & ASCII.LF &
     "  ""version"": ""0.1.0-alpha""," & ASCII.LF &
     "  ""description"": ""Read and validate JSON from file or stdin""," & ASCII.LF &
     "  ""inputs"": [{" & ASCII.LF &
     "    ""name"": ""json""," & ASCII.LF &
     "    ""type"": ""string""," & ASCII.LF &
     "    ""source"": [""stdin"", ""file""]," & ASCII.LF &
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
      Put_Line ("json_read - Read and validate JSON");
      Put_Line ("Version: " & Version);
      Put_Line ("");
      Put_Line ("Usage: json_read [OPTIONS] [FILE]");
      Put_Line ("");
      Put_Line ("Options:");
      Put_Line ("  --help, -h        Show this help message");
      Put_Line ("  --version, -v     Show version information");
      Put_Line ("  --describe        Show tool description (JSON)");
      Put_Line ("");
      Put_Line ("Arguments:");
      Put_Line ("  FILE              JSON file to read (default: stdin)");
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

   function Read_File (Filename : String) return String is
      File   : File_Type;
      Result : Unbounded_String := Null_Unbounded_String;
      Line   : String (1 .. 4096);
      Last   : Natural;
   begin
      Open (File, In_File, Filename);
      while not End_Of_File (File) loop
         Get_Line (File, Line, Last);
         Append (Result, Line (1 .. Last));
      end loop;
      Close (File);
      return To_String (Result);
   exception
      when others =>
         Print_Error ("Cannot read file: " & Filename);
         return "";
   end Read_File;

   function Is_Valid_JSON (S : String) return Boolean is
      Brace_Count : Integer := 0;
      In_String   : Boolean := False;
      Escaped     : Boolean := False;
   begin
      if S'Length = 0 then
         return False;
      end if;

      for C of S loop
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
                  Brace_Count := Brace_Count + 1;
               when '}' | ']' =>
                  Brace_Count := Brace_Count - 1;
                  if Brace_Count < 0 then
                     return False;
                  end if;
               when others =>
                  null;
            end case;
         end if;
      end loop;

      return Brace_Count = 0 and not In_String;
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

   --  Read input
   declare
      Content : Unbounded_String := Null_Unbounded_String;
   begin
      if Argument_Count = 0 or else Argument (Argument_Count) (1) = '-' then
         Content := To_Unbounded_String (Read_Stdin);
      else
         Content := To_Unbounded_String (Read_File (Argument (Argument_Count)));
      end if;

      if Length (Content) = 0 then
         Print_Error ("Empty input");
         Set_Exit_Status (Exit_Error);
         return;
      end if;

      if Is_Valid_JSON (To_String (Content)) then
         Put_Line ("true");
         Set_Exit_Status (Exit_Success);
      else
         Put_Line ("false");
         Set_Exit_Status (Exit_Error);
      end if;
   end;

exception
   when others =>
      Print_Error ("Unexpected error");
      Set_Exit_Status (Exit_Error);
end Json_Read;