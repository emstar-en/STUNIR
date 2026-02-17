--  file_writer - Write files with error handling
--  File writing utility for STUNIR powertools
--  Phase 5 Utility for STUNIR

pragma SPARK_Mode (Off);

with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;

procedure File_Writer is
   use Ada.Command_Line;
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;

   Exit_Success          : constant := 0;
   Exit_Validation_Error : constant := 1;
   Exit_Processing_Error : constant := 2;
   Exit_Resource_Error   : constant := 3;

   Output_File   : Unbounded_String := Null_Unbounded_String;
   Show_Version  : Boolean := False;
   Show_Help     : Boolean := False;
   Show_Describe : Boolean := False;
   Force_Write   : Boolean := False;

   Version : constant String := "0.1.0-alpha";

   Describe_Output : constant String :=
     "{" & ASCII.LF &
     "  ""tool"": ""file_writer""," & ASCII.LF &
     "  ""version"": ""0.1.0-alpha""," & ASCII.LF &
     "  ""description"": ""Write files with error handling""," & ASCII.LF &
     "  ""inputs"": [{" & ASCII.LF &
     "    ""name"": ""content""," & ASCII.LF &
     "    ""type"": ""text""," & ASCII.LF &
     "    ""source"": ""stdin""," & ASCII.LF &
     "    ""required"": true" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  ""outputs"": [{" & ASCII.LF &
     "    ""name"": ""file_path""," & ASCII.LF &
     "    ""type"": ""string""," & ASCII.LF &
     "    ""source"": ""file""" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  ""complexity"": ""O(n)""," & ASCII.LF &
     "  ""options"": [" & ASCII.LF &
     "    ""--help"", ""--version"", ""--describe""," & ASCII.LF &
     "    ""--output"", ""--force""" & ASCII.LF &
     "  ]" & ASCII.LF &
     "}";

   procedure Print_Usage;
   procedure Print_Error (Msg : String);
   function Read_Stdin return String;

   procedure Print_Usage is
   begin
      Put_Line ("file_writer - Write files with error handling");
      Put_Line ("Version: " & Version);
      Put_Line ("");
      Put_Line ("Usage: file_writer --output=FILE [OPTIONS]");
      Put_Line ("       cat content.txt | file_writer --output=FILE");
      Put_Line ("");
      Put_Line ("Options:");
      Put_Line ("  --help, -h        Show this help message");
      Put_Line ("  --version, -v     Show version information");
      Put_Line ("  --describe        Show tool description (JSON)");
      Put_Line ("  --output FILE     Output file path (required)");
      Put_Line ("  --force           Overwrite existing file");
      Put_Line ("");
      Put_Line ("Exit Codes:");
      Put_Line ("  0                 Success");
      Put_Line ("  1                 Invalid arguments");
      Put_Line ("  2                 Processing error");
      Put_Line ("  3                 Resource error (file exists, no permission)");
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

   function File_Exists (Path : String) return Boolean is
      File : File_Type;
   begin
      Open (File, In_File, Path);
      Close (File);
      return True;
   exception
      when others =>
         return False;
   end File_Exists;

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
         elsif Arg = "--force" then
            Force_Write := True;
         elsif Arg'Length > 9 and then Arg (1 .. 9) = "--output=" then
            Output_File := To_Unbounded_String (Arg (10 .. Arg'Last));
         elsif Arg = "--output" then
            if I < Argument_Count then
               Output_File := To_Unbounded_String (Argument (I + 1));
            end if;
         end if;
      end;
   end loop;

   if Show_Help then
      Print_Usage;
      return;
   end if;

   if Show_Version then
      Put_Line ("file_writer " & Version);
      return;
   end if;

   if Show_Describe then
      Put_Line (Describe_Output);
      return;
   end if;

   if Length (Output_File) = 0 then
      Print_Error ("No output file specified. Use --output=FILE");
      Set_Exit_Status (Exit_Validation_Error);
      return;
   end if;

   declare
      Path : constant String := To_String (Output_File);
   begin
      if File_Exists (Path) and not Force_Write then
         Print_Error ("File already exists: " & Path & ". Use --force to overwrite");
         Set_Exit_Status (Exit_Resource_Error);
         return;
      end if;

      declare
         Content : constant String := Read_Stdin;
         Out_File : File_Type;
      begin
         Create (Out_File, Out_File, Path);
         Put (Out_File, Content);
         Close (Out_File);
         Put_Line ("Written: " & Path);
         Set_Exit_Status (Exit_Success);
      exception
         when others =>
            Print_Error ("Failed to write file: " & Path);
            Set_Exit_Status (Exit_Resource_Error);
      end;
   end;

end File_Writer;
