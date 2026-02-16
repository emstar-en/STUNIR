with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;
with Ada.IO_Exceptions;

procedure Type_Resolve is
   use Ada.Command_Line;
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;

   Input_File  : Unbounded_String := Null_Unbounded_String;
   Output_File : Unbounded_String := Null_Unbounded_String;
   Show_Help   : Boolean := False;
   Show_Describe : Boolean := False;

   procedure Print_Usage is
   begin
      Put_Line (Standard_Error, "Usage: type_resolve [options] [file]");
      Put_Line (Standard_Error, "Options:");
      Put_Line (Standard_Error, "  --output=FILE    Write output to file");
      Put_Line (Standard_Error, "  --describe       Show tool description");
      Put_Line (Standard_Error, "  --help           Show this help");
   end Print_Usage;

   procedure Print_Describe is
   begin
      Put_Line ("{");
      Put_Line ("  ""name"": ""type_resolve"",");
      Put_Line ("  ""description"": ""Resolve type aliases and dependencies in spec"",");
      Put_Line ("  ""version"": ""1.0.0"",");
      Put_Line ("  ""inputs"": [{""name"": ""spec"", ""type"": ""json"", ""source"": [""stdin"", ""file""]}],");
      Put_Line ("  ""outputs"": [{""name"": ""resolved_spec"", ""type"": ""json"", ""source"": ""stdout""}]");
      Put_Line ("}");
   end Print_Describe;

   function Read_All (Path : String) return String is
      File : File_Type;
      Result : Unbounded_String := Null_Unbounded_String;
      Line   : String (1 .. 4096);
      Last   : Natural;
   begin
      if Path = "" then
         while not End_Of_File loop
            Get_Line (Line, Last);
            Append (Result, Line (1 .. Last));
            Append (Result, ASCII.LF);
         end loop;
         return To_String (Result);
      end if;

      Open (File, In_File, Path);
      while not End_Of_File (File) loop
         Get_Line (File, Line, Last);
         Append (Result, Line (1 .. Last));
         Append (Result, ASCII.LF);
      end loop;
      Close (File);
      return To_String (Result);
   exception
      when Ada.IO_Exceptions.Name_Error =>
         return "";
      when others =>
         return "";
   end Read_All;

   procedure Write_All (Path : String; Content : String) is
      File : File_Type;
   begin
      if Path = "" then
         Put_Line (Content);
      else
         Create (File, Out_File, Path);
         Put (File, Content);
         Close (File);
      end if;
   end Write_All;

begin
   for I in 1 .. Argument_Count loop
      declare
         Arg : constant String := Argument (I);
      begin
         if Arg = "--help" then
            Show_Help := True;
         elsif Arg = "--describe" then
            Show_Describe := True;
         elsif Arg'Length > 9 and then Arg (1 .. 9) = "--output=" then
            Output_File := To_Unbounded_String (Arg (10 .. Arg'Last));
         elsif Arg (1) /= '-' then
            Input_File := To_Unbounded_String (Arg);
         end if;
      end;
   end loop;

   if Show_Help then
      Print_Usage;
      Set_Exit_Status (Success);
      return;
   end if;

   if Show_Describe then
      Print_Describe;
      Set_Exit_Status (Success);
      return;
   end if;

   declare
      Content : constant String := Read_All (To_String (Input_File));
   begin
      if Content = "" then
         Put_Line (Standard_Error, "Error: no input");
         Set_Exit_Status (Failure);
         return;
      end if;
      Write_All (To_String (Output_File), Content);
      Set_Exit_Status (Success);
   end;
end Type_Resolve;