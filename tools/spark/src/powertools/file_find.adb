with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Directories;
with Ada.Strings.Unbounded;

procedure file_find is
   use Ada.Text_IO;
   use Ada.Directories;
   use Ada.Command_Line;
   use Ada.Strings.Unbounded;

   -- Tool metadata
   tool_name : constant String := "file_find";
   version : constant String := "0.1.0-alpha";

   -- Command line flags
   show_help : Boolean := False;
   show_version : Boolean := False;
   show_describe : Boolean := False;

   directory_path : Unbounded_String;
   pattern : Unbounded_String := To_Unbounded_String("*");

begin
   -- Initialize default values
   if Argument_Count = 0 then
      Put_Line(Standard_Error, "Error: Missing directory path");
      Set_Exit_Status(2);
      return;
   end if;

   directory_path := To_Unbounded_String(Argument(1));

   -- Parse flags first (convention)
   for i in 1..Argument_Count loop
      declare
         arg : constant String := Argument(i);
      begin
         if arg = "--help" then
            show_help := True;
         elsif arg = "--version" then
            show_version := True;
         elsif arg = "--describe" then
            show_describe := True;
         end if;
      end;
   end loop;

   -- Handle flags before processing arguments
   if show_help then
      Put_Line("Usage: file_find <directory> [pattern]");
      Put_Line("Options:");
      Put_Line("  --help        Show this help message");
      Put_Line("  --version     Show version information");
      Put_Line("  --describe    Show tool description in JSON format");
      Set_Exit_Status(0);
      return;
   end if;

   if show_version then
      Put_Line(tool_name & " version " & version);
      Set_Exit_Status(0);
      return;
   end if;

   if show_describe then
      Put_Line("{
  ""tool"": """ & tool_name & """,
  ""version"": """ & version & """,
  ""description"": ""Find files matching pattern recursively"",
  ""inputs"": [
    {""name"": ""directory"", ""type"": ""argument"", ""required"": true},
    {""name"": ""pattern"", ""type"": ""argument"", ""required"": false}
  ],
  ""outputs"": [{""name"": ""file_paths"", ""type"": ""text"", ""source"": ""stdout""}],
  ""options"": [""--help"", ""--version"", ""--describe""],
  ""complexity"": ""O(n) where n is number of files"",
  ""pipeline_stage"": ""core""
}");
      Set_Exit_Status(0);
      return;
   end if;

   -- Process remaining arguments (directory and pattern)
   declare
      effective_args : Integer := Argument_Count - (if show_help or show_version or show_describe then 1 else 0);
      dir_index : Integer := 1; -- Start after flags

      -- Find first non-flag argument (directory path)
      while dir_index <= effective_args and then Argument(dir_index) in ["--help", "--version", "--describe"] loop
         dir_index := dir_index + 1;
      end loop;

      if dir_index > effective_args then
         Put_Line(Standard_Error, "Error: Missing directory path");
         Set_Exit_Status(2);
         return;
      end if;

      -- Get directory path (first non-flag argument)
      directory_path := To_Unbounded_String(Argument(dir_index));

      -- Check for pattern argument
      if effective_args > dir_index then
         pattern := To_Unbounded_String(Argument(dir_index + 1));
      end if;

      -- Search for files matching the pattern recursively
      declare
         search_context : Search_Type;
         current_dir : constant String := Get_Directory(directory_path);
         found : Boolean;
      begin
         Start_Search(search_context, current_dir, "*", True); -- Recursive search

         while not End_Search(search_context) loop
            declare
               full_name : constant String := Full_Name(search_context);
               simple_name : constant String := Simple_Name(full_name);
            begin
               if Match(simple_name, pattern) then
                  Put_Line(full_name);
               end if;
            exception
               when others =>
                  -- Skip files we can't access
                  null;
            end;
         end loop;

         Set_Exit_Status(0); -- Success
      exception
         when Data_Error =>
            Put_Line(Standard_Error, "Error: Directory not found");
            Set_Exit_Status(2);
         when others =>
            Put_Line(Standard_Error, "Error: Failed to search directory");
            Set_Exit_Status(2);
      end;
   end;
end file_find;
