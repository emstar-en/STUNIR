with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Directories;
with Ada.Strings.Unbounded;
with Ada.IO_Exceptions;

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
      Put_Line("{" &
        """tool"": """ & tool_name & """," &
        """version"": """ & version & """," &
        """description"": ""Find files matching pattern recursively""," &
        """inputs"": [" &
          "{""name"": ""directory"", ""type"": ""argument"", ""required"": true}," &
          "{""name"": ""pattern"", ""type"": ""argument"", ""required"": false}" &
        "]," &
        """outputs"": [{""name"": ""file_paths"", ""type"": ""text"", ""source"": ""stdout""}]," &
        """options"": [""--help"", ""--version"", ""--describe""]," &
        """complexity"": ""O(n) where n is number of files""," &
        """pipeline_stage"": ""core""" &
      "}");
      Set_Exit_Status(0);
      return;
   end if;

   declare
      dir_index : Integer := 1;
      found_dir : Boolean := False;
   begin
      while dir_index <= Argument_Count loop
         declare
            arg : constant String := Argument(dir_index);
         begin
            if arg /= "--help" and arg /= "--version" and arg /= "--describe" then
               if not found_dir then
                  directory_path := To_Unbounded_String(arg);
                  found_dir := True;
               else
                  pattern := To_Unbounded_String(arg);
               end if;
            end if;
         end;
         dir_index := dir_index + 1;
      end loop;

      if not found_dir then
         Put_Line(Standard_Error, "Error: Missing directory path");
         Set_Exit_Status(2);
         return;
      end if;

      declare
         procedure Search_Recursive(Dir : String) is
            Search : Search_Type;
            Dir_Entry : Directory_Entry_Type;
            Pattern_Str : constant String := To_String(pattern);
         begin
            Start_Search(Search, Dir, "*");
            while More_Entries(Search) loop
               Get_Next_Entry(Search, Dir_Entry);
               declare
                  Name : constant String := Simple_Name(Dir_Entry);
                  Full : constant String := Full_Name(Dir_Entry);
               begin
                  if Name /= "." and Name /= ".." then
                     if Kind(Dir_Entry) = Directory then
                        Search_Recursive(Full);
                     else
                        if Pattern_Str = "*" or else Name = Pattern_Str then
                           Put_Line(Full);
                        end if;
                     end if;
                  end if;
               exception
                  when others =>
                     null;
               end;
            end loop;
            End_Search(Search);
         exception
            when others =>
               null;
         end Search_Recursive;
      begin
         Search_Recursive(To_String(directory_path));
         Set_Exit_Status(0);
      exception
         when Ada.IO_Exceptions.Name_Error =>
            Put_Line(Standard_Error, "Error: Directory not found");
            Set_Exit_Status(2);
         when others =>
            Put_Line(Standard_Error, "Error: Failed to search directory");
            Set_Exit_Status(2);
      end;
   end;
end file_find;
