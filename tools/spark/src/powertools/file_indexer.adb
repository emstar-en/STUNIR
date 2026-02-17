with Ada.Text_IO;
with Ada.Command_Line;
with Ada.Directories;
with Ada.Streams;
with Ada.Streams.Stream_IO;
with Ada.Calendar;
with Ada.Calendar.Formatting;
with Ada.Strings.Unbounded;
with GNAT.SHA256;

procedure File_Indexer is
   use Ada.Text_IO;
   use Ada.Command_Line;
   use Ada.Strings.Unbounded;

   --  Separate use clause to avoid conflicts
   package Dirs renames Ada.Directories;

   procedure Print_Usage is
   begin
      Put_Line (Standard_Error, "Usage: file_indexer [options] [directory]");
      Put_Line (Standard_Error, "");
      Put_Line (Standard_Error, "Options:");
      Put_Line (Standard_Error, "  --recursive      Scan directory recursively (default)");
      Put_Line (Standard_Error, "  --no-recursive   Scan only top-level directory");
      Put_Line (Standard_Error, "  --no-hash        Skip hash computation (faster)");
      Put_Line (Standard_Error, "  --include=GLOB   Include files matching glob (e.g. *.c,*.h)");
      Put_Line (Standard_Error, "  --output=FILE    Write to file instead of stdout");
      Put_Line (Standard_Error, "  --help           Show this help");
      Put_Line (Standard_Error, "  --version        Show version");
      Put_Line (Standard_Error, "  --describe       Show AI introspection data");
   end Print_Usage;

   procedure Print_Describe is
   begin
      Put_Line ("{");
      Put_Line ("  ""name"": ""file_indexer"",");
      Put_Line ("  ""description"": ""Index directory and output manifest JSON with file metadata and hashes"",");
      Put_Line ("  ""version"": ""0.1.0-alpha"",");
      Put_Line ("  ""inputs"": [");
      Put_Line ("    {""name"": ""directory"", ""type"": ""argument"", ""description"": ""Root directory to scan (default: current)""}");
      Put_Line ("  ],");
      Put_Line ("  ""outputs"": [");
      Put_Line ("    {""type"": ""json"", ""schema"": ""stunir_manifest_v1""}");
      Put_Line ("  ],");
      Put_Line ("  ""options"": [");
      Put_Line ("    {""name"": ""--recursive"", ""type"": ""boolean""},");
      Put_Line ("    {""name"": ""--no-hash"", ""type"": ""boolean""},");
      Put_Line ("    {""name"": ""--include"", ""type"": ""string""}");
      Put_Line ("  ]");
      Put_Line ("}");
   end Print_Describe;

   --  Configuration
   Recursive   : Boolean := True;
   Compute_Hash: Boolean := True;
   Root_Dir    : Unbounded_String := To_Unbounded_String (".");
   Output_File : Unbounded_String := Null_Unbounded_String;
   First_Item  : Boolean := True;
   Out_File    : File_Type;
   Use_Stdout  : Boolean := True;

   --  Hash Computation
   function Calculate_Hash (Path : String) return String is
      use Ada.Streams;
      use Ada.Streams.Stream_IO;
      F           : Ada.Streams.Stream_IO.File_Type;
      C           : GNAT.SHA256.Context := GNAT.SHA256.Initial_Context;
      Buffer      : Stream_Element_Array (1 .. 32768);
      Last        : Stream_Element_Offset;
   begin
      if not Dirs.Exists (Path) then
         return "";
      end if;

      Open (F, In_File, Path);
      while not End_Of_File (F) loop
         Read (F, Buffer, Last);
         if Last >= Buffer'First then
            GNAT.SHA256.Update (C, Buffer (Buffer'First .. Last));
         end if;
      end loop;
      Close (F);
      return GNAT.SHA256.Digest (C);
   exception
      when others =>
         if Is_Open (F) then Close (F); end if;
         return "ERROR";
   end Calculate_Hash;

   procedure Write_Output (Text : String) is
   begin
      if Use_Stdout then
         Put_Line (Text);
      else
         Put_Line (Out_File, Text);
      end if;
   end Write_Output;

   procedure Process_File (Item : Dirs.Directory_Entry_Type) is
      Path      : constant String := Dirs.Full_Name (Item);
      SName     : constant String := Dirs.Simple_Name (Item);
      FSize     : constant Dirs.File_Size := Dirs.Size (Item);
      Mod_Time  : constant Ada.Calendar.Time := Dirs.Modification_Time (Item);
      Hash      : Unbounded_String := Null_Unbounded_String;
   begin
      if SName = "." or else SName = ".." then
         return;
      end if;

      --  Compute Hash if requested
      if Compute_Hash then
         Hash := To_Unbounded_String (Calculate_Hash (Path));
      end if;

      if not First_Item then
         if Use_Stdout then Put (","); else Put (Out_File, ","); end if;
         Write_Output ("");
      end if;
      First_Item := False;

      Write_Output ("    {");
      Write_Output ("      ""path"": """ & Path & """,");
      Write_Output ("      ""size"": " & FSize'Image & ",");
      if Compute_Hash then
         Write_Output ("      ""hash"": """ & To_String (Hash) & """,");
      end if;
      Write_Output ("      ""modified"": """ & Ada.Calendar.Formatting.Image (Mod_Time) & """");
      if Use_Stdout then Put ("    }"); else Put (Out_File, "    }"); end if;

   exception
      when others =>
         Put_Line (Standard_Error, "Warning: Failed to process " & Path);
   end Process_File;

   procedure Walk (Dir : String) is
      Search : Dirs.Search_Type;
      Item   : Dirs.Directory_Entry_Type;
   begin
      Dirs.Start_Search (Search, Dir, "");
      while Dirs.More_Entries (Search) loop
         Dirs.Get_Next_Entry (Search, Item);
         declare
            Name : constant String := Dirs.Simple_Name (Item);
         begin
            if Name /= "." and then Name /= ".." then
               case Dirs.Kind (Item) is
                  when Dirs.Directory =>
                     if Recursive then
                        Walk (Dirs.Full_Name (Item));
                     end if;
                  when Dirs.Ordinary_File =>
                     Process_File (Item);
                  when others =>
                     null;
               end case;
            end if;
         end;
      end loop;
      Dirs.End_Search (Search);
   exception
      when Dirs.Name_Error =>
         Put_Line (Standard_Error, "Error: Directory not found - " & Dir);
      when others =>
         Put_Line (Standard_Error, "Error: Failed to walk directory " & Dir);
   end Walk;

begin
   --  Parse Arguments
   for I in 1 .. Argument_Count loop
      declare
         Arg : constant String := Argument (I);
      begin
         if Arg = "--help" then
            Print_Usage;
            return;
         elsif Arg = "--version" then
            Put_Line ("file_indexer 1.0.0");
            return;
         elsif Arg = "--describe" then
            Print_Describe;
            return;
         elsif Arg = "--recursive" then
            Recursive := True;
         elsif Arg = "--no-recursive" then
            Recursive := False;
         elsif Arg = "--no-hash" then
            Compute_Hash := False;
         elsif Arg'Length > 9 and then Arg (1 .. 9) = "--output=" then
            Output_File := To_Unbounded_String (Arg (10 .. Arg'Last));
            Use_Stdout := False;
         elsif Arg (1) /= '-' then
            Root_Dir := To_Unbounded_String (Arg);
         end if;
      end;
   end loop;

   if not Use_Stdout then
      Create (Out_File, Ada.Text_IO.Out_File, To_String (Output_File));
   end if;

   Write_Output ("{");
   Write_Output ("  ""generated_at"": """ & Ada.Calendar.Formatting.Image (Ada.Calendar.Clock) & """,");
   Write_Output ("  ""root"": """ & To_String (Root_Dir) & """,");
   Write_Output ("  ""files"": [");

   Walk (To_String (Root_Dir));

   Write_Output ("");
   Write_Output ("  ]");
   Write_Output ("}");

   if not Use_Stdout then
      Close (Out_File);
   end if;

end File_Indexer;
