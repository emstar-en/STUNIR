with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Directories;
with Ada.Strings.Unbounded;
with Ada.Streams.Stream_IO;
with GNAT.SHA256;

procedure manifest_generate is
   use Ada.Text_IO;
   use Ada.Command_Line;
   use Ada.Streams;
   use type Ada.Streams.Stream_Element_Offset;

   function Compute_Hash(File_Name : String) return String is
      File_Stream : Ada.Streams.Stream_IO.File_Type;
      Hash_Context : GNAT.SHA256.Context := GNAT.SHA256.Initial_Context;
      Chunk : constant := 8192;
      Buffer : Stream_Element_Array(1..Chunk);
      Last   : Stream_Element_Offset;
   begin
      Ada.Streams.Stream_IO.Open(File_Stream, Ada.Streams.Stream_IO.In_File, File_Name);

      while not Ada.Streams.Stream_IO.End_Of_File(File_Stream) loop
         Ada.Streams.Stream_IO.Read(File_Stream, Buffer, Last);
         if Last >= Buffer'First then
            GNAT.SHA256.Update(Hash_Context, Buffer(Buffer'First..Last));
         end if;
      end loop;

      Ada.Streams.Stream_IO.Close(File_Stream);
      return GNAT.SHA256.Digest(Hash_Context);
   exception
      when others =>
         Put_Line(Standard_Error, "Warning: Could not read file: " & File_Name);
         return "";
   end Compute_Hash;

   procedure Print_Usage is
   begin
      Put_Line("Usage: manifest_generate [--help] [--version] [--describe]");
      Put_Line("  --help     Show this help message");
      Put_Line("  --version  Show version information");
      Put_Line("  --describe Output tool description in JSON format");
   end Print_Usage;

   procedure Print_Describe is
   begin
      Put_Line("{" &
        """tool"": ""manifest_generate""," &
        """version"": ""0.1.0-alpha""," &
        """description"": ""Generate JSON manifest with file hashes""," &
        """inputs"": [{" &
          """name"": ""file_paths"", ""type"": ""text"", ""source"": ""stdin""" &
        "}]," &
        """outputs"": [{" &
          """name"": ""manifest_json"", ""type"": ""json"", ""source"": ""stdout""" &
        "}]," &
        """options"": [""--help"", ""--version"", ""--describe""]," &
        """complexity"": ""O(n*m) where n is files, m is avg file size""," &
        """pipeline_stage"": ""core""}");
   end Print_Describe;

   procedure Print_Version is
   begin
      Put_Line("0.1.0-alpha");
   end Print_Version;

   Count : Natural := 0;
   Line : String(1..1024);
   Last : Natural;

begin
   for I in 1 .. Argument_Count loop
      declare
         Arg : constant String := Argument(I);
      begin
         if Arg = "--help" then
            Print_Usage;
            return;
         elsif Arg = "--version" then
            Print_Version;
            return;
         elsif Arg = "--describe" then
            Print_Describe;
            return;
         end if;
      end;
   end loop;

   Put_Line("[");
   
   while not End_Of_File(Standard_Input) loop
      Get_Line(Standard_Input, Line, Last);
      
      if Last > 0 then
         declare
            File_Name : constant String := Line(1..Last);
            Hash : constant String := Compute_Hash(File_Name);
            File_Size : Long_Long_Integer;
         begin
            if Hash /= "" then
               begin
                  File_Size := Long_Long_Integer(Ada.Directories.Size(File_Name));
                  
                  if Count > 0 then
                     Put_Line(",");
                  end if;
                  
                  Put("  {""path"": """ & File_Name & """, ");
                  Put("""hash"": """ & Hash & """, ");
                  Put("""size"": " & Long_Long_Integer'Image(File_Size) & "}");
                  
                  Count := Count + 1;
               exception
                  when others =>
                     Put_Line(Standard_Error, "Warning: Skipping invalid file: " & File_Name);
               end;
            end if;
         end;
      end if;
   end loop;
   
   if Count > 0 then
      New_Line;
   end if;
   Put_Line("]");
   
end manifest_generate;
