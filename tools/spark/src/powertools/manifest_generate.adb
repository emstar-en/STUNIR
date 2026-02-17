with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Directories;
with GNAT.SHA256;

package Manifest_Generate is

   procedure Main;

private

   type File_Record (Size : Natural) is record
      Path  : String (1 .. Size);
      Hash  : String (1 .. 64); -- SHA-256 in hex
      Size  : Long_Long_Integer;
   end record;

end Manifest_Generate;

package body Manifest_Generate is

   procedure Print_Usage is
   begin
      Ada.Text_IO.Put_Line("Usage: manifest_generate [--help] [--version] [--describe]");
      Ada.Text_IO.Put_Line("  --help     Show this help message");
      Ada.Text_IO.Put_Line("  --version  Show version information");
      Ada.Text_IO.Put_Line("  --describe Output tool description in JSON format");
   end Print_Usage;

   procedure Print_Describe is
   begin
      Ada.Text_IO.Put_Line("{");
      Ada.Text_IO.Put_Line("  ""tool"": ""manifest_generate"",");
      Ada.Text_IO.Put_Line("  ""version"": ""0.1.0-alpha"",");
      Ada.Text_IO.Put_Line("  ""description"": ""Generate JSON manifest with file hashes"",");
      Ada.Text_IO.Put_Line("  ""inputs"": [{");
      Ada.Text_IO.Put_Line("    ""name"": ""file_paths"", ""type"": ""text"", ""source"": ""stdin""}");
      Ada.Text_IO.Put_Line("  }],");
      Ada.Text_IO.Put_Line("  ""outputs"": [{");
      Ada.Text_IO.Put_Line("    ""name"": ""manifest_json"", ""type"": ""json"", ""source"": ""stdout""}");
      Ada.Text_IO.Put_Line("  }],");
      Ada.Text_IO.Put_Line("  ""options"": [""--help"", ""--version"", ""--describe""],");
      Ada.Text_IO.Put_Line("  ""complexity"": ""O(n*m) where n is files, m is avg file size"",");
      Ada.Text_IO.Put_Line("  ""pipeline_stage"": ""core""}");
      Ada.Text_IO.Put_Line("}");
   end Print_Describe;

   procedure Print_Version is
   begin
      Ada.Text_IO.Put_Line("0.1.0-alpha");
   end Print_Version;

   function Compute_Hash(File_Name : String) return String is
      File  : Ada.Streams.Stream_Access;
      Hash  : GNAT.SHA256.Digest_Type;
      Chunk : constant := 8_192; -- 8KB chunks
      Buffer : String (1 .. Chunk);
      Last   : Natural;
   begin
      File := Ada.Streams.Stream_IO.Stream (Ada.Text_IO.Mode_In, File_Name);

      GNAT.SHA256.Initialize (Hash);

      loop
         Ada.Text_IO.Get_Line(File, Buffer, Last);
         if Last = 0 then exit; end if;
         GNAT.SHA256.Update (Hash, Buffer(1 .. Last));
      end loop;

      declare
         Hex : String (1 .. 64);
      begin
         for I in 1 .. 32 loop
            Hex(I*2-1) := Character'Val((Hash(I) and 16#F#) + 16#0#);
            Hex(I*2)   := Character'Val((Hash(I) and 16#0F#) + 16#0#);
         end loop;
         return Hex(1 .. 64);
      end;

   exception
      when others =>
         Ada.Text_IO.Put_Line(Ada.Text_IO.Standard_Error, "Warning: Could not read file: " & File_Name);
         return "";
   end Compute_Hash;

   procedure Main is
      use type Ada.Directories.File_Type;
      Current_Dir : constant String := Ada.Directories.Current_Directory;
      Manifest    : array (1 .. 1024) of File_Record;
      Count       : Natural := 0;
      Path        : Unbounded_String;
   begin
      if Ada.Command_Line.Argument_Count = 0 then
         Print_Usage;
         return;
      end if;

      for I in 1 .. Ada.Command_Line.Argument_Count loop
         declare
            Arg : constant String := Ada.Command_Line.Argument(I);
         begin
            if Arg = "--help" then
               Print_Usage; return;
            elsif Arg = "--version" then
               Print_Version; return;
            elsif Arg = "--describe" then
               Print_Describe; return;
            end if;
         end;
      end loop;

      Ada.Text_IO.Put("[");
      declare
         Line : String (1 .. 256);
         Last : Natural;
      begin
         while not Ada.Text_IO.End_of_File(Ada.Text_IO.Standard_Input) loop
            Ada.Text_IO.Get_Line(Standard_Input, Line, Last);

            if Last > 0 then
               declare
                  File_Name : constant String := Line(1 .. Last);
                  Hash      : constant String := Compute_Hash(File_Name);
                  Size      : Long_Long_Integer;
               begin
                  Size := Ada.Directories.Size(File_Name);
                  Count := Count + 1;

                  if Count > 1 then
                     Ada.Text_IO.Put(",");
                  end if;

                  Ada.Text_IO.Put("{");
                  Ada.Text_IO.Put("\"path\": \"" & File_Name & "\", ");
                  Ada.Text_IO.Put("\"hash\": \"" & Hash & "\", ");
                  Ada.Text_IO.Put("\"size\": " & Size.To_String);
                  Ada.Text_IO.Put("}");
               exception
                  when others =>
                     Ada.Text_IO.Put_Line(Ada.Text_IO.Standard_Error, "Warning: Skipping invalid file: " & File_Name);
               end;
            end if;
         end loop;
      end;

      Ada.Text_IO.Put("]");
   end Main;

end Manifest_Generate;