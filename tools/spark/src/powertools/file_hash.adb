with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Streams;
with Ada.Streams.Stream_IO;
with GNAT.SHA256;

procedure file_hash is
   use Ada.Text_IO;
   use Ada.Streams;
   use Ada.Streams.Stream_IO;
   use Ada.Command_Line;

   type Stream_Access is access root Stream_Type'Class;

   function Compute_Hash(file_name : String) return String is
      file_stream : File_Type;
      hash_context : GNAT.SHA256.Context;
      chunk_size : constant := 8192; -- 8KB chunks
      buffer : Stream_Element_Array(1..chunk_size);
      length : Natural := 0;
      bytes_processed : Natural := 0;
      file_size : Natural := 0;
begin
      Open(File => file_stream, Mode => In_File, Name => file_name);

      -- Get file size for progress calculation
      Reset(file_stream);
      Set_Mode(file_stream, In_File);
      Seek(file_stream, 1);
      file_size := Position(file_stream) - 1;
      Reset(file_stream);

      GNAT.SHA256.Initialize(hash_context);

      while not End_Of_File(file_stream) loop
         Read(file_stream, buffer(1..chunk_size), length);
         if length > 0 then
            GNAT.SHA256.Update(hash_context, buffer(1..length));
            bytes_processed := bytes_processed + length;

            -- Progress reporting with percentage for large files (>8KB)
            if file_size > chunk_size and then bytes_processed mod (chunk_size * 4) = 0 then
               declare
                  progress : constant Natural := (bytes_processed * 100) / file_size;
               begin
                  Put_Line("Processing: " & Natural'Image(progress) & "% (" &
                           Natural'Image(bytes_processed) & "/" &
                           Natural'Image(file_size) & " bytes)");
               end;
            end if;
         end if;
      end loop;

      Close(file_stream);
      return GNAT.SHA256.Digest(hash_context);
   exception
      when Name_Error =>
         Set_Exit_Status(EXIT_FILE_NOT_FOUND);
         raise;
      when Use_Error =>
         Set_Exit_Status(EXIT_PERMISSION_DENIED);
         raise;
      when others =>
         Set_Exit_Status(EXIT_IO_ERROR);
         raise;
   end Compute_Hash;

begin
   -- Constants for exit codes
   EXIT_SUCCESS : constant := 0;
   EXIT_FILE_NOT_FOUND : constant := 1;
   EXIT_PERMISSION_DENIED : constant := 2;
   EXIT_IO_ERROR : constant := 3;
   -- Parse command line arguments and options
   if Argument_Count = 0 then
      Put_Line(Standard_Error, "Usage: file_hash [--describe] <file_path>");
      Put_Line(Standard_Error, "Options:");
      Put_Line(Standard_Error, "  --describe    Show metadata about the hash computation");
      Set_Exit_Status(2);
      return;
   end if;

   declare
      describe_flag : Boolean := False;
      file_path : constant String := Argument(1);

      -- Check for --describe flag
      if Argument_Count > 1 and then Argument(1) = "--describe" then
         describe_flag := True;
         if Argument_Count < 2 then
            Put_Line(Standard_Error, "Error: Missing file path");
            Set_Exit_Status(2);
            return;
         end if;
         file_path := Argument(2);
      end if;

      hash_value : constant String := Compute_Hash(file_path);

      -- Validate hash length (should be 64 characters for SHA-256)
      if hash_value'Length /= 64 then
         Put_Line(Standard_Error, "Error: Invalid hash format");
         Set_Exit_Status(3);
         return;
      end if;

      -- Output the hash
      Put_Line(hash_value);

      -- Describe mode output
      if describe_flag then
         Put_Line("--describe output:");
         Put_Line("Algorithm: SHA-256");
         Put_Line("Hash length: 64 characters");
         Put_Line("Format: Lowercase hexadecimal");
         Put_Line("File processed: " & file_path);
      end if;

      Set_Exit_Status(EXIT_SUCCESS); -- Success
   exception
      when others =>
         Put_Line(Standard_Error, "Error: File not found or could not be read");
         Set_Exit_Status(2);
   end;
end file_hash;

