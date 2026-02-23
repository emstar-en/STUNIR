with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Streams;
with Ada.Streams.Stream_IO;
with Ada.Strings.Unbounded;
with GNAT.SHA256;

procedure file_hash is
   use Ada.Text_IO;
   use Ada.Streams;
   use Ada.Command_Line;
   use Ada.Strings.Unbounded;

   EXIT_SUCCESS : constant := 0;
   EXIT_FILE_NOT_FOUND : constant := 1;
   EXIT_PERMISSION_DENIED : constant := 2;
   EXIT_IO_ERROR : constant := 3;

   function Compute_Hash(file_name : String) return String is
      file_stream : Ada.Streams.Stream_IO.File_Type;
      hash_context : GNAT.SHA256.Context := GNAT.SHA256.Initial_Context;
      chunk_size : constant := 8192;
      buffer : Stream_Element_Array(1..chunk_size);
      last : Stream_Element_Offset;
   begin
      Ada.Streams.Stream_IO.Open(File => file_stream, Mode => Ada.Streams.Stream_IO.In_File, Name => file_name);

      while not Ada.Streams.Stream_IO.End_Of_File(file_stream) loop
         Ada.Streams.Stream_IO.Read(file_stream, buffer, last);
         if last >= buffer'First then
            GNAT.SHA256.Update(hash_context, buffer(buffer'First..last));
         end if;
      end loop;

      Ada.Streams.Stream_IO.Close(file_stream);
      return GNAT.SHA256.Digest(hash_context);
   exception
      when Ada.Text_IO.Name_Error =>
         Set_Exit_Status(EXIT_FILE_NOT_FOUND);
         raise;
      when Ada.Text_IO.Use_Error =>
         Set_Exit_Status(EXIT_PERMISSION_DENIED);
         raise;
      when others =>
         Set_Exit_Status(EXIT_IO_ERROR);
         raise;
   end Compute_Hash;

   describe_flag : Boolean := False;
   file_path : Unbounded_String;
   hash_value : Unbounded_String;

begin
   if Argument_Count = 0 then
      Put_Line(Standard_Error, "Usage: file_hash [--describe] <file_path>");
      Put_Line(Standard_Error, "Options:");
      Put_Line(Standard_Error, "  --describe    Show metadata about the hash computation");
      Set_Exit_Status(2);
      return;
   end if;

   if Argument_Count > 0 and then Argument(1) = "--describe" then
      describe_flag := True;
      if Argument_Count < 2 then
         Put_Line(Standard_Error, "Error: Missing file path");
         Set_Exit_Status(2);
         return;
      end if;
      file_path := To_Unbounded_String(Argument(2));
   else
      file_path := To_Unbounded_String(Argument(1));
   end if;

   hash_value := To_Unbounded_String(Compute_Hash(To_String(file_path)));

   if Length(hash_value) /= 64 then
      Put_Line(Standard_Error, "Error: Invalid hash format");
      Set_Exit_Status(3);
      return;
   end if;

   Put_Line(To_String(hash_value));

   if describe_flag then
      Put_Line("--describe output:");
      Put_Line("Algorithm: SHA-256");
      Put_Line("Hash length: 64 characters");
      Put_Line("Format: Lowercase hexadecimal");
      Put_Line("File processed: " & To_String(file_path));
   end if;

   Set_Exit_Status(EXIT_SUCCESS);
exception
   when others =>
      Put_Line(Standard_Error, "Error: File not found or could not be read");
      Set_Exit_Status(2);
end file_hash;
