with Ada.Text_IO;

procedure test_file_hash is
   function Compute_Hash(file_name : String) return String is
      file_stream : File_Type;
      hash_context : GNAT.SHA256.Context;
      chunk_size : constant := 8192; -- 8KB chunks
      buffer : Stream_Element_Array(1..chunk_size);
      length : Natural := 0;
   begin
      Open(File => file_stream, Mode => In_File, Name => file_name);
      GNAT.SHA256.Initialize(hash_context);

      while not End_Of_File(file_stream) loop
         Read(file_stream, buffer(1..chunk_size), length);
         if length > 0 then
            GNAT.SHA256.Update(hash_context, buffer(1..length));
         end if;
      end loop;

      Close(file_stream);
      return GNAT.SHA256.Digest(hash_context);
   exception
      when others =>
         raise;
   end Compute_Hash;

begin
   -- Create a test file with some content
   declare
      test_file : File_Type;
      test_content : constant String := "This is a test string for file_hash.";
   begin
      Create(File => test_file, Mode => Out_File, Name => "test.txt");
      Write(test_file, test_content);
      Close(test_file);

      -- Compute and print the hash
      Put_Line("Hash of test file:");
      Put_Line(Compute_Hash("test.txt"));
   end;
end test_file_hash;