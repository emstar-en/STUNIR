"with Ada.Text_IO;
procedure test_progress_reporting is
   procedure test_file_size_calculation is
      file_stream : Ada.Text_IO.File_Type;
      file_size : Natural := 0;
   begin
      -- Create a test file
      Ada.Text_IO.Create(file_stream, Ada.Text_IO.Out_File, \"test_small.txt\");
      Ada.Text_IO.Put_Line(file_stream, \"Small test content\");
      Ada.Text_IO.Close(file_stream);

      -- Calculate file size (simulating what file_hash does)
      Ada.Text_IO.Reset(file_stream);
      Ada.Text_IO.Set_Mode(file_stream, Ada.Text_IO.In_File);
      Ada.Text_IO.Seek(file_stream, 1);
      file_size := Ada.Text_IO.Position(file_stream) - 1;
      Ada.Text_IO.Reset(file_stream);

      Ada.Text_IO.Put_Line(\"Test file size: \" & Natural'Image(file_size));
   end test_file_size_calculation;

begin
   test_file_size_calculation;
end test_progress_reporting;"