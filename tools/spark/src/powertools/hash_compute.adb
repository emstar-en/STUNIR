with Ada.Text_IO;
with Ada.Command_Line;
with Ada.Streams;
with Ada.Streams.Stream_IO;
with Ada.Text_IO.Text_Streams;
with Ada.Directories;
with GNAT.SHA256;
with Ada.Strings.Unbounded; 

procedure Hash_Compute is
   use Ada.Text_IO;
   use Ada.Command_Line;
   use Ada.Strings.Unbounded;
   use Ada.Streams;

   procedure Print_Usage is
   begin
      Put_Line (Standard_Error, "Usage: hash_compute [options] [file]");
      Put_Line (Standard_Error, "");
      Put_Line (Standard_Error, "Options:");
      Put_Line (Standard_Error, "  --algorithm=ALGO Use specific algorithm (default: sha256)");
      Put_Line (Standard_Error, "  --json           Output result as JSON");
      Put_Line (Standard_Error, "  --verify=HASH    Verify content matches this hash");
      Put_Line (Standard_Error, "  --help           Show this help");
      Put_Line (Standard_Error, "  --version        Show version");
      Put_Line (Standard_Error, "  --describe       Show AI introspection data");
   end Print_Usage;

   procedure Print_Describe is
   begin
      Put_Line ("{");
      Put_Line ("  ""name"": ""hash_compute"",");
      Put_Line ("  ""description"": ""Compute SHA-256 hash of file content"",");
      Put_Line ("  ""version"": ""0.1.0-alpha"",");
      Put_Line ("  ""inputs"": [");
      Put_Line ("    {""name"": ""file"", ""type"": ""argument"", ""description"": ""File to hash (default: stdin)""}");
      Put_Line ("  ],");
      Put_Line ("  ""outputs"": [");
      Put_Line ("    {""type"": ""string"", ""description"": ""Hex encoded hash""}");
      Put_Line ("  ],");
      Put_Line ("  ""options"": [");
      Put_Line ("    {""name"": ""--json"", ""type"": ""boolean""},");
      Put_Line ("    {""name"": ""--verify"", ""type"": ""string""}");
      Put_Line ("  ]");
      Put_Line ("}");
   end Print_Describe;

   Input_File    : Unbounded_String := Null_Unbounded_String;
   Algorithm     : Unbounded_String := To_Unbounded_String ("sha256");
   Verify_Hash   : Unbounded_String := Null_Unbounded_String;
   Output_Json   : Boolean := False;

   function Calculate_Hash (Path : String) return String is
      use Ada.Streams;
      use Ada.Streams.Stream_IO;
      F           : Ada.Streams.Stream_IO.File_Type;
      C           : GNAT.SHA256.Context := GNAT.SHA256.Initial_Context;
      Buffer      : Stream_Element_Array (1 .. 32768);
      Last        : Stream_Element_Offset;
   begin
      --  Handle Stdin
      if Path = "-" then
         declare
            use Ada.Text_IO.Text_Streams;
            Stream : constant Ada.Text_IO.Text_Streams.Stream_Access :=
               Text_Streams.Stream (Standard_Input);
         begin
            loop
               Read (Stream.all, Buffer, Last);
               if Last < Buffer'First then exit; end if;
               GNAT.SHA256.Update (C, Buffer (Buffer'First .. Last));
            end loop;
            return GNAT.SHA256.Digest (C);
         exception
            when Ada.Streams.Stream_IO.End_Error =>
               return GNAT.SHA256.Digest (C); -- End of stream
            when others => return "ERROR";
         end;
      end if;

      --  Handle File
      if not Ada.Directories.Exists (Path) then return ""; end if;
      
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

begin
   for I in 1 .. Argument_Count loop
      declare
         Arg : constant String := Argument (I);
      begin
         if Arg = "--help" then
            Print_Usage; return;
         elsif Arg = "--version" then
            Put_Line ("hash_compute 0.1.0-alpha"); return;
         elsif Arg = "--describe" then
            Print_Describe; return;
         elsif Arg = "--json" then
            Output_Json := True;
         elsif Arg'Length > 12 and then Arg (1 .. 12) = "--algorithm=" then
            Algorithm := To_Unbounded_String (Arg (13 .. Arg'Last));
         elsif Arg'Length > 9 and then Arg (1 .. 9) = "--verify=" then
            Verify_Hash := To_Unbounded_String (Arg (10 .. Arg'Last));
         elsif Arg (1) /= '-' then
            Input_File := To_Unbounded_String (Arg);
         end if;
      end;
   end loop;

   if Input_File = Null_Unbounded_String then
      Input_File := To_Unbounded_String ("-");
   end if;

   declare
      Hash : constant String := Calculate_Hash (To_String (Input_File));
   begin
      if Hash = "" then
         Put_Line (Standard_Error, "Error: File not found or unreadable: " & To_String (Input_File));
         Set_Exit_Status (Failure);
         return;
      end if;

      if Verify_Hash /= Null_Unbounded_String then
         if Hash = To_String (Verify_Hash) then
            if Output_Json then
               Put_Line ("{""verified"": true, ""hash"": """ & Hash & """}");
            end if;
            Set_Exit_Status (Success);
         else
            if Output_Json then
               Put_Line ("{""verified"": false, ""expected"": """ & To_String (Verify_Hash) & """, ""actual"": """ & Hash & """}");
            else
               Put_Line (Standard_Error, "Verification failed.");
               Put_Line (Standard_Error, "Expected: " & To_String (Verify_Hash));
               Put_Line (Standard_Error, "Actual:   " & Hash);
            end if;
            Set_Exit_Status (Failure);
         end if;
      else
         if Output_Json then
            Put_Line ("{""file"": """ & To_String (Input_File) & """, ""algorithm"": """ & To_String (Algorithm) & """, ""hash"": """ & Hash & """}");
         else
            Put_Line (Hash);
         end if;
      end if;
   end;

end Hash_Compute;
