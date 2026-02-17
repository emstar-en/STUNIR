with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;
with Ada.IO_Exceptions;
with Ada.Streams;
with GNAT.SHA256;

procedure Receipt_Generate is
   use Ada.Command_Line;
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;

   Spec_File     : Unbounded_String := Null_Unbounded_String;
   Manifest_File : Unbounded_String := Null_Unbounded_String;
   Output_File   : Unbounded_String := Null_Unbounded_String;
   Show_Help     : Boolean := False;
   Show_Describe : Boolean := False;

   procedure Print_Usage is
   begin
      Put_Line (Standard_Error, "Usage: receipt_generate --spec FILE --manifest FILE [--output=FILE]");
      Put_Line (Standard_Error, "Options:");
      Put_Line (Standard_Error, "  --spec FILE       Path to spec JSON");
      Put_Line (Standard_Error, "  --manifest FILE   Path to manifest JSON");
      Put_Line (Standard_Error, "  --output=FILE     Write output to file (default: stdout)");
      Put_Line (Standard_Error, "  --describe        Show tool description");
      Put_Line (Standard_Error, "  --help            Show this help");
   end Print_Usage;

   procedure Print_Describe is
   begin
      Put_Line ("{");
      Put_Line ("  ""name"": ""receipt_generate"",");
      Put_Line ("  ""description"": ""Create verification receipt with hashes"",");
      Put_Line ("  ""version"": ""0.1.0-alpha"",");
      Put_Line ("  ""inputs"": [{""name"": ""spec"", ""type"": ""json"", ""source"": ""file""}, {""name"": ""manifest"", ""type"": ""json"", ""source"": ""file""}],");
      Put_Line ("  ""outputs"": [{""name"": ""receipt"", ""type"": ""json"", ""source"": ""stdout""}]");
      Put_Line ("}");
   end Print_Describe;

   function Read_All (Path : String) return String is
      File   : File_Type;
      Result : Unbounded_String := Null_Unbounded_String;
      Line   : String (1 .. 4096);
      Last   : Natural;
   begin
      Open (File, In_File, Path);
      while not End_Of_File (File) loop
         Get_Line (File, Line, Last);
         Append (Result, Line (1 .. Last));
         Append (Result, ASCII.LF);
      end loop;
      Close (File);
      return To_String (Result);
   exception
      when Ada.IO_Exceptions.Name_Error =>
         return "";
      when others =>
         return "";
   end Read_All;

   function Hash_Content (Content : String) return String is
      C : GNAT.SHA256.Context := GNAT.SHA256.Initial_Context;
      use Ada.Streams;
      Buffer : Stream_Element_Array (1 .. 32768);
      Last   : Stream_Element_Offset := 0;
   begin
      for I in Content'Range loop
         if Last = Buffer'Last then
            GNAT.SHA256.Update (C, Buffer (Buffer'First .. Last));
            Last := 0;
         end if;
         Last := Last + 1;
         Buffer (Last) := Stream_Element (Character'Pos (Content (I)));
      end loop;
      if Last >= Buffer'First then
         GNAT.SHA256.Update (C, Buffer (Buffer'First .. Last));
      end if;
      return GNAT.SHA256.Digest (C);
   end Hash_Content;

   procedure Write_All (Path : String; Content : String) is
      File : File_Type;
   begin
      if Path = "" then
         Put_Line (Content);
      else
         Create (File, Out_File, Path);
         Put (File, Content);
         Close (File);
      end if;
   end Write_All;

begin
   for I in 1 .. Argument_Count loop
      declare
         Arg : constant String := Argument (I);
      begin
         if Arg = "--help" then
            Show_Help := True;
         elsif Arg = "--describe" then
            Show_Describe := True;
         elsif Arg = "--spec" and I < Argument_Count then
            Spec_File := To_Unbounded_String (Argument (I + 1));
         elsif Arg = "--manifest" and I < Argument_Count then
            Manifest_File := To_Unbounded_String (Argument (I + 1));
         elsif Arg'Length > 9 and then Arg (1 .. 9) = "--output=" then
            Output_File := To_Unbounded_String (Arg (10 .. Arg'Last));
         end if;
      end;
   end loop;

   if Show_Help then
      Print_Usage;
      Set_Exit_Status (Success);
      return;
   end if;

   if Show_Describe then
      Print_Describe;
      Set_Exit_Status (Success);
      return;
   end if;

   if Spec_File = Null_Unbounded_String or else Manifest_File = Null_Unbounded_String then
      Print_Usage;
      Set_Exit_Status (Failure);
      return;
   end if;

   declare
      Spec_Content     : constant String := Read_All (To_String (Spec_File));
      Manifest_Content : constant String := Read_All (To_String (Manifest_File));
   begin
      if Spec_Content = "" or else Manifest_Content = "" then
         Put_Line (Standard_Error, "Error: missing or unreadable input files");
         Set_Exit_Status (Failure);
         return;
      end if;

      declare
         Spec_Hash     : constant String := Hash_Content (Spec_Content);
         Manifest_Hash : constant String := Hash_Content (Manifest_Content);
         Receipt : constant String :=
           "{" &
           """spec_hash"": """ & Spec_Hash & """," &
           """source_index"": """ & Manifest_Hash & """," &
           """links"":[]" &
           "}";
      begin
         Write_All (To_String (Output_File), Receipt);
         Set_Exit_Status (Success);
      end;
   end;
end Receipt_Generate;
