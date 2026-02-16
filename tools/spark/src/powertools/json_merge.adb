with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;
with Ada.IO_Exceptions;

procedure JSON_Merge is
   use Ada.Command_Line;
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;

   Input_Files : array (1 .. 32) of Unbounded_String;
   Input_Count : Natural := 0;
   Output_Json : Boolean := False;
   Show_Help   : Boolean := False;
   Show_Describe : Boolean := False;

   procedure Print_Usage is
   begin
      Put_Line (Standard_Error, "Usage: json_merge [options] [file1 file2 ...]");
      Put_Line (Standard_Error, "Options:");
      Put_Line (Standard_Error, "  --json           Output status as JSON (default: raw)" );
      Put_Line (Standard_Error, "  --describe       Show tool description" );
      Put_Line (Standard_Error, "  --help           Show this help" );
   end Print_Usage;

   procedure Print_Describe is
   begin
      Put_Line ("{");
      Put_Line ("  ""name"": ""json_merge"",");
      Put_Line ("  ""description"": ""Merge multiple JSON objects or arrays"",");
      Put_Line ("  ""version"": ""1.0.0"",");
      Put_Line ("  ""inputs"": [{""name"": ""json_inputs"", ""type"": ""json"", ""source"": [""stdin"", ""file""]}]");
      Put_Line ("}");
   end Print_Describe;

   function Read_File (Path : String) return String is
      File : File_Type;
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
   end Read_File;

   function Read_Stdin return String is
      Result : Unbounded_String := Null_Unbounded_String;
      Line   : String (1 .. 4096);
      Last   : Natural;
   begin
      while not End_Of_File loop
         Get_Line (Line, Last);
         Append (Result, Line (1 .. Last));
         Append (Result, ASCII.LF);
      end loop;
      return To_String (Result);
   end Read_Stdin;

   function Trim (S : String) return String is
      First : Natural := S'First;
      Last  : Natural := S'Last;
   begin
      while First <= S'Last and then S (First) in ' ' | ASCII.HT | ASCII.LF | ASCII.CR loop
         First := First + 1;
      end loop;
      while Last >= S'First and then S (Last) in ' ' | ASCII.HT | ASCII.LF | ASCII.CR loop
         Last := Last - 1;
      end loop;
      if First > Last then
         return "";
      end if;
      return S (First .. Last);
   end Trim;

   function Strip_Outer (S : String) return String is
      T : constant String := Trim (S);
   begin
      if T'Length >= 2 then
         if (T (T'First) = '{' and then T (T'Last) = '}') or else
            (T (T'First) = '[' and then T (T'Last) = ']') then
            return T (T'First + 1 .. T'Last - 1);
         end if;
      end if;
      return T;
   end Strip_Outer;

   function Is_Object (S : String) return Boolean is
      T : constant String := Trim (S);
   begin
      return T'Length >= 2 and then T (T'First) = '{' and then T (T'Last) = '}';
   end Is_Object;

   function Is_Array (S : String) return Boolean is
      T : constant String := Trim (S);
   begin
      return T'Length >= 2 and then T (T'First) = '[' and then T (T'Last) = ']';
   end Is_Array;

   function Merge_Inputs return String is
      Merged : Unbounded_String := Null_Unbounded_String;
      Use_Object : Boolean := True;
   begin
      if Input_Count = 0 then
         declare
            S : constant String := Read_Stdin;
         begin
            if Is_Array (S) then
               Use_Object := False;
            end if;
            Append (Merged, Strip_Outer (S));
         end;
      else
         for I in 1 .. Input_Count loop
            declare
               S : constant String := Read_File (To_String (Input_Files (I)));
               T : constant String := Trim (S);
            begin
               if S = "" then
                  return "";
               end if;
               if I = 1 then
                  if Is_Array (T) then
                     Use_Object := False;
                  elsif Is_Object (T) then
                     Use_Object := True;
                  else
                     return "";
                  end if;
                  Append (Merged, Strip_Outer (T));
               else
                  if (Use_Object and then not Is_Object (T)) or else
                     ((not Use_Object) and then not Is_Array (T)) then
                     return "";
                  end if;
                  if Length (Merged) > 0 then
                     Append (Merged, ",");
                  end if;
                  Append (Merged, Strip_Outer (T));
               end if;
            end;
         end loop;
      end if;

      if Use_Object then
         return "{" & To_String (Merged) & "}";
      else
         return "[" & To_String (Merged) & "]";
      end if;
   end Merge_Inputs;

begin
   for I in 1 .. Argument_Count loop
      declare
         Arg : constant String := Argument (I);
      begin
         if Arg = "--help" then
            Show_Help := True;
         elsif Arg = "--describe" then
            Show_Describe := True;
         elsif Arg = "--json" then
            Output_Json := True;
         elsif Arg (1) /= '-' then
            if Input_Count < Input_Files'Last then
               Input_Count := Input_Count + 1;
               Input_Files (Input_Count) := To_Unbounded_String (Arg);
            end if;
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

   declare
      Merged : constant String := Merge_Inputs;
   begin
      if Merged = "" then
         if Output_Json then
            Put_Line ("{""status"": ""error""}");
         else
            Put_Line (Standard_Error, "Error: invalid inputs");
         end if;
         Set_Exit_Status (Failure);
         return;
      end if;

      Put_Line (Merged);
      Set_Exit_Status (Success);
   end;
end JSON_Merge;
