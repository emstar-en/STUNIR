--  type_normalize - Normalize C type declarations
--  Converts C type strings to canonical form
--  Phase 1 Powertool for STUNIR

pragma SPARK_Mode (Off);

with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;
with Ada.Strings.Fixed;
with Ada.Strings.Maps;

with GNAT.Command_Line;

procedure Type_Normalize is
   use Ada.Command_Line;
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;
   use Ada.Strings.Fixed;

   --  Exit codes
   Exit_Success          : constant := 0;
   Exit_Validation_Error : constant := 1;
   Exit_Processing_Error : constant := 2;

   --  Configuration
   Input_Type    : Unbounded_String := Null_Unbounded_String;
   Output_File   : Unbounded_String := Null_Unbounded_String;
   Verbose_Mode  : Boolean := False;
   Show_Version  : Boolean := False;
   Show_Help     : Boolean := False;
   Show_Describe : Boolean := False;
   From_Stdin    : Boolean := False;

   Version : constant String := "1.0.0";

   --  Description output
   Describe_Output : constant String :=
     "{" & ASCII.LF &
     "  ""tool"": ""type_normalize""," & ASCII.LF &
     "  ""version"": ""1.0.0""," & ASCII.LF &
     "  ""description"": ""Normalize C type declarations to canonical form""," & ASCII.LF &
     "  ""inputs"": [{" & ASCII.LF &
     "    ""name"": ""c_type""," & ASCII.LF &
     "    ""type"": ""string""," & ASCII.LF &
     "    ""source"": [""stdin"", ""argument""]," & ASCII.LF &
     "    ""required"": true" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  ""outputs"": [{" & ASCII.LF &
     "    ""name"": ""normalized_type""," & ASCII.LF &
     "    ""type"": ""string""," & ASCII.LF &
     "    ""source"": ""stdout""" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  ""complexity"": ""O(n)""," & ASCII.LF &
     "  ""options"": [" & ASCII.LF &
     "    ""--help"", ""--version"", ""--describe"", ""--output"", ""--verbose""" & ASCII.LF &
     "  ]" & ASCII.LF &
     "}";

   procedure Print_Usage;
   procedure Print_Error (Msg : String);
   procedure Print_Info (Msg : String);
   function Normalize_Type (Type_Str : String) return String;
   procedure Write_Output (Content : String);

   procedure Print_Usage is
   begin
      Put_Line ("type_normalize - Normalize C type declarations");
      Put_Line ("Version: " & Version);
      Put_Line ("");
      Put_Line ("Usage: type_normalize [OPTIONS] [TYPE]");
      Put_Line ("");
      Put_Line ("Options:");
      Put_Line ("  --help, -h        Show this help message");
      Put_Line ("  --version, -v     Show version information");
      Put_Line ("  --describe        Show tool description (JSON)");
      Put_Line ("  --output FILE     Output file (default: stdout)");
      Put_Line ("  --verbose         Verbose output");
      Put_Line ("");
      Put_Line ("Arguments:");
      Put_Line ("  TYPE              C type string to normalize");
      Put_Line ("                    (reads from stdin if not provided)");
      Put_Line ("");
      Put_Line ("Examples:");
      Put_Line ("  type_normalize 'const char*'");
      Put_Line ("  echo 'unsigned int*' | type_normalize");
   end Print_Usage;

   procedure Print_Error (Msg : String) is
   begin
      Put_Line (Standard_Error, "ERROR: " & Msg);
   end Print_Error;

   procedure Print_Info (Msg : String) is
   begin
      if Verbose_Mode then
         Put_Line (Standard_Error, "INFO: " & Msg);
      end if;
   end Print_Info;

   function Trim_Spaces (S : String) return String is
      use Ada.Strings.Maps;
      Result : Unbounded_String := Null_Unbounded_String;
      In_Space : Boolean := False;
   begin
      for I in S'Range loop
         if S (I) = ' ' or else S (I) = ASCII.HT then
            if not In_Space then
               Append (Result, ' ');
               In_Space := True;
            end if;
         else
            Append (Result, S (I));
            In_Space := False;
         end if;
      end loop;
      return To_String (Result);
   end Trim_Spaces;

   function Normalize_Type (Type_Str : String) return String is
      Result : Unbounded_String := Null_Unbounded_String;
      Trimmed : constant String := Trim (Type_Str, Ada.Strings.Both);
      Work : Unbounded_String := To_Unbounded_String (Trimmed);
   begin
      if Trimmed'Length = 0 then
         return "";
      end if;

      --  Step 1: Normalize whitespace
      Work := To_Unbounded_String (Trim_Spaces (Trimmed));

      --  Step 2: Normalize pointer syntax
      declare
         S : constant String := To_String (Work);
         Ptr_Pos : Natural := 0;
      begin
         for I in S'Range loop
            if S (I) = '*' then
               Ptr_Pos := I;
               exit;
            end if;
         end loop;

         if Ptr_Pos > 0 then
            --  Found pointer, normalize spacing
            declare
               Before : constant String := Trim (S (S'First .. Ptr_Pos - 1), Ada.Strings.Both);
               After  : constant String := Trim (S (Ptr_Pos + 1 .. S'Last), Ada.Strings.Both);
            begin
               if After'Length = 0 then
                  Work := To_Unbounded_String (Before & " *");
               else
                  Work := To_Unbounded_String (Before & " *" & After);
               end if;
            end;
         end if;
      end;

      --  Step 3: Normalize const placement
      declare
         S : constant String := To_String (Work);
         Const_Pos : Natural := 0;
      begin
         --  Look for "const" keyword
         for I in S'First .. S'Last - 4 loop
            if (S (I .. I + 4) = "const" or else S (I .. I + 4) = "CONST") and then
               (I = S'First or else S (I - 1) = ' ') and then
               (I + 5 > S'Last or else S (I + 5) = ' ')
            then
               Const_Pos := I;
               exit;
            end if;
         end loop;

         if Const_Pos > 0 then
            --  Move const to beginning if not already there
            if Const_Pos > S'First then
               declare
                  Before : constant String := Trim (S (S'First .. Const_Pos - 1), Ada.Strings.Both);
                  After  : constant String := Trim (S (Const_Pos + 5 .. S'Last), Ada.Strings.Both);
               begin
                  Work := To_Unbounded_String ("const " & Before & After);
               end;
            end if;
         end if;
      end;

      --  Step 4: Normalize unsigned/signed placement
      declare
         S : constant String := To_String (Work);
         Unsigned_Pos : Natural := 0;
         Signed_Pos   : Natural := 0;
      begin
         for I in S'First .. S'Last - 8 loop
            if (S (I .. I + 8) = "unsigned " or else S (I .. I + 8) = "UNSIGNED ") and then
               (I = S'First or else S (I - 1) = ' ')
            then
               Unsigned_Pos := I;
               exit;
            end if;
         end loop;

         for I in S'First .. S'Last - 6 loop
            if (S (I .. I + 6) = "signed " or else S (I .. I + 6) = "SIGNED ") and then
               (I = S'First or else S (I - 1) = ' ')
            then
               Signed_Pos := I;
               exit;
            end if;
         end loop;

         if Unsigned_Pos > 0 and then Unsigned_Pos > S'First then
            declare
               Before : constant String := Trim (S (S'First .. Unsigned_Pos - 1), Ada.Strings.Both);
               After  : constant String := Trim (S (Unsigned_Pos + 9 .. S'Last), Ada.Strings.Both);
            begin
               Work := To_Unbounded_String ("unsigned " & Before & " " & After);
            end;
         elsif Signed_Pos > 0 and then Signed_Pos > S'First then
            declare
               Before : constant String := Trim (S (S'First .. Signed_Pos - 1), Ada.Strings.Both);
               After  : constant String := Trim (S (Signed_Pos + 7 .. S'Last), Ada.Strings.Both);
            begin
               Work := To_Unbounded_String ("signed " & Before & " " & After);
            end;
         end if;
      end;

      --  Step 5: Normalize struct/enum/union keywords
      declare
         S : constant String := To_String (Work);
      begin
         for I in S'Range loop
            if I + 6 <= S'Last and then
               (S (I .. I + 6) = "struct " or else S (I .. I + 6) = "STRUCT ") and then
               (I = S'First or else S (I - 1) = ' ')
            then
               declare
                  Before : constant String := Trim (S (S'First .. I - 1), Ada.Strings.Both);
                  After  : constant String := Trim (S (I + 7 .. S'Last), Ada.Strings.Both);
               begin
                  Work := To_Unbounded_String (Before & "struct " & After);
                  exit;
               end;
            elsif I + 4 <= S'Last and then
               (S (I .. I + 4) = "enum " or else S (I .. I + 4) = "ENUM ") and then
               (I = S'First or else S (I - 1) = ' ')
            then
               declare
                  Before : constant String := Trim (S (S'First .. I - 1), Ada.Strings.Both);
                  After  : constant String := Trim (S (I + 5 .. S'Last), Ada.Strings.Both);
               begin
                  Work := To_Unbounded_String (Before & "enum " & After);
                  exit;
               end;
            elsif I + 5 <= S'Last and then
               (S (I .. I + 5) = "union " or else S (I .. I + 5) = "UNION ") and then
               (I = S'First or else S (I - 1) = ' ')
            then
               declare
                  Before : constant String := Trim (S (S'First .. I - 1), Ada.Strings.Both);
                  After  : constant String := Trim (S (I + 6 .. S'Last), Ada.Strings.Both);
               begin
                  Work := To_Unbounded_String (Before & "union " & After);
                  exit;
               end;
            end if;
         end loop;
      end;

      --  Final cleanup: single spaces only
      return Trim_Spaces (To_String (Work));
   end Normalize_Type;

   procedure Write_Output (Content : String) is
   begin
      if Output_File = Null_Unbounded_String then
         Put_Line (Content);
      else
         declare
            File : File_Type;
         begin
            Create (File, Out_File, To_String (Output_File));
            Put (File, Content);
            Close (File);
         exception
            when others =>
               Print_Error ("Cannot write: " & To_String (Output_File));
         end;
      end if;
   end Write_Output;

   Config : GNAT.Command_Line.Command_Line_Configuration;

begin
   GNAT.Command_Line.Define_Switch (Config, Show_Help'Access, "-h", "--help");
   GNAT.Command_Line.Define_Switch (Config, Show_Version'Access, "-v", "--version");
   GNAT.Command_Line.Define_Switch (Config, Show_Describe'Access, "", "--describe");
   GNAT.Command_Line.Define_Switch (Config, Verbose_Mode'Access, "", "--verbose");
   GNAT.Command_Line.Define_Switch (Config, Output_File'Access, "-o:", "--output=");

   begin
      GNAT.Command_Line.Getopt (Config);
   exception
      when others =>
         Print_Error ("Invalid arguments");
         Set_Exit_Status (Exit_Processing_Error);
         return;
   end;

   if Show_Help then
      Print_Usage;
      Set_Exit_Status (Exit_Success);
      return;
   end if;

   if Show_Version then
      Put_Line (Version);
      Set_Exit_Status (Exit_Success);
      return;
   end if;

   if Show_Describe then
      Put_Line (Describe_Output);
      Set_Exit_Status (Exit_Success);
      return;
   end if;

   --  Get type from argument or stdin
   declare
      Arg : String := GNAT.Command_Line.Get_Argument;
   begin
      if Arg'Length > 0 then
         Input_Type := To_Unbounded_String (Arg);
      else
         From_Stdin := True;
      end if;
   end;

   if From_Stdin then
      declare
         Line : String (1 .. 1024);
         Last : Natural;
      begin
         Get_Line (Line, Last);
         Input_Type := To_Unbounded_String (Line (1 .. Last));
      exception
         when others =>
            Print_Error ("No input provided");
            Set_Exit_Status (Exit_Validation_Error);
            return;
      end;
   end if;

   if Input_Type = Null_Unbounded_String then
      Print_Error ("No type string provided");
      Set_Exit_Status (Exit_Validation_Error);
      return;
   end if;

   declare
      Result : constant String := Normalize_Type (To_String (Input_Type));
   begin
      Print_Info ("Input:  '" & To_String (Input_Type) & "'");
      Print_Info ("Output: '" & Result & "'");
      Write_Output (Result);
      Set_Exit_Status (Exit_Success);
   end;

exception
   when others =>
      Print_Error ("Unexpected error");
      Set_Exit_Status (Exit_Processing_Error);
end Type_Normalize;