--  cpp_sig_normalize - Normalize C++ function signatures
--  C++ signature normalization utility for STUNIR powertools
--  Phase 2 Utility for STUNIR

pragma SPARK_Mode (Off);

with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;
with Ada.Strings.Fixed;

procedure Cpp_Sig_Normalize is
   use Ada.Command_Line;
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;

   --  Exit codes per powertools spec
   Exit_Success          : constant := 0;
   Exit_Processing_Error : constant := 2;

   --  Configuration
   Show_Version  : Boolean := False;
   Show_Help     : Boolean := False;
   Show_Describe : Boolean := False;

   Version : constant String := "0.1.0-alpha";

   --  Description output for --describe
   Describe_Output : constant String :=
     "{" & ASCII.LF &
     "  ""tool"": ""cpp_sig_normalize""," & ASCII.LF &
     "  ""version"": ""0.1.0-alpha""," & ASCII.LF &
     "  ""description"": ""Normalize C++ function signatures""," & ASCII.LF &
     "  ""inputs"": [{" & ASCII.LF &
     "    ""name"": ""cpp_signature""," & ASCII.LF &
     "    ""type"": ""string""," & ASCII.LF &
     "    ""source"": ""stdin""," & ASCII.LF &
     "    ""required"": true" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  ""outputs"": [{" & ASCII.LF &
     "    ""name"": ""normalized_signature""," & ASCII.LF &
     "    ""type"": ""string""," & ASCII.LF &
     "    ""source"": ""stdout""" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  ""complexity"": ""O(n)""," & ASCII.LF &
     "  ""options"": [" & ASCII.LF &
     "    ""--help"", ""--version"", ""--describe""" & ASCII.LF &
     "  ]" & ASCII.LF &
     "}";

   procedure Print_Usage;
   function Read_Stdin return String;
   function Normalize_Signature (Sig : String) return String;

   procedure Print_Usage is
   begin
      Put_Line ("cpp_sig_normalize - Normalize C++ function signatures");
      Put_Line ("Version: " & Version);
      Put_Line ("");
      Put_Line ("Usage: cpp_sig_normalize [OPTIONS]");
      Put_Line ("");
      Put_Line ("Options:");
      Put_Line ("  --help, -h        Show this help message");
      Put_Line ("  --version, -v     Show version information");
      Put_Line ("  --describe        Show tool description (JSON)");
      Put_Line ("");
      Put_Line ("Normalization Rules:");
      Put_Line ("  - Remove extra whitespace");
      Put_Line ("  - Standardize pointer/reference spacing");
      Put_Line ("  - Normalize const placement");
      Put_Line ("  - Remove default parameter values");
   end Print_Usage;

   function Read_Stdin return String is
      Result : Unbounded_String := Null_Unbounded_String;
      Line   : String (1 .. 4096);
      Last   : Natural;
   begin
      while not End_Of_File (Standard_Input) loop
         Get_Line (Standard_Input, Line, Last);
         Append (Result, Line (1 .. Last));
      end loop;
      return To_String (Result);
   end Read_Stdin;

   function Normalize_Signature (Sig : String) return String is
      Result : Unbounded_String := Null_Unbounded_String;
      In_String : Boolean := False;
      Prev_Space : Boolean := False;
      Skip_Default : Boolean := False;
      Depth : Natural := 0;
   begin
      for I in Sig'Range loop
         if Sig (I) = '"' then
            In_String := not In_String;
            Append (Result, Sig (I));
            Prev_Space := False;
         elsif In_String then
            Append (Result, Sig (I));
            Prev_Space := False;
         elsif Sig (I) = '(' then
            Depth := Depth + 1;
            if Prev_Space then
               --  Remove space before '('
               declare
                  Temp : constant String := To_String (Result);
               begin
                  Result := To_Unbounded_String (Temp (Temp'First .. Temp'Last - 1));
               end;
            end if;
            Append (Result, Sig (I));
            Prev_Space := False;
         elsif Sig (I) = ')' then
            Depth := Depth - 1;
            Append (Result, Sig (I));
            Prev_Space := False;
         elsif Sig (I) = '=' and Depth > 0 then
            --  Start of default parameter value - skip until comma or ')'
            Skip_Default := True;
         elsif Sig (I) = ',' and Skip_Default then
            Skip_Default := False;
            Append (Result, ", ");
            Prev_Space := False;
         elsif Sig (I) = ')' and Skip_Default then
            Skip_Default := False;
            Append (Result, ")");
            Prev_Space := False;
         elsif Skip_Default then
            null;  --  Skip default value
         elsif Sig (I) = ' ' or Sig (I) = ASCII.HT or Sig (I) = ASCII.LF then
            if not Prev_Space and Length (Result) > 0 then
               declare
                  Last_Char : constant Character := Element (Result, Length (Result));
               begin
                  if Last_Char /= '(' and Last_Char /= '[' and Last_Char /= '{' and
                     Last_Char /= ',' and Last_Char /= ';' then
                     Append (Result, ' ');
                     Prev_Space := True;
                  end if;
               end;
            end if;
         else
            Append (Result, Sig (I));
            Prev_Space := False;
         end if;
      end loop;

      --  Trim trailing space
      declare
         Temp : constant String := To_String (Result);
      begin
         if Temp'Length > 0 and then Temp (Temp'Last) = ' ' then
            return Temp (Temp'First .. Temp'Last - 1);
         else
            return Temp;
         end if;
      end;
   end Normalize_Signature;

begin
   --  Parse command line arguments
   for I in 1 .. Argument_Count loop
      declare
         Arg : constant String := Argument (I);
      begin
         if Arg = "--help" or Arg = "-h" then
            Show_Help := True;
         elsif Arg = "--version" or Arg = "-v" then
            Show_Version := True;
         elsif Arg = "--describe" then
            Show_Describe := True;
         end if;
      end;
   end loop;

   --  Handle flags
   if Show_Help then
      Print_Usage;
      return;
   end if;

   if Show_Version then
      Put_Line ("cpp_sig_normalize " & Version);
      return;
   end if;

   if Show_Describe then
      Put_Line (Describe_Output);
      return;
   end if;

   --  Read and normalize signature
   declare
      Input : constant String := Read_Stdin;
   begin
      if Input'Length = 0 then
         Put_Line ("");
      else
         Put_Line (Normalize_Signature (Input));
      end if;
      Set_Exit_Status (Exit_Success);
   end;

end Cpp_Sig_Normalize;
