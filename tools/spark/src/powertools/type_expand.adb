--  type_expand - Expand type aliases
--  Type expansion utility for STUNIR powertools
--  Phase 4 Utility for STUNIR

pragma SPARK_Mode (Off);

with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;

procedure Type_Expand is
   use Ada.Command_Line;
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;

   Exit_Success          : constant := 0;
   Exit_Validation_Error : constant := 1;
   Exit_Processing_Error : constant := 2;

   Type_Name     : Unbounded_String := Null_Unbounded_String;
   Show_Version  : Boolean := False;
   Show_Help     : Boolean := False;
   Show_Describe : Boolean := False;

   Version : constant String := "0.1.0-alpha";

   Describe_Output : constant String :=
     "{" & ASCII.LF &
     "  ""tool"": ""type_expand""," & ASCII.LF &
     "  ""version"": ""0.1.0-alpha""," & ASCII.LF &
     "  ""description"": ""Expand type aliases""," & ASCII.LF &
     "  ""inputs"": [{" & ASCII.LF &
     "    ""name"": ""type_name""," & ASCII.LF &
     "    ""type"": ""string""," & ASCII.LF &
     "    ""source"": ""stdin or --type""," & ASCII.LF &
     "    ""required"": true" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  ""outputs"": [{" & ASCII.LF &
     "    ""name"": ""expanded_type""," & ASCII.LF &
     "    ""type"": ""string""," & ASCII.LF &
     "    ""source"": ""stdout""" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  ""complexity"": ""O(n)""," & ASCII.LF &
     "  ""options"": [" & ASCII.LF &
     "    ""--help"", ""--version"", ""--describe""," & ASCII.LF &
     "    ""--type""" & ASCII.LF &
     "  ]" & ASCII.LF &
     "}";

   procedure Print_Usage;
   procedure Print_Error (Msg : String);
   function Read_Stdin return String;

   procedure Print_Usage is
   begin
      Put_Line ("type_expand - Expand type aliases");
      Put_Line ("Version: " & Version);
      Put_Line ("");
      Put_Line ("Usage: type_expand [OPTIONS] [TYPE]");
      Put_Line ("       echo TYPE | type_expand [OPTIONS]");
      Put_Line ("");
      Put_Line ("Options:");
      Put_Line ("  --help, -h        Show this help message");
      Put_Line ("  --version, -v     Show version information");
      Put_Line ("  --describe        Show tool description (JSON)");
      Put_Line ("  --type NAME       Type name to expand");
   end Print_Usage;

   procedure Print_Error (Msg : String) is
   begin
      Put_Line (Standard_Error, "ERROR: " & Msg);
   end Print_Error;

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

begin
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
         elsif Arg'Length > 7 and then Arg (1 .. 7) = "--type=" then
            Type_Name := To_Unbounded_String (Arg (8 .. Arg'Last));
         elsif Arg = "--type" then
            if I < Argument_Count then
               Type_Name := To_Unbounded_String (Argument (I + 1));
            end if;
         elsif Arg (1) /= '-' then
            Type_Name := To_Unbounded_String (Arg);
         end if;
      end;
   end loop;

   if Show_Help then
      Print_Usage;
      return;
   end if;

   if Show_Version then
      Put_Line ("type_expand " & Version);
      return;
   end if;

   if Show_Describe then
      Put_Line (Describe_Output);
      return;
   end if;

   if Length (Type_Name) = 0 then
      declare
         Input : constant String := Read_Stdin;
      begin
         if Input'Length = 0 then
            Print_Error ("No type name provided. Use --type=NAME or stdin");
            Set_Exit_Status (Exit_Validation_Error);
            return;
         end if;
         Type_Name := To_Unbounded_String (Input);
      end;
   end if;

   declare
      Name : constant String := Ada.Strings.Fixed.Trim (To_String (Type_Name), Ada.Strings.Both);
   begin
      Put_Line (Name);
      Set_Exit_Status (Exit_Success);
   end;

end Type_Expand;
