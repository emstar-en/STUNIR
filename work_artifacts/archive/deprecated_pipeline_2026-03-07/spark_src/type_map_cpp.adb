--  type_map_cpp - Map STUNIR types to C++ types
--  Type mapping utility for STUNIR powertools
--  Phase 2 Utility for STUNIR

pragma SPARK_Mode (Off);

with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;
with Ada.Strings.Fixed;
with Ada.Characters.Handling;

procedure Type_Map_Cpp is
   use Ada.Command_Line;
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;
   use Ada.Characters.Handling;

   --  Exit codes per powertools spec
   Exit_Success          : constant := 0;
   Exit_Validation_Error : constant := 1;

   --  Configuration
   Type_Name     : Unbounded_String := Null_Unbounded_String;
   Show_Version  : Boolean := False;
   Show_Help     : Boolean := False;
   Show_Describe : Boolean := False;

   Version : constant String := "0.1.0-alpha";

   --  Description output for --describe
   Describe_Output : constant String :=
     "{" & ASCII.LF &
     "  ""tool"": ""type_map_cpp""," & ASCII.LF &
     "  ""version"": ""0.1.0-alpha""," & ASCII.LF &
     "  ""description"": ""Map STUNIR type names to C++ types""," & ASCII.LF &
     "  ""inputs"": [{" & ASCII.LF &
     "    ""name"": ""stunir_type""," & ASCII.LF &
     "    ""type"": ""string""," & ASCII.LF &
     "    ""source"": ""stdin or --type""," & ASCII.LF &
     "    ""required"": true" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  ""outputs"": [{" & ASCII.LF &
     "    ""name"": ""cpp_type""," & ASCII.LF &
     "    ""type"": ""string""," & ASCII.LF &
     "    ""source"": ""stdout""" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  ""complexity"": ""O(1)""," & ASCII.LF &
     "  ""options"": [" & ASCII.LF &
     "    ""--help"", ""--version"", ""--describe""," & ASCII.LF &
     "    ""--type""" & ASCII.LF &
     "  ]" & ASCII.LF &
     "}";

   procedure Print_Usage;
   procedure Print_Error (Msg : String);
   function Read_Stdin return String;
   function Map_Type (STUNIR_Type : String) return String;

   procedure Print_Usage is
   begin
      Put_Line ("type_map_cpp - Map STUNIR types to C++ types");
      Put_Line ("Version: " & Version);
      Put_Line ("");
      Put_Line ("Usage: type_map_cpp [OPTIONS] [TYPE]");
      Put_Line ("       echo TYPE | type_map_cpp [OPTIONS]");
      Put_Line ("");
      Put_Line ("Options:");
      Put_Line ("  --help, -h        Show this help message");
      Put_Line ("  --version, -v     Show version information");
      Put_Line ("  --describe        Show tool description (JSON)");
      Put_Line ("  --type TYPE       Type to map (alternative to stdin)");
      Put_Line ("");
      Put_Line ("Type Mapping Rules:");
      Put_Line ("  i8   → int8_t");
      Put_Line ("  i16  → int16_t");
      Put_Line ("  i32  → int32_t");
      Put_Line ("  i64  → int64_t");
      Put_Line ("  u8   → uint8_t");
      Put_Line ("  u16  → uint16_t");
      Put_Line ("  u32  → uint32_t");
      Put_Line ("  u64  → uint64_t");
      Put_Line ("  f32  → float");
      Put_Line ("  f64  → double");
      Put_Line ("  bool → bool");
      Put_Line ("  str  → std::string");
      Put_Line ("  void → void");
      Put_Line ("");
      Put_Line ("Exit Codes:");
      Put_Line ("  0                 Success");
      Put_Line ("  1                 Unknown type");
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

   function Map_Type (STUNIR_Type : String) return String is
      Normalized : constant String := To_Lower (STUNIR_Type);
   begin
      if Normalized = "i8" then
         return "int8_t";
      elsif Normalized = "i16" then
         return "int16_t";
      elsif Normalized = "i32" then
         return "int32_t";
      elsif Normalized = "i64" then
         return "int64_t";
      elsif Normalized = "u8" then
         return "uint8_t";
      elsif Normalized = "u16" then
         return "uint16_t";
      elsif Normalized = "u32" then
         return "uint32_t";
      elsif Normalized = "u64" then
         return "uint64_t";
      elsif Normalized = "f32" then
         return "float";
      elsif Normalized = "f64" then
         return "double";
      elsif Normalized = "bool" then
         return "bool";
      elsif Normalized = "str" then
         return "std::string";
      elsif Normalized = "void" then
         return "void";
      else
         return "";  --  Unknown type
      end if;
   end Map_Type;

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
         elsif Arg'Length > 7 and then Arg (1 .. 7) = "--type=" then
            Type_Name := To_Unbounded_String (Arg (8 .. Arg'Last));
         elsif Arg = "--type" then
            if I < Argument_Count then
               Type_Name := To_Unbounded_String (Argument (I + 1));
            end if;
         elsif Arg (1) /= '-' then
            --  Positional argument
            Type_Name := To_Unbounded_String (Arg);
         end if;
      end;
   end loop;

   --  Handle flags
   if Show_Help then
      Print_Usage;
      return;
   end if;

   if Show_Version then
      Put_Line ("type_map_cpp " & Version);
      return;
   end if;

   if Show_Describe then
      Put_Line (Describe_Output);
      return;
   end if;

   --  Get type name
   if Length (Type_Name) = 0 then
      declare
         Input : constant String := Read_Stdin;
      begin
         if Input'Length = 0 then
            Print_Error ("No type name provided. Use --type=TYPE or stdin");
            Set_Exit_Status (Exit_Validation_Error);
            return;
         end if;
         Type_Name := To_Unbounded_String (Input);
      end;
   end if;

   --  Map type
   declare
      STUNIR_Type : constant String := Ada.Strings.Fixed.Trim (To_String (Type_Name), Ada.Strings.Both);
      Cpp_Type : constant String := Map_Type (STUNIR_Type);
   begin
      if Cpp_Type'Length = 0 then
         Print_Error ("Unknown type: " & STUNIR_Type);
         Set_Exit_Status (Exit_Validation_Error);
      else
         Put_Line (Cpp_Type);
         Set_Exit_Status (Exit_Success);
      end if;
   end;

end Type_Map_Cpp;
