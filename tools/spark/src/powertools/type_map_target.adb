--  type_map_target - Map STUNIR types to target language types
--  Maps internal type names to language-specific type names
--  Phase 1 Powertool for STUNIR

pragma SPARK_Mode (Off);

with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;
with Ada.Characters.Handling;
with GNAT.Command_Line;
with GNAT.Strings;

procedure Type_Map_Target is
   use Ada.Command_Line;
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;
   use Ada.Characters.Handling;
   use GNAT.Strings;

   --  Exit codes
   Exit_Success : constant := 0;
   Exit_Error   : constant := 1;

   --  Configuration
   Target_Lang   : aliased GNAT.Strings.String_Access := null;
   Show_Version  : aliased Boolean := False;
   Show_Help     : aliased Boolean := False;
   Show_Describe : aliased Boolean := False;

   Version : constant String := "0.1.0-alpha";

   Describe_Output : constant String :=
     "{" & ASCII.LF &
     "  ""tool"": ""type_map_target""," & ASCII.LF &
     "  ""version"": ""0.1.0-alpha""," & ASCII.LF &
     "  ""description"": ""Map STUNIR types to target language types""," & ASCII.LF &
     "  ""inputs"": [{" & ASCII.LF &
     "    ""name"": ""type_name""," & ASCII.LF &
     "    ""type"": ""string""," & ASCII.LF &
     "    ""source"": [""stdin""]," & ASCII.LF &
     "    ""required"": true" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  ""outputs"": [{" & ASCII.LF &
     "    ""name"": ""mapped_type""," & ASCII.LF &
     "    ""type"": ""string""," & ASCII.LF &
     "    ""source"": ""stdout""" & ASCII.LF &
     "  }]" & ASCII.LF &
     "}";

   procedure Print_Usage is
   begin
      Put_Line ("type_map_target - Map STUNIR types to target language");
      Put_Line ("Version: " & Version);
      Put_Line ("");
      Put_Line ("Usage: type_map_target [OPTIONS] --target LANG < type.txt");
      Put_Line ("");
      Put_Line ("Options:");
      Put_Line ("  --help, -h        Show this help message");
      Put_Line ("  --version, -v     Show version information");
      Put_Line ("  --describe        Show tool description (JSON)");
      Put_Line ("  --target LANG     Target language: rust|c|python");
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
      while not End_Of_File loop
         Get_Line (Line, Last);
         Append (Result, Line (1 .. Last));
      end loop;
      return To_String (Result);
   end Read_Stdin;

   function Map_Rust_Type (Type_Name : String) return String is
      Lower_Type : constant String := To_Lower (Type_Name);
   begin
      if Lower_Type = "i8" then return "i8";
      elsif Lower_Type = "i16" then return "i16";
      elsif Lower_Type = "i32" then return "i32";
      elsif Lower_Type = "i64" then return "i64";
      elsif Lower_Type = "u8" then return "u8";
      elsif Lower_Type = "u16" then return "u16";
      elsif Lower_Type = "u32" then return "u32";
      elsif Lower_Type = "u64" then return "u64";
      elsif Lower_Type = "f32" then return "f32";
      elsif Lower_Type = "f64" then return "f64";
      elsif Lower_Type = "bool" then return "bool";
      elsif Lower_Type = "str" then return "&str";
      else return Type_Name;
      end if;
   end Map_Rust_Type;

   function Map_C_Type (Type_Name : String) return String is
      Lower_Type : constant String := To_Lower (Type_Name);
   begin
      if Lower_Type = "i8" then return "int8_t";
      elsif Lower_Type = "i16" then return "int16_t";
      elsif Lower_Type = "i32" then return "int32_t";
      elsif Lower_Type = "i64" then return "int64_t";
      elsif Lower_Type = "u8" then return "uint8_t";
      elsif Lower_Type = "u16" then return "uint16_t";
      elsif Lower_Type = "u32" then return "uint32_t";
      elsif Lower_Type = "u64" then return "uint64_t";
      elsif Lower_Type = "f32" then return "float";
      elsif Lower_Type = "f64" then return "double";
      elsif Lower_Type = "bool" then return "bool";
      elsif Lower_Type = "str" then return "const char*";
      else return Type_Name;
      end if;
   end Map_C_Type;

   function Map_Python_Type (Type_Name : String) return String is
      Lower_Type : constant String := To_Lower (Type_Name);
   begin
      if Lower_Type in "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64" then
         return "int";
      elsif Lower_Type = "f32" or Lower_Type = "f64" then
         return "float";
      elsif Lower_Type = "bool" then
         return "bool";
      elsif Lower_Type = "str" then
         return "str";
      else
         return Type_Name;
      end if;
   end Map_Python_Type;

   Config : GNAT.Command_Line.Command_Line_Configuration;

begin
   GNAT.Command_Line.Define_Switch (Config, Show_Help'Access, "-h", "--help");
   GNAT.Command_Line.Define_Switch (Config, Show_Version'Access, "-v", "--version");
   GNAT.Command_Line.Define_Switch (Config, Show_Describe'Access, "", "--describe");
   GNAT.Command_Line.Define_Switch (Config, Target_Lang'Access, "-t:", "--target=");

   begin
      GNAT.Command_Line.Getopt (Config);
   exception
      when others =>
         Print_Error ("Invalid arguments");
         Set_Exit_Status (Exit_Error);
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

   if Target_Lang = null then
      Print_Error ("Target language required (--target)");
      Set_Exit_Status (Exit_Error);
      return;
   end if;

   declare
      Input_Type : constant String := Read_Stdin;
      Lang : constant String := To_Lower (Target_Lang.all);
   begin
      if Input_Type'Length = 0 then
         Print_Error ("Empty input");
         Set_Exit_Status (Exit_Error);
         return;
      end if;

      declare
         Mapped : constant String :=
           (if Lang = "rust" then
               Map_Rust_Type (Input_Type)
            elsif Lang = "c" then
               Map_C_Type (Input_Type)
            elsif Lang = "python" then
               Map_Python_Type (Input_Type)
            else
               "");
      begin
         if Mapped'Length = 0 then
            Print_Error ("Unsupported target language: " & Lang);
            Set_Exit_Status (Exit_Error);
            return;
         end if;
         Put_Line (Mapped);
      end;
   end;

   Set_Exit_Status (Exit_Success);

exception
   when others =>
      Print_Error ("Unexpected error");
      Set_Exit_Status (Exit_Error);
end Type_Map_Target;