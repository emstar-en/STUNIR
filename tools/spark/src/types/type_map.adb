--  type_map - Map C types to target language types
--  Converts canonical C types to C++, Python, Rust, or Go types
--  Phase 1 Powertool for STUNIR

pragma SPARK_Mode (Off);

with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;
with Ada.Strings.Fixed;
with Ada.Strings.Maps;

with GNAT.Command_Line;
with GNAT.Strings;

procedure Type_Map is
   use Ada.Command_Line;
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;
   use Ada.Strings.Fixed;

   --  Exit codes
   Exit_Success          : constant := 0;
   Exit_Validation_Error : constant := 1;
   Exit_Processing_Error : constant := 2;

   --  Target languages
   type Target_Language is (Lang_C, Lang_CPP, Lang_Python, Lang_Rust, Lang_Go);

   --  Configuration
   Input_Type    : Unbounded_String := Null_Unbounded_String;
   Target_Lang   : Target_Language := Lang_CPP;
   Output_File   : aliased GNAT.Strings.String_Access := new String'("");
   Verbose_Mode  : aliased Boolean := False;
   Show_Version  : aliased Boolean := False;
   Show_Help     : aliased Boolean := False;
   Show_Describe : aliased Boolean := False;
   From_Stdin    : Boolean := False;

   Version : constant String := "0.1.0-alpha";

   --  Description output
   Describe_Output : constant String :=
     "{" & ASCII.LF &
     "  ""tool"": ""type_map""," & ASCII.LF &
     "  ""version"": ""0.1.0-alpha""," & ASCII.LF &
     "  ""description"": ""Map C types to target language types""," & ASCII.LF &
     "  ""inputs"": [{" & ASCII.LF &
     "    ""name"": ""c_type""," & ASCII.LF &
     "    ""type"": ""string""," & ASCII.LF &
     "    ""source"": [""stdin"", ""argument""]," & ASCII.LF &
     "    ""required"": true" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  ""outputs"": [{" & ASCII.LF &
     "    ""name"": ""mapped_type""," & ASCII.LF &
     "    ""type"": ""string""," & ASCII.LF &
     "    ""source"": ""stdout""" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  ""complexity"": ""O(1)""," & ASCII.LF &
     "  ""options"": [" & ASCII.LF &
     "    ""--help"", ""--version"", ""--describe"", ""--lang"", ""--output"", ""--verbose""" & ASCII.LF &
     "  ]" & ASCII.LF &
     "}";

   procedure Print_Usage;
   procedure Print_Error (Msg : String);
   procedure Print_Info (Msg : String);
   function Map_Type (Type_Str : String; Lang : Target_Language) return String;
   procedure Write_Output (Content : String);

   procedure Print_Usage is
   begin
      Put_Line ("type_map - Map C types to target language");
      Put_Line ("Version: " & Version);
      Put_Line ("");
      Put_Line ("Usage: type_map [OPTIONS] [TYPE]");
      Put_Line ("");
      Put_Line ("Options:");
      Put_Line ("  --help, -h        Show this help message");
      Put_Line ("  --version, -v     Show version information");
      Put_Line ("  --describe        Show tool description (JSON)");
      Put_Line ("  --lang LANG       Target language (c, cpp, python, rust, go)");
      Put_Line ("                    (default: cpp)");
      Put_Line ("  --output FILE     Output file (default: stdout)");
      Put_Line ("  --verbose         Verbose output");
      Put_Line ("");
      Put_Line ("Arguments:");
      Put_Line ("  TYPE              C type string to map");
      Put_Line ("                    (reads from stdin if not provided)");
      Put_Line ("");
      Put_Line ("Examples:");
      Put_Line ("  type_map --lang rust 'const char*'");
      Put_Line ("  echo 'int' | type_map --lang python");
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

   function Is_Pointer_Type (Type_Str : String) return Boolean is
   begin
      for I in Type_Str'Range loop
         if Type_Str (I) = '*' then
            return True;
         end if;
      end loop;
      return False;
   end Is_Pointer_Type;

   function Is_Const_Type (Type_Str : String) return Boolean is
   begin
      for I in Type_Str'First .. Type_Str'Last - 4 loop
         if (Type_Str (I .. I + 4) = "const" or else Type_Str (I .. I + 4) = "CONST") and then
            (I = Type_Str'First or else Type_Str (I - 1) = ' ')
         then
            return True;
         end if;
      end loop;
      return False;
   end Is_Const_Type;

   function Base_Type (Type_Str : String) return String is
      Result : Unbounded_String := Null_Unbounded_String;
   begin
      for I in Type_Str'Range loop
         if Type_Str (I) = '*' or else Type_Str (I) = ' ' then
            exit;
         end if;
         Append (Result, Type_Str (I));
      end loop;
      return To_String (Result);
   end Base_Type;

   function Map_Type (Type_Str : String; Lang : Target_Language) return String is
      Trimmed : constant String := Trim (Type_Str, Ada.Strings.Both);
      Is_Ptr  : constant Boolean := Is_Pointer_Type (Trimmed);
      Is_Const : constant Boolean := Is_Const_Type (Trimmed);
      Base    : constant String := Base_Type (Trimmed);

      --  Mapping tables
      function Map_C_Type (B : String) return String is
      begin
         if B = "void" then
            return "void";
         elsif B = "int" then
            return "int";
         elsif B = "char" then
            return "char";
         elsif B = "float" then
            return "float";
         elsif B = "double" then
            return "double";
         elsif B = "short" then
            return "short";
         elsif B = "long" then
            return "long";
         elsif B = "unsigned" then
            return "unsigned";
         elsif B = "signed" then
            return "signed";
         elsif B = "size_t" then
            return "size_t";
         elsif B = "bool" or else B = "_Bool" then
            return "bool";
         elsif B = "FILE" then
            return "FILE";
         else
            return B;  --  Custom type
         end if;
      end Map_C_Type;

      function Map_CPP_Type (B : String) return String is
      begin
         if B = "void" then
            return (if Is_Ptr then "void*" else "void");
         elsif B = "int" then
            return "int";
         elsif B = "char" then
            if Is_Ptr then
               return (if Is_Const then "const char*" else "char*");
            else
               return "char";
            end if;
         elsif B = "float" then
            return "float";
         elsif B = "double" then
            return "double";
         elsif B = "short" then
            return "short";
         elsif B = "long" then
            return "long";
         elsif B = "unsigned" then
            return "unsigned";
         elsif B = "signed" then
            return "signed";
         elsif B = "size_t" then
            return "size_t";
         elsif B = "bool" or else B = "_Bool" then
            return "bool";
         elsif B = "FILE" then
            return "FILE*";
         else
            --  Custom type - keep as-is, add pointer if needed
            return B & (if Is_Ptr then "*" else "");
         end if;
      end Map_CPP_Type;

      function Map_Python_Type (B : String) return String is
      begin
         if B = "void" then
            return (if Is_Ptr then "ctypes.c_void_p" else "None");
         elsif B = "int" then
            return "int";
         elsif B = "char" then
            if Is_Ptr then
               return "str";
            else
               return "str";
            end if;
         elsif B = "float" then
            return "float";
         elsif B = "double" then
            return "float";
         elsif B = "short" then
            return "int";
         elsif B = "long" then
            return "int";
         elsif B = "unsigned" then
            return "int";
         elsif B = "signed" then
            return "int";
         elsif B = "size_t" then
            return "int";
         elsif B = "bool" or else B = "_Bool" then
            return "bool";
         elsif B = "FILE" then
            return "io.IOBase";
         else
            return "Any";
         end if;
      end Map_Python_Type;

      function Map_Rust_Type (B : String) return String is
      begin
         if B = "void" then
            return (if Is_Ptr then "*mut c_void" else "()");
         elsif B = "int" then
            return "c_int";
         elsif B = "char" then
            if Is_Ptr then
               return (if Is_Const then "*const c_char" else "*mut c_char");
            else
               return "c_char";
            end if;
         elsif B = "float" then
            return "c_float";
         elsif B = "double" then
            return "c_double";
         elsif B = "short" then
            return "c_short";
         elsif B = "long" then
            return "c_long";
         elsif B = "unsigned" then
            return "c_uint";
         elsif B = "signed" then
            return "c_int";
         elsif B = "size_t" then
            return "usize";
         elsif B = "bool" or else B = "_Bool" then
            return "bool";
         elsif B = "FILE" then
            return "*mut FILE";
         else
            --  Custom type
            return B & (if Is_Ptr then "*" else "");
         end if;
      end Map_Rust_Type;

      function Map_Go_Type (B : String) return String is
      begin
         if B = "void" then
            return (if Is_Ptr then "unsafe.Pointer" else "");
         elsif B = "int" then
            return "C.int";
         elsif B = "char" then
            if Is_Ptr then
               return "*C.char";
            else
               return "C.char";
            end if;
         elsif B = "float" then
            return "C.float";
         elsif B = "double" then
            return "C.double";
         elsif B = "short" then
            return "C.short";
         elsif B = "long" then
            return "C.long";
         elsif B = "unsigned" then
            return "C.uint";
         elsif B = "signed" then
            return "C.int";
         elsif B = "size_t" then
            return "C.size_t";
         elsif B = "bool" or else B = "_Bool" then
            return "C.bool";
         elsif B = "FILE" then
            return "*C.FILE";
         else
            --  Custom type - use C.<type>
            return "C." & B;
         end if;
      end Map_Go_Type;

   begin
      case Lang is
         when Lang_C =>
            return Map_C_Type (Base);
         when Lang_CPP =>
            return Map_CPP_Type (Base);
         when Lang_Python =>
            return Map_Python_Type (Base);
         when Lang_Rust =>
            return Map_Rust_Type (Base);
         when Lang_Go =>
            return Map_Go_Type (Base);
      end case;
   end Map_Type;

   procedure Write_Output (Content : String) is
   begin
      if Output_File.all = "" then
         Put_Line (Content);
      else
         declare
            File : File_Type;
         begin
            Create (File, Out_File, Output_File.all);
            Put (File, Content);
            Close (File);
         exception
            when others =>
               Print_Error ("Cannot write: " & Output_File.all);
         end;
      end if;
   end Write_Output;

   Config : GNAT.Command_Line.Command_Line_Configuration;
   Lang_Str : aliased GNAT.Strings.String_Access := new String'("cpp");

begin
   GNAT.Command_Line.Define_Switch (Config, Show_Help'Access, "-h", "--help");
   GNAT.Command_Line.Define_Switch (Config, Show_Version'Access, "-v", "--version");
   GNAT.Command_Line.Define_Switch (Config, Show_Describe'Access, "", "--describe");
   GNAT.Command_Line.Define_Switch (Config, Verbose_Mode'Access, "", "--verbose");
   GNAT.Command_Line.Define_Switch (Config, Lang_Str'Access, "-l:", "--lang=");
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

   --  Parse language
   declare
      L : constant String := Trim (Lang_Str.all, Ada.Strings.Both);
   begin
      if L = "c" or else L = "C" then
         Target_Lang := Lang_C;
      elsif L = "cpp" or else L = "c++" or else L = "CXX" then
         Target_Lang := Lang_CPP;
      elsif L = "python" or else L = "py" then
         Target_Lang := Lang_Python;
      elsif L = "rust" or else L = "rs" then
         Target_Lang := Lang_Rust;
      elsif L = "go" or else L = "golang" then
         Target_Lang := Lang_Go;
      else
         Print_Error ("Unknown language: " & L);
         Set_Exit_Status (Exit_Validation_Error);
         return;
      end if;
   end;

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
      Result : constant String := Map_Type (To_String (Input_Type), Target_Lang);
   begin
      Print_Info ("Input:  '" & To_String (Input_Type) & "'");
      Print_Info ("Target: " & Target_Lang'Img);
      Print_Info ("Output: '" & Result & "'");
      Write_Output (Result);
      Set_Exit_Status (Exit_Success);
   end;

exception
   when others =>
      Print_Error ("Unexpected error");
      Set_Exit_Status (Exit_Processing_Error);
end Type_Map;