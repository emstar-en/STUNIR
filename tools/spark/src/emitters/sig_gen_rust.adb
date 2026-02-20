--  sig_gen_rust - Generate Rust function signatures from STUNIR spec
--  Generates Rust FFI declarations from function specifications
--  Phase 3 Powertool for STUNIR

pragma SPARK_Mode (Off);

with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;
with Ada.Strings.Fixed;

with GNAT.Command_Line;
with GNAT.Strings;

with STUNIR_JSON_Parser;
with STUNIR_Types;

procedure Sig_Gen_Rust is
   use Ada.Command_Line;
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;
   use GNAT.Strings;
   use STUNIR_JSON_Parser;
   use STUNIR_Types;

   --  Exit codes
   Exit_Success          : constant := 0;
   Exit_Validation_Error : constant := 1;
   Exit_Processing_Error : constant := 2;

   --  Configuration
   Input_File    : Unbounded_String := Null_Unbounded_String;
   Output_File   : aliased GNAT.Strings.String_Access := new String'("");
   Module_Name   : aliased GNAT.Strings.String_Access := new String'("");
   Verbose_Mode  : aliased Boolean := False;
   Show_Version  : aliased Boolean := False;
   Show_Help     : aliased Boolean := False;
   Show_Describe : aliased Boolean := False;
   Unsafe_FFI    : aliased Boolean := True;

   Version : constant String := "0.1.0-alpha";

   --  Description output
   Describe_Output : constant String :=
     "{" & ASCII.LF &
     "  \"tool\": \"sig_gen_rust\"," & ASCII.LF &
     "  \"version\": \"0.1.0-alpha\"," & ASCII.LF &
     "  \"description\": \"Generate Rust FFI signatures from STUNIR spec\"," & ASCII.LF &
     "  \"inputs\": [{" & ASCII.LF &
     "    \"name\": \"spec_json\"," & ASCII.LF &
     "    \"type\": \"json\"," & ASCII.LF &
     "    \"source\": [\"stdin\", \"file\"]," & ASCII.LF &
     "    \"required\": true" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  \"outputs\": [{" & ASCII.LF &
     "    \"name\": \"rust_code\"," & ASCII.LF &
     "    \"type\": \"string\"," & ASCII.LF &
     "    \"source\": \"stdout\"" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  \"complexity\": \"O(n)\"," & ASCII.LF &
     "  \"options\": [" & ASCII.LF &
     "    \"--help\", \"--version\", \"--describe\", \"--module\", \"--output\", \"--safe\", \"--verbose\"" & ASCII.LF &
     "  ]" & ASCII.LF &
     "}";

   procedure Print_Usage;
   procedure Print_Error (Msg : String);
   procedure Print_Info (Msg : String);
   function Read_Input return String;
   procedure Write_Output (Content : String);
   function Map_C_To_Rust (C_Type : String) return String;
   function Generate_Rust (Spec_JSON : String) return String;

   procedure Print_Usage is
   begin
      Put_Line ("sig_gen_rust - Generate Rust FFI signatures");
      Put_Line ("Version: " & Version);
      Put_Line ("");
      Put_Line ("Usage: sig_gen_rust [OPTIONS] [INPUT]");
      Put_Line ("");
      Put_Line ("Options:");
      Put_Line ("  --help, -h        Show this help message");
      Put_Line ("  --version, -v     Show version information");
      Put_Line ("  --describe        Show tool description (JSON)");
      Put_Line ("  --module NAME     Wrap in mod block");
      Put_Line ("  --output FILE     Output file (default: stdout)");
      Put_Line ("  --safe            Generate safe wrappers (default: unsafe FFI)");
      Put_Line ("  --verbose         Verbose output");
      Put_Line ("");
      Put_Line ("Arguments:");
      Put_Line ("  INPUT             STUNIR spec JSON (default: stdin)");
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

   function Read_Input return String is
      Result : Unbounded_String := Null_Unbounded_String;
   begin
      if Input_File = Null_Unbounded_String then
         while not End_Of_File loop
            declare
               Line : String (1 .. 4096);
               Last : Natural;
            begin
               Get_Line (Line, Last);
               Append (Result, Line (1 .. Last));
               Append (Result, ASCII.LF);
            end;
         end loop;
      else
         declare
            File : File_Type;
            Line : String (1 .. 4096);
            Last : Natural;
         begin
            Open (File, In_File, To_String (Input_File));
            while not End_Of_File (File) loop
               Get_Line (File, Line, Last);
               Append (Result, Line (1 .. Last));
               Append (Result, ASCII.LF);
            end loop;
            Close (File);
         exception
            when others =>
               Print_Error ("Cannot read: " & To_String (Input_File));
               return "";
         end;
      end if;
      return To_String (Result);
   end Read_Input;

   procedure Write_Output (Content : String) is
   begin
      if Output_File.all'Length = 0 then
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

   function Map_C_To_Rust (C_Type : String) return String is
      Trimmed : constant String := Ada.Strings.Fixed.Trim (C_Type, Ada.Strings.Both);
   begin
      --  Basic C to Rust FFI type mappings
      if Trimmed = "void" then
         return "c_void";
      elsif Trimmed = "int" then
         return "c_int";
      elsif Trimmed = "char" then
         return "c_char";
      elsif Trimmed = "float" then
         return "c_float";
      elsif Trimmed = "double" then
         return "c_double";
      elsif Trimmed = "short" then
         return "c_short";
      elsif Trimmed = "long" then
         return "c_long";
      elsif Trimmed = "unsigned int" then
         return "c_uint";
      elsif Trimmed = "unsigned char" then
         return "c_uchar";
      elsif Trimmed = "unsigned short" then
         return "c_ushort";
      elsif Trimmed = "unsigned long" then
         return "c_ulong";
      elsif Trimmed = "size_t" then
         return "usize";
      elsif Trimmed = "bool" or else Trimmed = "_Bool" then
         return "bool";
      elsif Trimmed = "const char*" then
         return "*const c_char";
      elsif Trimmed = "char*" then
         return "*mut c_char";
      elsif Trimmed = "void*" then
         return "*mut c_void";
      elsif Trimmed = "const void*" then
         return "*const c_void";
      elsif Trimmed = "FILE*" then
         return "*mut FILE";
      else
         --  Custom type - assume it's a C type that needs repr(C)
         return Trimmed;
      end if;
   end Map_C_To_Rust;

   function Generate_Rust (Spec_JSON : String) return String is
      use STUNIR_JSON_Parser;
      State  : Parser_State;
      Status : STUNIR_Types.Status_Code;
      Input_Str : STUNIR_Types.JSON_String;

      Result : Unbounded_String := Null_Unbounded_String;
      In_Functions : Boolean := False;
      Function_Count : Natural := 0;

      Current_Func : JSON_String := JSON_Strings.Null_Bounded_String;
      Current_Return : JSON_String := JSON_Strings.To_Bounded_String ("c_void");
      Current_Params : Unbounded_String := Null_Unbounded_String;
   begin
      if Spec_JSON'Length = 0 then
         return "// Empty spec";
      end if;

      Input_Str := STUNIR_Types.JSON_Strings.To_Bounded_String (Spec_JSON);
      Initialize_Parser (State, Input_Str, Status);
      if Status /= STUNIR_Types.Success then
         return "// Failed to parse spec";
      end if;

      --  Generate header
      Append (Result, "use std::os::raw::{c_void, c_int, c_char, c_float, c_double};" & ASCII.LF);
      Append (Result, "use std::os::raw::{c_short, c_long, c_uint, c_uchar, c_ushort, c_ulong};" & ASCII.LF);
      Append (Result, ASCII.LF);

      --  Module start
      if Module_Name.all'Length > 0 then
         Append (Result, "pub mod " & Module_Name.all & " {" & ASCII.LF);
         Append (Result, ASCII.LF);
         Append (Result, "    use std::os::raw::{c_void, c_int, c_char, c_float, c_double};" & ASCII.LF);
         Append (Result, "    use std::os::raw::{c_short, c_long, c_uint, c_uchar, c_ushort, c_ulong};" & ASCII.LF);
         Append (Result, ASCII.LF);
      end if;

      --  Parse functions
      loop
         Next_Token (State, Status);
         exit when Status /= STUNIR_Types.Success or else State.Current_Token = STUNIR_Types.Token_EOF;

         if State.Current_Token = Token_String then
            declare
               Key : constant String := STUNIR_Types.JSON_Strings.To_String (State.Token_Value);
            begin
               if Key = "functions" then
                  Expect_Token (State, Token_Colon, Status);
                  if Status = STUNIR_Types.Success then
                     Expect_Token (State, Token_Array_Start, Status);
                     if Status = STUNIR_Types.Success then
                        In_Functions := True;
                     end if;
                  end if;
               end if;
            end;
         elsif State.Current_Token = Token_Object_Start and then In_Functions then
            --  New function
            Current_Func := JSON_Strings.Null_Bounded_String;
            Current_Return := JSON_Strings.To_Bounded_String ("c_void");
            Current_Params := Null_Unbounded_String;

         elsif State.Current_Token = Token_String and then In_Functions then
            declare
               Member_Key : constant String := STUNIR_Types.JSON_Strings.To_String (State.Token_Value);
            begin
               Expect_Token (State, Token_Colon, Status);
               exit when Status /= STUNIR_Types.Success;

               Next_Token (State, Status);
               exit when Status /= STUNIR_Types.Success;

               if Member_Key = "name" and then State.Current_Token = Token_String then
                  Current_Func := State.Token_Value;
               elsif Member_Key = "return_type" and then State.Current_Token = Token_String then
                  Current_Return := JSON_Strings.To_Bounded_String (Map_C_To_Rust (STUNIR_Types.JSON_Strings.To_String (State.Token_Value)));
               elsif Member_Key = "parameters" and then State.Current_Token = Token_Array_Start then
                  --  Parse parameters array
                  declare
                     First_Param : Boolean := True;
                     Param_Depth : Natural := 1;
                  begin
                     loop
                        Next_Token (State, Status);
                        exit when Status /= STUNIR_Types.Success or else Param_Depth = 0;

                        if State.Current_Token = Token_Object_Start then
                           --  Parse parameter object
                           declare
                              Param_Name : Unbounded_String := Null_Unbounded_String;
                              Param_Type : Unbounded_String := To_Unbounded_String ("c_void");
                           begin
                              loop
                                 Next_Token (State, Status);
                                 exit when Status /= STUNIR_Types.Success or else State.Current_Token = Token_Object_End;

                                 if State.Current_Token = Token_String then
                                    declare
                                       PKey : constant String := STUNIR_Types.JSON_Strings.To_String (State.Token_Value);
                                    begin
                                       Expect_Token (State, Token_Colon, Status);
                                       exit when Status /= STUNIR_Types.Success;

                                       Next_Token (State, Status);
                                       exit when Status /= STUNIR_Types.Success;

                                       if PKey = "name" and then State.Current_Token = Token_String then
                                          Param_Name := To_Unbounded_String (STUNIR_Types.JSON_Strings.To_String (State.Token_Value));
                                       elsif PKey = "type" and then State.Current_Token = Token_String then
                                          Param_Type := To_Unbounded_String (Map_C_To_Rust (STUNIR_Types.JSON_Strings.To_String (State.Token_Value)));
                                       else
                                          Skip_Value (State, Status);
                                       end if;
                                    end;
                                 elsif State.Current_Token = Token_Comma then
                                    null;
                                 end if;
                              end loop;

                              if not First_Param then
                                 Append (Current_Params, ", ");
                              end if;
                              First_Param := False;
                              Append (Current_Params, To_String (Param_Name));
                              Append (Current_Params, ": ");
                              Append (Current_Params, To_String (Param_Type));
                           end;
                        elsif State.Current_Token = Token_Array_End then
                           Param_Depth := Param_Depth - 1;
                        end if;
                     end loop;
                  end;
               else
                  Skip_Value (State, Status);
               end if;
            end;
         elsif State.Current_Token = Token_Object_End and then In_Functions then
            --  End of function - generate signature
            if JSON_Strings."/=" (Current_Func, JSON_Strings.Null_Bounded_String) then
               Function_Count := Function_Count + 1;

               if Module_Name.all'Length > 0 then
                  Append (Result, "    ");
               end if;

               if Unsafe_FFI then
                  Append (Result, "pub unsafe extern \"C\" fn ");
                  Append (Result, STUNIR_Types.JSON_Strings.To_String (Current_Func));
                  Append (Result, "(" & To_String (Current_Params) & ")");
                  Append (Result, " -> " & STUNIR_Types.JSON_Strings.To_String (Current_Return));
                  Append (Result, ";" & ASCII.LF);
               else
                  --  Safe wrapper
                  Append (Result, "pub fn ");
                  Append (Result, STUNIR_Types.JSON_Strings.To_String (Current_Func));
                  Append (Result, "(" & To_String (Current_Params) & ")");
                  Append (Result, " -> " & STUNIR_Types.JSON_Strings.To_String (Current_Return));
                  Append (Result, " {" & ASCII.LF);
                  if Module_Name.all'Length > 0 then
                     Append (Result, "        ");
                  else
                     Append (Result, "    ");
                  end if;
                  Append (Result, "// TODO: Safe wrapper implementation" & ASCII.LF);
                  if Module_Name.all'Length > 0 then
                     Append (Result, "    }" & ASCII.LF);
                  else
                     Append (Result, "}" & ASCII.LF);
                  end if;
               end if;
               Append (Result, ASCII.LF);
            end if;
         elsif State.Current_Token = Token_Array_End and then In_Functions then
            In_Functions := False;
         end if;
      end loop;

      --  Module end
      if Module_Name.all'Length > 0 then
         Append (Result, "} // mod " & Module_Name.all & ASCII.LF);
      end if;

      Print_Info ("Generated" & Function_Count'Img & " Rust signatures");

      return To_String (Result);
   end Generate_Rust;

   Config : GNAT.Command_Line.Command_Line_Configuration;

begin
   GNAT.Command_Line.Define_Switch (Config, Show_Help'Access, "-h", "--help");
   GNAT.Command_Line.Define_Switch (Config, Show_Version'Access, "-v", "--version");
   GNAT.Command_Line.Define_Switch (Config, Show_Describe'Access, "", "--describe");
   GNAT.Command_Line.Define_Switch (Config, Verbose_Mode'Access, "", "--verbose");
   GNAT.Command_Line.Define_Switch (Config, Unsafe_FFI'Access, "", "--safe");
   GNAT.Command_Line.Define_Switch (Config, Module_Name'Access, "-m:", "--module=");
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

   declare
      Arg : String := GNAT.Command_Line.Get_Argument;
   begin
      if Arg'Length > 0 then
         Input_File := To_Unbounded_String (Arg);
      end if;
   end;

   if Module_Name.all'Length > 0 then
      null;  --  Module name is set
   end if;

   declare
      Content : constant String := Read_Input;
      Result  : constant String := Generate_Rust (Content);
   begin
      if Content'Length = 0 then
         Print_Error ("Empty input");
         Set_Exit_Status (Exit_Processing_Error);
         return;
      end if;

      Write_Output (Result);
      Set_Exit_Status (Exit_Success);
   end;

exception
   when others =>
      Print_Error ("Unexpected error");
      Set_Exit_Status (Exit_Processing_Error);
end Sig_Gen_Rust;
