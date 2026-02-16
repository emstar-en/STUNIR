--  sig_gen_cpp - Generate C++ function signatures from STUNIR spec
--  Generates C++ header declarations from function specifications
--  Phase 3 Powertool for STUNIR

pragma SPARK_Mode (Off);

with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;
with Ada.Strings.Fixed;

with GNAT.Command_Line;

with STUNIR_JSON_Parser;
with STUNIR_Types;

procedure Sig_Gen_CPP is
   use Ada.Command_Line;
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;

   --  Exit codes
   Exit_Success          : constant := 0;
   Exit_Validation_Error : constant := 1;
   Exit_Processing_Error : constant := 2;

   --  Configuration
   Input_File    : Unbounded_String := Null_Unbounded_String;
   Output_File   : Unbounded_String := Null_Unbounded_String;
   Namespace     : Unbounded_String := Null_Unbounded_String;
   Header_Guard  : Unbounded_String := Null_Unbounded_String;
   Verbose_Mode  : Boolean := False;
   Show_Version  : Boolean := False;
   Show_Help     : Boolean := False;
   Show_Describe : Boolean := False;
   Generate_Impl : Boolean := False;

   Version : constant String := "1.0.0";

   --  Description output
   Describe_Output : constant String :=
     "{" & ASCII.LF &
     "  ""tool"": ""sig_gen_cpp""," & ASCII.LF &
     "  ""version"": ""1.0.0""," & ASCII.LF &
     "  ""description"": ""Generate C++ function signatures from STUNIR spec""," & ASCII.LF &
     "  ""inputs"": [{" & ASCII.LF &
     "    ""name"": ""spec_json""," & ASCII.LF &
     "    ""type"": ""json""," & ASCII.LF &
     "    ""source"": [""stdin"", ""file""]," & ASCII.LF &
     "    ""required"": true" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  ""outputs"": [{" & ASCII.LF &
     "    ""name"": ""cpp_header""," & ASCII.LF &
     "    ""type"": ""string""," & ASCII.LF &
     "    ""source"": ""stdout""" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  ""complexity"": ""O(n)""," & ASCII.LF &
     "  ""options"": [" & ASCII.LF &
     "    ""--help"", ""--version"", ""--describe"", ""--namespace"", ""--guard"", ""--output"", ""--impl"", ""--verbose""" & ASCII.LF &
     "  ]" & ASCII.LF &
     "}";

   procedure Print_Usage;
   procedure Print_Error (Msg : String);
   procedure Print_Info (Msg : String);
   function Read_Input return String;
   procedure Write_Output (Content : String);
   function Map_C_To_CPP (C_Type : String) return String;
   function Generate_CPP (Spec_JSON : String) return String;

   procedure Print_Usage is
   begin
      Put_Line ("sig_gen_cpp - Generate C++ function signatures");
      Put_Line ("Version: " & Version);
      Put_Line ("");
      Put_Line ("Usage: sig_gen_cpp [OPTIONS] [INPUT]");
      Put_Line ("");
      Put_Line ("Options:");
      Put_Line ("  --help, -h        Show this help message");
      Put_Line ("  --version, -v     Show version information");
      Put_Line ("  --describe        Show tool description (JSON)");
      Put_Line ("  --namespace NS    Wrap in namespace");
      Put_Line ("  --guard NAME      Header guard macro name");
      Put_Line ("  --output FILE     Output file (default: stdout)");
      Put_Line ("  --impl            Generate implementation stubs");
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

   function Map_C_To_CPP (C_Type : String) return String is
      Trimmed : constant String := Trim (C_Type, Ada.Strings.Both);
   begin
      --  Basic C to C++ type mappings
      if Trimmed = "void" then
         return "void";
      elsif Trimmed = "int" then
         return "int";
      elsif Trimmed = "char" then
         return "char";
      elsif Trimmed = "float" then
         return "float";
      elsif Trimmed = "double" then
         return "double";
      elsif Trimmed = "short" then
         return "short";
      elsif Trimmed = "long" then
         return "long";
      elsif Trimmed = "unsigned int" then
         return "unsigned int";
      elsif Trimmed = "unsigned char" then
         return "unsigned char";
      elsif Trimmed = "unsigned short" then
         return "unsigned short";
      elsif Trimmed = "unsigned long" then
         return "unsigned long";
      elsif Trimmed = "size_t" then
         return "std::size_t";
      elsif Trimmed = "bool" or else Trimmed = "_Bool" then
         return "bool";
      elsif Trimmed = "const char*" or else Trimmed = "char*" then
         return "const char*";
      elsif Trimmed = "void*" then
         return "void*";
      elsif Trimmed = "FILE*" then
         return "std::FILE*";
      else
         --  Custom type - keep as-is
         return Trimmed;
      end if;
   end Map_C_To_CPP;

   function Generate_CPP (Spec_JSON : String) return String is
      use STUNIR_JSON_Parser;
      State  : Parser_State;
      Status : STUNIR_Types.Status_Code;
      Input_Str : STUNIR_Types.JSON_String;

      Result : Unbounded_String := Null_Unbounded_String;
      In_Functions : Boolean := False;
      Function_Count : Natural := 0;

      Current_Func : Unbounded_String := Null_Unbounded_String;
      Current_Return : Unbounded_String := To_Unbounded_String ("void");
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

      --  Generate header guard
      if Header_Guard /= Null_Unbounded_String then
         Append (Result, "#ifndef " & To_String (Header_Guard) & ASCII.LF);
         Append (Result, "#define " & To_String (Header_Guard) & ASCII.LF);
         Append (Result, ASCII.LF);
      end if;

      --  Add includes
      Append (Result, "#include <cstddef>" & ASCII.LF);
      Append (Result, "#include <cstdio>" & ASCII.LF);
      Append (Result, ASCII.LF);

      --  Namespace start
      if Namespace /= Null_Unbounded_String then
         Append (Result, "namespace " & To_String (Namespace) & " {" & ASCII.LF);
         Append (Result, ASCII.LF);
      end if;

      --  Parse functions
      loop
         Next_Token (State, Status);
         exit when Status /= STUNIR_Types.Success or else State.Current_Token = Token_EOF;

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
            Current_Func := Null_Unbounded_String;
            Current_Return := To_Unbounded_String ("void");
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
                  Current_Return := To_Unbounded_String (Map_C_To_CPP (STUNIR_Types.JSON_Strings.To_String (State.Token_Value)));
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
                              Param_Type : Unbounded_String := To_Unbounded_String ("void");
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
                                          Param_Name := State.Token_Value;
                                       elsif PKey = "type" and then State.Current_Token = Token_String then
                                          Param_Type := To_Unbounded_String (Map_C_To_CPP (STUNIR_Types.JSON_Strings.To_String (State.Token_Value)));
                                       else
                                          Skip_Value (State, Status);
                                       end if;
                                    end;
                                 elsif State.Current_Token = Token_Comma then
                                    continue;
                                 end if;
                              end loop;

                              if not First_Param then
                                 Append (Current_Params, ", ");
                              end if;
                              First_Param := False;
                              Append (Current_Params, STUNIR_Types.JSON_Strings.To_String (Param_Type));
                              Append (Current_Params, " ");
                              Append (Current_Params, STUNIR_Types.JSON_Strings.To_String (Param_Name));
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
            if Current_Func /= Null_Unbounded_String then
               Function_Count := Function_Count + 1;

               if Generate_Impl then
                  Append (Result, STUNIR_Types.JSON_Strings.To_String (Current_Return) & " ");
                  Append (Result, STUNIR_Types.JSON_Strings.To_String (Current_Func));
                  Append (Result, "(" & To_String (Current_Params) & ") {" & ASCII.LF);
                  Append (Result, "    // TODO: Implementation" & ASCII.LF);
                  Append (Result, "}" & ASCII.LF);
               else
                  Append (Result, STUNIR_Types.JSON_Strings.To_String (Current_Return) & " ");
                  Append (Result, STUNIR_Types.JSON_Strings.To_String (Current_Func));
                  Append (Result, "(" & To_String (Current_Params) & ");" & ASCII.LF);
               end if;
               Append (Result, ASCII.LF);
            end if;
         elsif State.Current_Token = Token_Array_End and then In_Functions then
            In_Functions := False;
         end if;
      end loop;

      --  Namespace end
      if Namespace /= Null_Unbounded_String then
         Append (Result, "} // namespace " & To_String (Namespace) & ASCII.LF);
         Append (Result, ASCII.LF);
      end if;

      --  Header guard end
      if Header_Guard /= Null_Unbounded_String then
         Append (Result, "#endif // " & To_String (Header_Guard) & ASCII.LF);
      end if;

      Print_Info ("Generated" & Function_Count'Img & " function signatures");

      return To_String (Result);
   end Generate_CPP;

   Config : GNAT.Command_Line.Command_Line_Configuration;

begin
   GNAT.Command_Line.Define_Switch (Config, Show_Help'Access, "-h", "--help");
   GNAT.Command_Line.Define_Switch (Config, Show_Version'Access, "-v", "--version");
   GNAT.Command_Line.Define_Switch (Config, Show_Describe'Access, "", "--describe");
   GNAT.Command_Line.Define_Switch (Config, Verbose_Mode'Access, "", "--verbose");
   GNAT.Command_Line.Define_Switch (Config, Generate_Impl'Access, "", "--impl");
   GNAT.Command_Line.Define_Switch (Config, Namespace'Access, "-n:", "--namespace=");
   GNAT.Command_Line.Define_Switch (Config, Header_Guard'Access, "-g:", "--guard=");
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

   declare
      Content : constant String := Read_Input;
      Result  : constant String := Generate_CPP (Content);
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
end Sig_Gen_CPP;