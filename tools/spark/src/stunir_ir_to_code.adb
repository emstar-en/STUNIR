-------------------------------------------------------------------------------
--  STUNIR IR to Code Emitter - Ada SPARK Implementation
--  PRIMARY IMPLEMENTATION (Ada SPARK is the default language for STUNIR tools)
--
--  This package implements deterministic code generation from STUNIR IR.
--
--  Copyright (c) 2026 STUNIR Project
--  License: MIT
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

with Ada.Text_IO;
with Ada.Directories;
with Ada.Command_Line;
with Ada.Characters.Handling;
with STUNIR_JSON_Utils;
with STUNIR_String_Builder;

package body STUNIR_IR_To_Code is

   use Ada.Text_IO;
   use Ada.Directories;
   use Ada.Characters.Handling;
   use STUNIR_JSON_Utils;

   --  Initialize emission configuration
   procedure Initialize_Config
     (Config        : out Emission_Config;
      IR_Path       : String;
      Template_Path : String;
      Output_Path   : String;
      Target        : Target_Language := Target_Python)
   is
   begin
      Config.IR_Path       := Path_Strings.To_Bounded_String (IR_Path);
      Config.Template_Path := Path_Strings.To_Bounded_String (Template_Path);
      Config.Output_Path   := Path_Strings.To_Bounded_String (Output_Path);
      Config.Target        := Target;
      Config.Emit_Comments := True;
      Config.Emit_Metadata := True;
   end Initialize_Config;

   --  Get file extension for target language
   function Get_File_Extension (Target : Target_Language) return String is
   begin
      case Target is
         when Target_Python     => return ".py";
         when Target_Rust       => return ".rs";
         when Target_C          => return ".c";
         when Target_Cpp        => return ".cpp";
         when Target_Go         => return ".go";
         when Target_JavaScript => return ".js";
         when Target_TypeScript => return ".ts";
         when Target_Java       => return ".java";
         when Target_CSharp     => return ".cs";
         when Target_WASM       => return ".wasm";
         when Target_Assembly_X86 => return ".asm";
         when Target_Assembly_ARM => return ".s";
      end case;
   end Get_File_Extension;

   --  Get target language from string
   function Parse_Target (S : String) return Target_Language is
      Lower_S : constant String := To_Lower (S);
   begin
      if Lower_S = "python" or Lower_S = "py" then
         return Target_Python;
      elsif Lower_S = "rust" or Lower_S = "rs" then
         return Target_Rust;
      elsif Lower_S = "c" then
         return Target_C;
      elsif Lower_S = "cpp" or Lower_S = "c++" then
         return Target_Cpp;
      elsif Lower_S = "go" or Lower_S = "golang" then
         return Target_Go;
      elsif Lower_S = "javascript" or Lower_S = "js" then
         return Target_JavaScript;
      elsif Lower_S = "typescript" or Lower_S = "ts" then
         return Target_TypeScript;
      elsif Lower_S = "java" then
         return Target_Java;
      elsif Lower_S = "csharp" or Lower_S = "cs" then
         return Target_CSharp;
      elsif Lower_S = "wasm" then
         return Target_WASM;
      elsif Lower_S = "x86" or Lower_S = "asm" then
         return Target_Assembly_X86;
      elsif Lower_S = "arm" then
         return Target_Assembly_ARM;
      else
         return Target_Python;  --  Default
      end if;
   end Parse_Target;

   --  Parse IR from JSON file (NOW SUPPORTS SEMANTIC IR FORMAT)
   procedure Parse_IR
     (IR_Path   : Path_String;
      Module    : out IR_Module;
      Success   : out Boolean)
   is
      use STUNIR_JSON_Utils;
      
      File_Name   : constant String := Path_Strings.To_String (IR_Path);
      File        : File_Type;
      JSON_Str    : String (1 .. 100_000);
      Last        : Natural;
   begin
      --  Initialize with minimal defaults
      Module.Schema := Name_Strings.Null_Bounded_String;
      Module.Module_Name := Name_Strings.Null_Bounded_String;
      Module.Description := Path_Strings.Null_Bounded_String;
      Module.Func_Count := 0;
      Success := False;

      if not Exists (File_Name) then
         Put_Line ("[ERROR] IR file not found: " & File_Name);
         return;
      end if;

      --  Read entire JSON file
      Open (File, In_File, File_Name);
      Last := 0;
      
      while not End_Of_File (File) and Last < JSON_Str'Last loop
         declare
            Line : constant String := Get_Line (File);
         begin
            if Last + Line'Length <= JSON_Str'Last then
               JSON_Str (Last + 1 .. Last + Line'Length) := Line;
               Last := Last + Line'Length;
            end if;
         end;
      end loop;
      
      Close (File);

      --  Extract key fields from JSON
      declare
         Schema_Str : constant String := Extract_String_Value (JSON_Str (1 .. Last), "schema");
         Mod_Str    : constant String := Extract_String_Value (JSON_Str (1 .. Last), "module_name");
         Funcs_Pos  : constant Natural := Find_Array (JSON_Str (1 .. Last), "functions");
      begin
         if Schema_Str'Length > 0 then
            Module.Schema := Name_Strings.To_Bounded_String (Schema_Str);
            Put_Line ("[INFO] Parsed IR with schema: " & Schema_Str);
         end if;
         
         if Mod_Str'Length > 0 then
            Module.Module_Name := Name_Strings.To_Bounded_String (Mod_Str);
            Put_Line ("[INFO] Module name: " & Mod_Str);
         end if;
         
         --  Parse functions array from IR
         if Funcs_Pos > 0 then
            declare
               Func_Pos : Natural := Funcs_Pos + 1;
               Obj_Start, Obj_End : Natural;
            begin
               Module.Func_Count := 0;
               
               while Module.Func_Count < Max_Functions loop
                  Get_Next_Object (JSON_Str (1 .. Last), Func_Pos, Obj_Start, Obj_End);
                  exit when Obj_Start = 0 or Obj_End = 0;
                  
                  declare
                     Func_JSON : constant String := JSON_Str (Obj_Start .. Obj_End);
                     Func_Name : constant String := Extract_String_Value (Func_JSON, "name");
                     Func_Ret  : constant String := Extract_String_Value (Func_JSON, "return_type");
                     Args_Pos  : constant Natural := Find_Array (Func_JSON, "args");
                  begin
                     if Func_Name'Length > 0 then
                        Module.Func_Count := Module.Func_Count + 1;
                        Module.Functions (Module.Func_Count).Name := 
                          Name_Strings.To_Bounded_String (Func_Name);
                        
                        if Func_Ret'Length > 0 then
                           Module.Functions (Module.Func_Count).Return_Type := 
                             Name_Strings.To_Bounded_String (Func_Ret);
                        else
                           Module.Functions (Module.Func_Count).Return_Type := 
                             Name_Strings.To_Bounded_String ("void");
                        end if;
                        
                        Module.Functions (Module.Func_Count).Param_Count := 0;
                        Module.Functions (Module.Func_Count).Is_Public := True;
                        Module.Functions (Module.Func_Count).Step_Count := 0;
                        
                        --  Parse args
                        if Args_Pos > 0 then
                           declare
                              Arg_Pos : Natural := Args_Pos + 1;
                              Arg_Start, Arg_End : Natural;
                              Func_Idx : constant Positive := Module.Func_Count;
                           begin
                              while Module.Functions (Func_Idx).Param_Count < Max_Params loop
                                 Get_Next_Object (Func_JSON, Arg_Pos, Arg_Start, Arg_End);
                                 exit when Arg_Start = 0 or Arg_End = 0;
                                 
                                 declare
                                    Arg_JSON : constant String := Func_JSON (Arg_Start .. Arg_End);
                                    Arg_Name : constant String := Extract_String_Value (Arg_JSON, "name");
                                    Arg_Type : constant String := Extract_String_Value (Arg_JSON, "type");
                                 begin
                                    if Arg_Name'Length > 0 then
                                       Module.Functions (Func_Idx).Param_Count := 
                                         Module.Functions (Func_Idx).Param_Count + 1;
                                       Module.Functions (Func_Idx).Params (Module.Functions (Func_Idx).Param_Count).Name :=
                                         Name_Strings.To_Bounded_String (Arg_Name);
                                       if Arg_Type'Length > 0 then
                                          Module.Functions (Func_Idx).Params (Module.Functions (Func_Idx).Param_Count).Type_Name :=
                                            Name_Strings.To_Bounded_String (Arg_Type);
                                       else
                                          Module.Functions (Func_Idx).Params (Module.Functions (Func_Idx).Param_Count).Type_Name :=
                                            Name_Strings.To_Bounded_String ("int");
                                       end if;
                                    end if;
                                 end;
                                 
                                 Arg_Pos := Arg_End + 1;
                              end loop;
                           end;
                        
                        --  Parse steps (IR operations)
                        declare
                           Steps_Pos : constant Natural := Find_Array (Func_JSON, "steps");
                        begin
                           if Steps_Pos > 0 then
                              declare
                                 Step_Pos : Natural := Steps_Pos + 1;
                                 Step_Start, Step_End : Natural;
                                 Func_Idx : constant Positive := Module.Func_Count;
                              begin
                                 while Module.Functions (Func_Idx).Step_Count < Max_Steps loop
                                    Get_Next_Object (Func_JSON, Step_Pos, Step_Start, Step_End);
                                    exit when Step_Start = 0 or Step_End = 0;
                                    
                                    declare
                                       Step_JSON : constant String := Func_JSON (Step_Start .. Step_End);
                                       Step_Op   : constant String := Extract_String_Value (Step_JSON, "op");
                                       Step_Tgt  : constant String := Extract_String_Value (Step_JSON, "target");
                                       Step_Val  : constant String := Extract_String_Value (Step_JSON, "value");
                                       Step_Cond : constant String := Extract_String_Value (Step_JSON, "condition");
                                       Step_Init : constant String := Extract_String_Value (Step_JSON, "init");
                                       Step_Incr : constant String := Extract_String_Value (Step_JSON, "increment");
                                       --  Extract block indices for flattened IR (v0.6.1)
                                       Block_Start_Val : constant Natural := Extract_Integer_Value (Step_JSON, "block_start");
                                       Block_Count_Val : constant Natural := Extract_Integer_Value (Step_JSON, "block_count");
                                       Else_Start_Val  : constant Natural := Extract_Integer_Value (Step_JSON, "else_start");
                                       Else_Count_Val  : constant Natural := Extract_Integer_Value (Step_JSON, "else_count");
                                    begin
                                       if Step_Op'Length > 0 then
                                          Module.Functions (Func_Idx).Step_Count := 
                                            Module.Functions (Func_Idx).Step_Count + 1;
                                          
                                          Module.Functions (Func_Idx).Steps (Module.Functions (Func_Idx).Step_Count).Op :=
                                            Name_Strings.To_Bounded_String (Step_Op);
                                          
                                          if Step_Tgt'Length > 0 then
                                             Module.Functions (Func_Idx).Steps (Module.Functions (Func_Idx).Step_Count).Target :=
                                               Name_Strings.To_Bounded_String (Step_Tgt);
                                          else
                                             Module.Functions (Func_Idx).Steps (Module.Functions (Func_Idx).Step_Count).Target :=
                                               Name_Strings.Null_Bounded_String;
                                          end if;
                                          
                                          if Step_Val'Length > 0 then
                                             Module.Functions (Func_Idx).Steps (Module.Functions (Func_Idx).Step_Count).Value :=
                                               Name_Strings.To_Bounded_String (Step_Val);
                                          else
                                             Module.Functions (Func_Idx).Steps (Module.Functions (Func_Idx).Step_Count).Value :=
                                               Name_Strings.Null_Bounded_String;
                                          end if;
                                          
                                          --  Parse control flow fields
                                          if Step_Cond'Length > 0 then
                                             Module.Functions (Func_Idx).Steps (Module.Functions (Func_Idx).Step_Count).Condition :=
                                               Name_Strings.To_Bounded_String (Step_Cond);
                                          else
                                             Module.Functions (Func_Idx).Steps (Module.Functions (Func_Idx).Step_Count).Condition :=
                                               Name_Strings.Null_Bounded_String;
                                          end if;
                                          
                                          if Step_Init'Length > 0 then
                                             Module.Functions (Func_Idx).Steps (Module.Functions (Func_Idx).Step_Count).Init :=
                                               Name_Strings.To_Bounded_String (Step_Init);
                                          else
                                             Module.Functions (Func_Idx).Steps (Module.Functions (Func_Idx).Step_Count).Init :=
                                               Name_Strings.Null_Bounded_String;
                                          end if;
                                          
                                          if Step_Incr'Length > 0 then
                                             Module.Functions (Func_Idx).Steps (Module.Functions (Func_Idx).Step_Count).Increment :=
                                               Name_Strings.To_Bounded_String (Step_Incr);
                                          else
                                             Module.Functions (Func_Idx).Steps (Module.Functions (Func_Idx).Step_Count).Increment :=
                                               Name_Strings.Null_Bounded_String;
                                          end if;
                                          
                                          --  Store block indices for flattened IR (v0.6.1)
                                          Module.Functions (Func_Idx).Steps (Module.Functions (Func_Idx).Step_Count).Block_Start := Block_Start_Val;
                                          Module.Functions (Func_Idx).Steps (Module.Functions (Func_Idx).Step_Count).Block_Count := Block_Count_Val;
                                          Module.Functions (Func_Idx).Steps (Module.Functions (Func_Idx).Step_Count).Else_Start := Else_Start_Val;
                                           
                                           --  Parse switch/case fields (v0.9.0)
                                           if Step_Op = "switch" then
                                              declare
                                                 Expr_Val : constant String := Extract_String_Value (Step_JSON, "expr");
                                                 Cases_Pos : constant Natural := Find_Array (Step_JSON, "cases");
                                                 Default_Start_Val : constant Natural := Extract_Integer_Value (Step_JSON, "default_start");
                                                 Default_Count_Val : constant Natural := Extract_Integer_Value (Step_JSON, "default_count");
                                                 Current_Step_Idx : constant Positive := Module.Functions (Func_Idx).Step_Count;
                                              begin
                                                 --  Store switch expression in Value field
                                                 if Expr_Val'Length > 0 then
                                                    Module.Functions (Func_Idx).Steps (Current_Step_Idx).Value :=
                                                      Name_Strings.To_Bounded_String (Expr_Val);
                                                 end if;
                                                 
                                                 --  Parse cases array
                                                 Module.Functions (Func_Idx).Steps (Current_Step_Idx).Case_Count := 0;
                                                 if Cases_Pos > 0 then
                                                    declare
                                                       Case_Pos : Natural := Cases_Pos + 1;
                                                       Case_Start, Case_End : Natural;
                                                    begin
                                                       while Module.Functions (Func_Idx).Steps (Current_Step_Idx).Case_Count < Max_Switch_Cases loop
                                                          Get_Next_Object (Step_JSON, Case_Pos, Case_Start, Case_End);
                                                          exit when Case_Start = 0 or Case_End = 0;
                                                          
                                                           declare
                                                              Case_JSON : constant String := Step_JSON (Case_Start .. Case_End);
                                                              Case_Val_Int : constant Natural := Extract_Integer_Value (Case_JSON, "value");
                                                              Case_Val_Str : constant String := Extract_String_Value (Case_JSON, "value");
                                                              Case_Block_Start : constant Natural := Extract_Integer_Value (Case_JSON, "block_start");
                                                              Case_Block_Count : constant Natural := Extract_Integer_Value (Case_JSON, "block_count");
                                                              Case_Idx : Positive;
                                                           begin
                                                              Module.Functions (Func_Idx).Steps (Current_Step_Idx).Case_Count :=
                                                                Module.Functions (Func_Idx).Steps (Current_Step_Idx).Case_Count + 1;
                                                              Case_Idx := Module.Functions (Func_Idx).Steps (Current_Step_Idx).Case_Count;
                                                              
                                                              --  v0.8.5: Handle both integer and string case values
                                                              if Case_Val_Str'Length > 0 then
                                                                 --  String value (e.g., "case_a")
                                                                 Module.Functions (Func_Idx).Steps (Current_Step_Idx).Cases (Case_Idx).Case_Value :=
                                                                   Name_Strings.To_Bounded_String (Case_Val_Str);
                                                              else
                                                                 --  Integer value (convert to string)
                                                                 declare
                                                                    Val_Img : constant String := Natural'Image (Case_Val_Int);
                                                                    --  Trim leading space from 'Image
                                                                    Val_Trimmed : constant String := (if Val_Img(Val_Img'First) = ' ' 
                                                                                                      then Val_Img (Val_Img'First + 1 .. Val_Img'Last) 
                                                                                                      else Val_Img);
                                                                 begin
                                                                    Module.Functions (Func_Idx).Steps (Current_Step_Idx).Cases (Case_Idx).Case_Value :=
                                                                      Name_Strings.To_Bounded_String (Val_Trimmed);
                                                                 end;
                                                              end if;
                                                             Module.Functions (Func_Idx).Steps (Current_Step_Idx).Cases (Case_Idx).Block_Start :=
                                                               Case_Block_Start;
                                                             Module.Functions (Func_Idx).Steps (Current_Step_Idx).Cases (Case_Idx).Block_Count :=
                                                               Case_Block_Count;
                                                          end;
                                                          
                                                          Case_Pos := Case_End + 1;
                                                       end loop;
                                                    end;
                                                 end if;
                                                 
                                                 --  Store default block indices
                                                 Module.Functions (Func_Idx).Steps (Current_Step_Idx).Default_Start := Default_Start_Val;
                                                 Module.Functions (Func_Idx).Steps (Current_Step_Idx).Default_Count := Default_Count_Val;
                                              end;
                                           end if;
                                          Module.Functions (Func_Idx).Steps (Module.Functions (Func_Idx).Step_Count).Else_Count := Else_Count_Val;
                                       end if;
                                    end;
                                    
                                    Step_Pos := Step_End + 1;
                                 end loop;
                              end;
                           end if;
                        end;
                        end if;
                     end if;
                  end;
                  
                  Func_Pos := Obj_End + 1;
               end loop;
            end;
         else
            --  No functions found, create empty module
            Module.Func_Count := 0;
         end if;
      end;

      Success := True;
      Put_Line ("[SUCCESS] IR parsed successfully with " & Natural'Image (Module.Func_Count) & " function(s)");

   exception
      when others =>
         if Is_Open (File) then
            Close (File);
         end if;
         Put_Line ("[ERROR] Failed to parse IR file");
         Success := False;
   end Parse_IR;

   --  Load template for target language
   procedure Load_Template
     (Template_Path : Path_String;
      Target        : Target_Language;
      Template      : out Path_String;
      Success       : out Boolean)
   is
      pragma Unreferenced (Target);
      Template_Dir : constant String := Path_Strings.To_String (Template_Path);
   begin
      Template := Path_Strings.Null_Bounded_String;
      Success := Exists (Template_Dir);

      if Success then
         Put_Line ("[INFO] Template directory found: " & Template_Dir);
         Template := Template_Path;
      else
         Put_Line ("[WARN] Template directory not found, using defaults");
         Success := True;  --  Continue with default templates
      end if;
   end Load_Template;

   --  Emit Python function
   procedure Emit_Python_Function
     (Func   : Function_Definition;
      File   : in out File_Type)
   is
   begin
      Put (File, "def " & Name_Strings.To_String (Func.Name) & "(");

      for I in 1 .. Func.Param_Count loop
         if I > 1 then
            Put (File, ", ");
         end if;
         Put (File, Name_Strings.To_String (Func.Params (I).Name));
         Put (File, ": " & Name_Strings.To_String (Func.Params (I).Type_Name));
      end loop;

      Put_Line (File, ") -> " & Name_Strings.To_String (Func.Return_Type) & ":");
      Put_Line (File, "    pass  # TODO: Implement");
      New_Line (File);
   end Emit_Python_Function;

   --  Emit Rust function
   procedure Emit_Rust_Function
     (Func   : Function_Definition;
      File   : in out File_Type)
   is
   begin
      if Func.Is_Public then
         Put (File, "pub ");
      end if;
      Put (File, "fn " & Name_Strings.To_String (Func.Name) & "(");

      for I in 1 .. Func.Param_Count loop
         if I > 1 then
            Put (File, ", ");
         end if;
         Put (File, Name_Strings.To_String (Func.Params (I).Name));
         Put (File, ": " & Name_Strings.To_String (Func.Params (I).Type_Name));
      end loop;

      Put_Line (File, ") -> " & Name_Strings.To_String (Func.Return_Type) & " {");
      Put_Line (File, "    todo!()  // TODO: Implement");
      Put_Line (File, "}");
      New_Line (File);
   end Emit_Rust_Function;

   --  Map STUNIR IR type to C type
   function Map_To_C_Type (IR_Type : String) return String is
   begin
      if IR_Type = "i8" then return "int8_t";
      elsif IR_Type = "i16" then return "int16_t";
      elsif IR_Type = "i32" then return "int32_t";
      elsif IR_Type = "i64" then return "int64_t";
      elsif IR_Type = "u8" then return "uint8_t";
      elsif IR_Type = "u16" then return "uint16_t";
      elsif IR_Type = "u32" then return "uint32_t";
      elsif IR_Type = "u64" then return "uint64_t";
      elsif IR_Type = "f32" then return "float";
      elsif IR_Type = "f64" then return "double";
      elsif IR_Type = "bool" then return "bool";
      elsif IR_Type = "void" then return "void";
      elsif IR_Type = "byte[]" then return "uint8_t*";
      elsif IR_Type = "char*" or IR_Type = "string" then return "char*";
      else return IR_Type;  -- Pass through unknown types
      end if;
   end Map_To_C_Type;

   --  Get default return value for C type
   function C_Default_Return (IR_Type : String) return String is
   begin
      if IR_Type = "void" then
         return "";
      elsif IR_Type = "bool" then
         return "false";
      elsif IR_Type = "f32" or IR_Type = "f64" or IR_Type = "float" or IR_Type = "double" then
         return "0.0";
      elsif IR_Type = "string" or IR_Type = "char*" or IR_Type = "byte[]" then
         return "NULL";
      else
         return "0";  -- Integer types
      end if;
   end C_Default_Return;

   --  Infer C type from value string (simple heuristic-based type inference)
   function Infer_C_Type_From_Value (Value : String) return String is
   begin
      --  Boolean values
      if Value = "true" or Value = "false" then
         return "bool";
      end if;
      
      --  Floating point (contains decimal point)
      if (for some C of Value => C = '.') then
         return "double";
      end if;
      
      --  Check if it's a negative integer
      if Value'Length > 0 and then Value (Value'First) = '-' then
         --  Check remaining chars are digits
         if (for all I in Value'First + 1 .. Value'Last => Value (I) in '0' .. '9') then
            return "int32_t";
         end if;
      end if;
      
      --  Check if it's a positive integer
      if (for all C of Value => C in '0' .. '9') then
         --  Small positive integers could be uint8_t
         declare
            Num : Natural := 0;
         begin
            for C of Value loop
               Num := Num * 10 + (Character'Pos (C) - Character'Pos ('0'));
               exit when Num > 255;
            end loop;
            
            if Num <= 255 then
               return "uint8_t";
            else
               return "int32_t";
            end if;
         exception
            when others =>
               return "int32_t";
         end;
      end if;
      
      --  Default to int32_t for complex expressions
      return "int32_t";
   end Infer_C_Type_From_Value;

   --  Translate IR steps to C code (v0.7.0 - RECURSIVE with bounded depth)
   --  This function now supports multi-level nesting (up to Max_Recursion_Depth = 5)
   function Translate_Steps_To_C 
     (Steps      : Step_Array;
      Step_Count : Natural;
      Ret_Type   : String;
      Depth      : Recursion_Depth := 1;
      Indent     : Natural := 1) return String
   is
      Max_Body_Size : constant := 32768;  -- Increased for 5-level nesting
      Result        : String (1 .. Max_Body_Size);
      Result_Len    : Natural := 0;
      NL            : constant String := [1 => Character'Val (10)];
      
      --  Local variable tracking (simplified - just tracks if var was declared)
      Max_Vars      : constant := 20;
      Local_Vars    : array (1 .. Max_Vars) of Name_String;
      Local_Types   : array (1 .. Max_Vars) of Name_String;
      Var_Count     : Natural := 0;
      
      --  Generate indentation string (v0.7.0)
      function Get_Indent return String is
         Spaces_Per_Level : constant := 2;
         Total_Spaces     : constant Natural := Indent * Spaces_Per_Level;
         Indent_Str       : constant String (1 .. Total_Spaces) := [others => ' '];
      begin
         if Total_Spaces > 0 then
            return Indent_Str;
         else
            return "";
         end if;
      end Get_Indent;
      
      procedure Append (S : String) is
      begin
         if Result_Len + S'Length <= Result'Last then
            Result (Result_Len + 1 .. Result_Len + S'Length) := S;
            Result_Len := Result_Len + S'Length;
         end if;
      end Append;
      
      function Is_Var_Declared (Var_Name : String) return Boolean is
      begin
         for I in 1 .. Var_Count loop
            if Name_Strings.To_String (Local_Vars (I)) = Var_Name then
               return True;
            end if;
         end loop;
         return False;
      end Is_Var_Declared;
      
      procedure Declare_Var (Var_Name : String; Var_Type : String) is
      begin
         if Var_Count < Max_Vars and not Is_Var_Declared (Var_Name) then
            Var_Count := Var_Count + 1;
            Local_Vars (Var_Count) := Name_Strings.To_Bounded_String (Var_Name);
            Local_Types (Var_Count) := Name_Strings.To_Bounded_String (Var_Type);
         end if;
      end Declare_Var;
      
      --  Track which steps are part of nested blocks (v0.6.1 -> v0.7.0)
      type Step_Processed_Array is array (1 .. Max_Steps) of Boolean;
      Processed : Step_Processed_Array := [others => False];
      
      Has_Return : Boolean := False;
   begin
      --  Check recursion depth (v0.7.0 - Bounded Recursion)
      if Depth > Max_Recursion_Depth then
         raise Recursion_Depth_Exceeded with 
           "Maximum recursion depth (" & Natural'Image (Max_Recursion_Depth) & ") exceeded at depth " & Natural'Image (Depth);
      end if;
      
      --  Handle empty function body
      if Step_Count = 0 then
         Append (Get_Indent & "/* Empty function body */");
         Append (NL);
         if Ret_Type = "void" then
            Append (Get_Indent & "return;");
         else
            Append (Get_Indent & "return " & C_Default_Return (Ret_Type) & ";");
         end if;
         return Result (1 .. Result_Len);
      end if;
      
      --  Mark all steps that are part of nested blocks
      for I in 1 .. Step_Count loop
         declare
            Step : constant IR_Step := Steps (I);
            Op   : constant String := Name_Strings.To_String (Step.Op);
         begin
            if Op = "if" or Op = "while" or Op = "for" then
               --  Mark then/body block steps as processed
               if Step.Block_Count > 0 and Step.Block_Start > 0 then
                  for J in Step.Block_Start .. Step.Block_Start + Step.Block_Count - 1 loop
                     if J <= Max_Steps then
                        Processed (J) := True;
                     end if;
                  end loop;
               end if;
               
               --  Mark else block steps as processed (if statements)
               if Step.Else_Count > 0 and Step.Else_Start > 0 then
                  for J in Step.Else_Start .. Step.Else_Start + Step.Else_Count - 1 loop
                     if J <= Max_Steps then
                        Processed (J) := True;
                     end if;
                  end loop;
               end if;
            end if;
         end;
      end loop;
      
      --  Process each step (skip steps that are part of nested blocks)
      for I in 1 .. Step_Count loop
         if not Processed (I) then
            declare
               Step   : constant IR_Step := Steps (I);
               Op     : constant String := Name_Strings.To_String (Step.Op);
               Target : constant String := Name_Strings.To_String (Step.Target);
               Value  : constant String := Name_Strings.To_String (Step.Value);
            begin
            if Op = "assign" then
                --  Variable assignment
                if Target'Length > 0 then
                   --  v0.8.5: In nested blocks (Depth > 1), don't redeclare variables
                   --  Assume they're declared in parent scope
                   if Depth > 1 then
                      --  Nested block - just assign without declaration
                      Append (Get_Indent & Target & " = " & Value & ";");
                      Append (NL);
                   elsif not Is_Var_Declared (Target) then
                      --  First use in top-level function - declare with type inference
                      declare
                         Var_Type : constant String := Infer_C_Type_From_Value (Value);
                      begin
                         Declare_Var (Target, Var_Type);
                         Append ("  " & Var_Type & " " & Target & " = " & Value & ";");
                         Append (NL);
                      end;
                   else
                      --  Already declared - just assign
                      Append ("  " & Target & " = " & Value & ";");
                      Append (NL);
                   end if;
                end if;
               
            elsif Op = "return" then
               --  Return statement
               Has_Return := True;
               if Value'Length > 0 then
                  Append ("  return " & Value & ";");
               elsif Ret_Type = "void" then
                  Append (Get_Indent & "return;");
               else
                  Append ("  return " & C_Default_Return (Ret_Type) & ";");
               end if;
               Append (NL);
               
            elsif Op = "call" then
               --  Function call
               --  Get the function call expression from value field
               --  Format: "function_name(arg1, arg2, ...)"
               if Target'Length > 0 then
                  --  Call with assignment
                  declare
                     Var_Exists : Boolean := False;
                  begin
                     --  Check if variable already declared
                     for J in 1 .. Var_Count loop
                        if Name_Strings.To_String (Local_Vars (J)) = Target then
                           Var_Exists := True;
                           exit;
                        end if;
                     end loop;
                     
                     if not Var_Exists and Var_Count < Max_Vars then
                        --  Declare new variable with default type int32_t
                        Var_Count := Var_Count + 1;
                        Local_Vars (Var_Count) := Name_Strings.To_Bounded_String (Target);
                        Local_Types (Var_Count) := Name_Strings.To_Bounded_String ("int32_t");
                        Append ("  int32_t " & Target & " = " & Value & ";");
                     else
                        --  Variable already declared, just assign
                        Append ("  " & Target & " = " & Value & ";");
                     end if;
                  end;
               else
                  --  Call without assignment
                  Append ("  " & Value & ";");
               end if;
               Append (NL);
               
            elsif Op = "nop" then
               --  No operation
               Append (Get_Indent & "/* nop */");
               Append (NL);
               
            elsif Op = "if" then
               --  If/else statement with RECURSIVE multi-level nesting (v0.7.1)
               declare
                  Cond : constant String := Name_Strings.To_String (Step.Condition);
               begin
                  Append (Get_Indent & "if (" & Cond & ") {");
                  Append (NL);
                  
                  --  Process then block RECURSIVELY using block indices
                   if Step.Block_Count > 0 and Step.Block_Start > 0 and Step.Block_Start <= Step_Count then
                      declare
                         Then_Block_Start : constant Natural := Step.Block_Start;
                         Then_Block_End   : Natural := Step.Block_Start + Step.Block_Count - 1;
                         Then_Steps       : Step_Array;
                         Then_Count       : Natural := 0;
                      begin
                         --  v0.8.5: Validate block end doesn't exceed step count
                         if Then_Block_End > Step_Count then
                            Then_Block_End := Step_Count;
                         end if;
                         
                         --  Extract sub-array for then block WITH INDEX ADJUSTMENT (v0.8.5)
                         for Block_I in Then_Block_Start .. Then_Block_End loop
                            if Block_I <= Step_Count then
                               Then_Count := Then_Count + 1;
                               Then_Steps (Then_Count) := Steps (Block_I);
                               
                               --  Adjust block indices to be relative to the extracted sub-array
                               --  Only adjust if the index is within the extracted range
                               if Then_Steps (Then_Count).Block_Start >= Then_Block_Start 
                                 and Then_Steps (Then_Count).Block_Start <= Then_Block_End then
                                  Then_Steps (Then_Count).Block_Start := 
                                    Then_Steps (Then_Count).Block_Start - Then_Block_Start + 1;
                               else
                                  Then_Steps (Then_Count).Block_Start := 0;  --  Out of range, disable
                               end if;
                               
                               if Then_Steps (Then_Count).Else_Start >= Then_Block_Start 
                                 and Then_Steps (Then_Count).Else_Start <= Then_Block_End then
                                  Then_Steps (Then_Count).Else_Start := 
                                    Then_Steps (Then_Count).Else_Start - Then_Block_Start + 1;
                               else
                                  Then_Steps (Then_Count).Else_Start := 0;  --  Out of range, disable
                               end if;
                            end if;
                         end loop;
                         
                         --  RECURSIVE CALL for then block
                         if Then_Count > 0 then
                            declare
                               Nested_Body : constant String := 
                                 Translate_Steps_To_C (Then_Steps, Then_Count, Ret_Type, Depth + 1, Indent + 1);
                            begin
                               Append (Nested_Body);
                            end;
                         end if;
                      end;
                   end if;
                  
                  --  Process else block if present RECURSIVELY
                   if Step.Else_Count > 0 and Step.Else_Start > 0 and Step.Else_Start <= Step_Count then
                      Append (Get_Indent & "} else {");
                      Append (NL);
                      
                      declare
                         Else_Block_Start : constant Natural := Step.Else_Start;
                         Else_Block_End   : Natural := Step.Else_Start + Step.Else_Count - 1;
                         Else_Steps       : Step_Array;
                         Else_Count       : Natural := 0;
                      begin
                         --  v0.8.5: Validate block end doesn't exceed step count
                         if Else_Block_End > Step_Count then
                            Else_Block_End := Step_Count;
                         end if;
                         
                         --  Extract sub-array for else block WITH INDEX ADJUSTMENT (v0.8.5)
                         for Block_I in Else_Block_Start .. Else_Block_End loop
                            if Block_I <= Step_Count then
                               Else_Count := Else_Count + 1;
                               Else_Steps (Else_Count) := Steps (Block_I);
                               
                               --  Adjust block indices to be relative to the extracted sub-array
                               --  Only adjust if the index is within the extracted range
                               if Else_Steps (Else_Count).Block_Start >= Else_Block_Start 
                                 and Else_Steps (Else_Count).Block_Start <= Else_Block_End then
                                  Else_Steps (Else_Count).Block_Start := 
                                    Else_Steps (Else_Count).Block_Start - Else_Block_Start + 1;
                               else
                                  Else_Steps (Else_Count).Block_Start := 0;  --  Out of range, disable
                               end if;
                               
                               if Else_Steps (Else_Count).Else_Start >= Else_Block_Start 
                                 and Else_Steps (Else_Count).Else_Start <= Else_Block_End then
                                  Else_Steps (Else_Count).Else_Start := 
                                    Else_Steps (Else_Count).Else_Start - Else_Block_Start + 1;
                               else
                                  Else_Steps (Else_Count).Else_Start := 0;  --  Out of range, disable
                               end if;
                            end if;
                         end loop;
                         
                         --  RECURSIVE CALL for else block
                         if Else_Count > 0 then
                            declare
                               Nested_Body : constant String := 
                                 Translate_Steps_To_C (Else_Steps, Else_Count, Ret_Type, Depth + 1, Indent + 1);
                            begin
                               Append (Nested_Body);
                            end;
                         end if;
                      end;
                   end if;
                  Append (Get_Indent & "}");
                  Append (NL);
               end;
               
            elsif Op = "while" then
               --  While loop with RECURSIVE multi-level nesting (v0.7.1)
               declare
                  Cond : constant String := Name_Strings.To_String (Step.Condition);
               begin
                  Append (Get_Indent & "while (" & Cond & ") {");
                  Append (NL);
                  
                  --  Process body RECURSIVELY using block indices
                   if Step.Block_Count > 0 and Step.Block_Start > 0 and Step.Block_Start <= Step_Count then
                      declare
                         Body_Block_Start : constant Natural := Step.Block_Start;
                         Body_Block_End   : Natural := Step.Block_Start + Step.Block_Count - 1;
                         Body_Steps       : Step_Array;
                         Body_Count       : Natural := 0;
                      begin
                         --  v0.8.5: Validate block end doesn't exceed step count
                         if Body_Block_End > Step_Count then
                            Body_Block_End := Step_Count;
                         end if;
                         
                         --  Extract sub-array for loop body WITH INDEX ADJUSTMENT (v0.8.5)
                         for Block_I in Body_Block_Start .. Body_Block_End loop
                            if Block_I <= Step_Count then
                               Body_Count := Body_Count + 1;
                               Body_Steps (Body_Count) := Steps (Block_I);
                               
                               --  Adjust block indices to be relative to the extracted sub-array
                               --  Only adjust if the index is within the extracted range
                               if Body_Steps (Body_Count).Block_Start >= Body_Block_Start 
                                 and Body_Steps (Body_Count).Block_Start <= Body_Block_End then
                                  Body_Steps (Body_Count).Block_Start := 
                                    Body_Steps (Body_Count).Block_Start - Body_Block_Start + 1;
                               else
                                  Body_Steps (Body_Count).Block_Start := 0;  --  Out of range, disable
                               end if;
                               
                               if Body_Steps (Body_Count).Else_Start >= Body_Block_Start 
                                 and Body_Steps (Body_Count).Else_Start <= Body_Block_End then
                                  Body_Steps (Body_Count).Else_Start := 
                                    Body_Steps (Body_Count).Else_Start - Body_Block_Start + 1;
                               else
                                  Body_Steps (Body_Count).Else_Start := 0;  --  Out of range, disable
                               end if;
                            end if;
                         end loop;
                         
                         --  RECURSIVE CALL for loop body
                         if Body_Count > 0 then
                            declare
                               Nested_Body : constant String := 
                                 Translate_Steps_To_C (Body_Steps, Body_Count, Ret_Type, Depth + 1, Indent + 1);
                            begin
                               Append (Nested_Body);
                            end;
                         end if;
                      end;
                   end if;
                  
                  Append (Get_Indent & "}");
                  Append (NL);
               end;
               
            elsif Op = "for" then
                --  For loop with RECURSIVE multi-level nesting (v0.7.1)
                declare
                   Init_Expr : constant String := Name_Strings.To_String (Step.Init);
                   Cond      : constant String := Name_Strings.To_String (Step.Condition);
                   Incr      : constant String := Name_Strings.To_String (Step.Increment);
                begin
                   --  v0.8.5: Extract and declare loop variable from init expression
                   --  Handle cases like "i = 0" or "int i = 0"
                   if Init_Expr'Length > 0 then
                      declare
                         Eq_Pos : Natural := 0;
                         Loop_Var : String (1 .. 64);
                         Loop_Var_Len : Natural := 0;
                      begin
                         --  Find the '=' sign
                         for I in Init_Expr'Range loop
                            if Init_Expr (I) = '=' then
                               Eq_Pos := I;
                               exit;
                            end if;
                         end loop;
                         
                         if Eq_Pos > Init_Expr'First then
                            --  Extract variable name (everything before '=')
                            declare
                               Before_Eq : constant String := Init_Expr (Init_Expr'First .. Eq_Pos - 1);
                               Start_Idx : Natural := Before_Eq'First;
                            begin
                               --  Skip leading whitespace and type keywords
                               --  Look for the last token before '='
                               for I in reverse Before_Eq'Range loop
                                  if Before_Eq (I) /= ' ' then
                                     --  Found end of variable name
                                     declare
                                        End_Idx : constant Natural := I;
                                     begin
                                        --  Find start of variable name (scan backwards)
                                        Start_Idx := End_Idx;
                                        for J in reverse Before_Eq'First .. End_Idx loop
                                           if Before_Eq (J) = ' ' then
                                              Start_Idx := J + 1;
                                              exit;
                                           end if;
                                           Start_Idx := J;
                                        end loop;
                                        
                                        --  Extract the variable name
                                        if Start_Idx <= End_Idx and (End_Idx - Start_Idx + 1) <= Loop_Var'Length then
                                           Loop_Var_Len := End_Idx - Start_Idx + 1;
                                           Loop_Var (1 .. Loop_Var_Len) := Before_Eq (Start_Idx .. End_Idx);
                                           
                                           --  Declare variable if not already declared
                                           if not Is_Var_Declared (Loop_Var (1 .. Loop_Var_Len)) then
                                              Declare_Var (Loop_Var (1 .. Loop_Var_Len), "int32_t");
                                              Append (Get_Indent & "int32_t " & Loop_Var (1 .. Loop_Var_Len) & ";");
                                              Append (NL);
                                           end if;
                                        end if;
                                     end;
                                     exit;
                                  end if;
                               end loop;
                            end;
                         end if;
                      end;
                   end if;
                   
                   Append (Get_Indent & "for (" & Init_Expr & "; " & Cond & "; " & Incr & ") {");
                   Append (NL);
                  
                   --  Process body RECURSIVELY using block indices
                   if Step.Block_Count > 0 and Step.Block_Start > 0 and Step.Block_Start <= Step_Count then
                      declare
                         Body_Block_Start : constant Natural := Step.Block_Start;
                         Body_Block_End   : Natural := Step.Block_Start + Step.Block_Count - 1;
                         Body_Steps       : Step_Array;
                         Body_Count       : Natural := 0;
                      begin
                         --  v0.8.5: Validate block end doesn't exceed step count
                         if Body_Block_End > Step_Count then
                            Body_Block_End := Step_Count;
                         end if;
                         
                         --  Extract sub-array for loop body WITH INDEX ADJUSTMENT (v0.8.5)
                         for Block_I in Body_Block_Start .. Body_Block_End loop
                            if Block_I <= Step_Count then
                               Body_Count := Body_Count + 1;
                               Body_Steps (Body_Count) := Steps (Block_I);
                               
                               --  Adjust block indices to be relative to the extracted sub-array
                               --  Only adjust if the index is within the extracted range
                               if Body_Steps (Body_Count).Block_Start >= Body_Block_Start 
                                 and Body_Steps (Body_Count).Block_Start <= Body_Block_End then
                                  Body_Steps (Body_Count).Block_Start := 
                                    Body_Steps (Body_Count).Block_Start - Body_Block_Start + 1;
                               else
                                  Body_Steps (Body_Count).Block_Start := 0;  --  Out of range, disable
                               end if;
                               
                               if Body_Steps (Body_Count).Else_Start >= Body_Block_Start 
                                 and Body_Steps (Body_Count).Else_Start <= Body_Block_End then
                                  Body_Steps (Body_Count).Else_Start := 
                                    Body_Steps (Body_Count).Else_Start - Body_Block_Start + 1;
                               else
                                  Body_Steps (Body_Count).Else_Start := 0;  --  Out of range, disable
                               end if;
                            end if;
                         end loop;
                         
                         --  RECURSIVE CALL for loop body
                         if Body_Count > 0 then
                            declare
                               Nested_Body : constant String := 
                                 Translate_Steps_To_C (Body_Steps, Body_Count, Ret_Type, Depth + 1, Indent + 1);
                            begin
                               Append (Nested_Body);
                            end;
                         end if;
                      end;
                   end if;
                  
                  Append (Get_Indent & "}");
                  Append (NL);
               end;
               
            elsif Op = "break" then
               --  v0.9.0: Break statement
               Append (Get_Indent & "break;");
               Append (NL);
               
            elsif Op = "continue" then
               --  v0.9.0: Continue statement
               Append (Get_Indent & "continue;");
               Append (NL);
               
            elsif Op = "switch" then
               --  v0.9.0: Switch/case statement
               declare
                  Expr : constant String := Name_Strings.To_String (Step.Value);
               begin
                  Append (Get_Indent & "switch (" & Expr & ") {");
                  Append (NL);
                  
                  --  Generate case labels
                   for Case_Idx in 1 .. Step.Case_Count loop
                      declare
                         Case_Val : constant String := Name_Strings.To_String (Step.Cases (Case_Idx).Case_Value);
                         Case_Block_Start : constant Natural := Step.Cases (Case_Idx).Block_Start;
                         Case_Block_End   : Natural := Step.Cases (Case_Idx).Block_Start + Step.Cases (Case_Idx).Block_Count - 1;
                         Case_Steps       : Step_Array;
                         Case_Step_Count  : Natural := 0;
                      begin
                         Append (Get_Indent & "  case " & Case_Val & ":");
                         Append (NL);
                         
                         --  v0.8.5: Validate block bounds
                         if Case_Block_Start > 0 and Case_Block_Start <= Step_Count then
                            if Case_Block_End > Step_Count then
                               Case_Block_End := Step_Count;
                            end if;
                            
                            --  Mark case block steps as processed
                            for J in Case_Block_Start .. Case_Block_End loop
                               if J <= Step_Count then
                                  Processed (J) := True;
                               end if;
                            end loop;
                            
                            --  Extract and translate case body RECURSIVELY
                            if Step.Cases (Case_Idx).Block_Count > 0 then
                               for Block_I in Case_Block_Start .. Case_Block_End loop
                                  if Block_I <= Step_Count then
                                     Case_Step_Count := Case_Step_Count + 1;
                                     Case_Steps (Case_Step_Count) := Steps (Block_I);
                                  end if;
                               end loop;
                               
                               if Case_Step_Count > 0 then
                                  declare
                                     Case_Code : constant String := 
                                       Translate_Steps_To_C (Case_Steps, Case_Step_Count, Ret_Type, Depth + 1, Indent + 2);
                                  begin
                                     Append (Case_Code);
                                  end;
                               end if;
                            end if;
                         end if;
                      end;
                   end loop;
                  
                   --  Generate default case if present
                   if Step.Default_Count > 0 and Step.Default_Start > 0 and Step.Default_Start <= Step_Count then
                      declare
                         Default_Block_Start : constant Natural := Step.Default_Start;
                         Default_Block_End   : Natural := Step.Default_Start + Step.Default_Count - 1;
                         Default_Steps       : Step_Array;
                         Default_Step_Count  : Natural := 0;
                      begin
                         --  v0.8.5: Validate block end doesn't exceed step count
                         if Default_Block_End > Step_Count then
                            Default_Block_End := Step_Count;
                         end if;
                         
                         Append (Get_Indent & "  default:");
                         Append (NL);
                         
                         --  Mark default block steps as processed
                         for J in Default_Block_Start .. Default_Block_End loop
                            if J <= Step_Count then
                               Processed (J) := True;
                            end if;
                         end loop;
                         
                         --  Extract and translate default body RECURSIVELY
                         for Block_I in Default_Block_Start .. Default_Block_End loop
                            if Block_I <= Step_Count then
                               Default_Step_Count := Default_Step_Count + 1;
                               Default_Steps (Default_Step_Count) := Steps (Block_I);
                            end if;
                         end loop;
                         
                         if Default_Step_Count > 0 then
                            declare
                               Default_Code : constant String := 
                                 Translate_Steps_To_C (Default_Steps, Default_Step_Count, Ret_Type, Depth + 1, Indent + 2);
                            begin
                               Append (Default_Code);
                            end;
                         end if;
                      end;
                   end if;
                  
                  Append (Get_Indent & "}");
                  Append (NL);
               end;
               
            elsif Op = "try" then
               --  v0.8.7: try/catch/finally exception handling
               --  Note: C doesn't have native exception handling; use setjmp/longjmp pattern
               Append (Get_Indent & "/* BEGIN TRY-CATCH BLOCK */");
               Append (NL);
               Append (Get_Indent & "{");
               Append (NL);
               Append (Get_Indent & "  jmp_buf __stunir_exception_buf;");
               Append (NL);
               Append (Get_Indent & "  int __stunir_exception_code = 0;");
               Append (NL);
               Append (Get_Indent & "  if ((__stunir_exception_code = setjmp(__stunir_exception_buf)) == 0) {");
               Append (NL);
               Append (Get_Indent & "    /* TRY */");
               Append (NL);
               
               --  Process try block using block_start/block_count
               if Step.Block_Start > 0 and Step.Block_Start <= Step_Count then
                  declare
                     Try_Block_End : Natural := Step.Block_Start + Step.Block_Count - 1;
                     Try_Steps : Step_Array;
                     Try_Step_Count : Natural := 0;
                  begin
                     if Try_Block_End > Step_Count then
                        Try_Block_End := Step_Count;
                     end if;
                     
                     for J in Step.Block_Start .. Try_Block_End loop
                        if J <= Step_Count then
                           Processed (J) := True;
                           Try_Step_Count := Try_Step_Count + 1;
                           Try_Steps (Try_Step_Count) := Steps (J);
                        end if;
                     end loop;
                     
                     if Try_Step_Count > 0 then
                        declare
                           Try_Code : constant String :=
                             Translate_Steps_To_C (Try_Steps, Try_Step_Count, Ret_Type, Depth + 1, Indent + 2);
                        begin
                           Append (Try_Code);
                        end;
                     end if;
                  end;
               end if;
               
               Append (Get_Indent & "  } else {");
               Append (NL);
               Append (Get_Indent & "    /* CATCH */");
               Append (NL);
               Append (Get_Indent & "    int e = __stunir_exception_code;");
               Append (NL);
               
               --  Process else block (catch) if present
               if Step.Else_Start > 0 and Step.Else_Start <= Step_Count then
                  declare
                     Catch_Block_End : Natural := Step.Else_Start + Step.Else_Count - 1;
                     Catch_Steps : Step_Array;
                     Catch_Step_Count : Natural := 0;
                  begin
                     if Catch_Block_End > Step_Count then
                        Catch_Block_End := Step_Count;
                     end if;
                     
                     for J in Step.Else_Start .. Catch_Block_End loop
                        if J <= Step_Count then
                           Processed (J) := True;
                           Catch_Step_Count := Catch_Step_Count + 1;
                           Catch_Steps (Catch_Step_Count) := Steps (J);
                        end if;
                     end loop;
                     
                     if Catch_Step_Count > 0 then
                        declare
                           Catch_Code : constant String :=
                             Translate_Steps_To_C (Catch_Steps, Catch_Step_Count, Ret_Type, Depth + 1, Indent + 2);
                        begin
                           Append (Catch_Code);
                        end;
                     end if;
                  end;
               end if;
               
               Append (Get_Indent & "  }");
               Append (NL);
               Append (Get_Indent & "}");
               Append (NL);
               Append (Get_Indent & "/* END TRY-CATCH BLOCK */");
               Append (NL);
               
            elsif Op = "throw" then
               --  v0.8.7: throw exception
               declare
                  Exc_Type : constant String := Name_Strings.To_String (Step.Target);
                  Exc_Msg  : constant String := Name_Strings.To_String (Step.Value);
               begin
                  Append (Get_Indent & "/* throw " & Exc_Type & ": " & Exc_Msg & " */");
                  Append (NL);
                  Append (Get_Indent & "longjmp(__stunir_exception_buf, 1);");
                  Append (NL);
               end;
               
            --  v0.8.8: Data structure operations - Arrays
            elsif Op = "array_new" then
               declare
                  Target : constant String := Name_Strings.To_String (Step.Target);
               begin
                  Append (Get_Indent & "/* array_new: " & Target & " */");
                  Append (NL);
                  Append (Get_Indent & "int32_t* " & Target & " = NULL;");
                  Append (NL);
                  Append (Get_Indent & "size_t " & Target & "_len = 0;");
                  Append (NL);
                  Append (Get_Indent & "size_t " & Target & "_cap = 0;");
                  Append (NL);
               end;
               
            elsif Op = "array_get" then
               declare
                  Target : constant String := Name_Strings.To_String (Step.Target);
                  Source : constant String := Name_Strings.To_String (Step.Value);
               begin
                  Append (Get_Indent & "/* array_get: " & Target & " = " & Source & "[...] */");
                  Append (NL);
                  Append (Get_Indent & "int32_t " & Target & " = " & Source & "[0];");
                  Append (NL);
               end;
               
            elsif Op = "array_set" then
               declare
                  Target : constant String := Name_Strings.To_String (Step.Target);
                  Value  : constant String := Name_Strings.To_String (Step.Value);
               begin
                  Append (Get_Indent & "/* array_set: " & Target & "[...] = " & Value & " */");
                  Append (NL);
                  Append (Get_Indent & Target & "[0] = " & Value & ";");
                  Append (NL);
               end;
               
            elsif Op = "array_push" or Op = "array_pop" or Op = "array_len" then
               Append (Get_Indent & "/* " & Op & " */");
               Append (NL);
               
            --  v0.8.8: Data structure operations - Maps
            elsif Op = "map_new" then
               declare
                  Target : constant String := Name_Strings.To_String (Step.Target);
               begin
                  Append (Get_Indent & "/* map_new: " & Target & " */");
                  Append (NL);
                  Append (Get_Indent & "void* " & Target & " = NULL;");
                  Append (NL);
                  Append (Get_Indent & "size_t " & Target & "_cap = 0;");
                  Append (NL);
               end;
               
            elsif Op = "map_get" or Op = "map_set" or Op = "map_delete" 
                  or Op = "map_has" or Op = "map_keys" then
               Append (Get_Indent & "/* " & Op & " */");
               Append (NL);
               
            --  v0.8.8: Data structure operations - Sets
            elsif Op = "set_new" then
               declare
                  Target : constant String := Name_Strings.To_String (Step.Target);
               begin
                  Append (Get_Indent & "/* set_new: " & Target & " */");
                  Append (NL);
                  Append (Get_Indent & "void* " & Target & " = NULL;");
                  Append (NL);
                  Append (Get_Indent & "size_t " & Target & "_cap = 0;");
                  Append (NL);
               end;
               
            elsif Op = "set_add" or Op = "set_remove" or Op = "set_has" 
                  or Op = "set_union" or Op = "set_intersect" then
               Append (Get_Indent & "/* " & Op & " */");
               Append (NL);
               
            --  v0.8.8: Data structure operations - Structs
            elsif Op = "struct_new" then
               declare
                  Target : constant String := Name_Strings.To_String (Step.Target);
               begin
                  Append (Get_Indent & "/* struct_new: " & Target & " */");
                  Append (NL);
                  Append (Get_Indent & "struct Object " & Target & " = {0};");
                  Append (NL);
               end;
               
            elsif Op = "struct_get" or Op = "struct_set" then
               Append (Get_Indent & "/* " & Op & " */");
               Append (NL);
               
            else
               --  Unknown operation
               Append ("  /* UNKNOWN OP: " & Op & " */");
               Append (NL);
            end if;
         end;
         end if;  --  if not Processed (I)
      end loop;
      
      --  Add default return if no return statement was found (only at top level)
      --  Don't add default return in nested recursive calls (Depth > 1)
      if not Has_Return and Depth = 1 then
         if Ret_Type = "void" then
            Append (Get_Indent & "return;");
         else
            Append ("  return " & C_Default_Return (Ret_Type) & ";");
         end if;
      end if;
      
      return Result (1 .. Result_Len);
   end Translate_Steps_To_C;

   --  Emit C function
   procedure Emit_C_Function
     (Func   : Function_Definition;
      File   : in out File_Type)
   is
      C_Return_Type : constant String := Map_To_C_Type (Name_Strings.To_String (Func.Return_Type));
   begin
      Put (File, C_Return_Type & " ");
      Put (File, Name_Strings.To_String (Func.Name) & "(");

      if Func.Param_Count = 0 then
         Put (File, "void");
      else
         for I in 1 .. Func.Param_Count loop
            if I > 1 then
               Put (File, ", ");
            end if;
            declare
               C_Type : constant String := Map_To_C_Type (Name_Strings.To_String (Func.Params (I).Type_Name));
            begin
               Put (File, C_Type & " ");
               Put (File, Name_Strings.To_String (Func.Params (I).Name));
            end;
         end loop;
      end if;

      Put_Line (File, ") {");
      
      --  Generate function body from steps
      if Func.Step_Count > 0 then
         declare
            Body_Code : constant String := Translate_Steps_To_C 
              (Func.Steps, Func.Step_Count, Name_Strings.To_String (Func.Return_Type));
         begin
            Put_Line (File, Body_Code);
         end;
      else
         --  No steps - emit stub
         Put_Line (File, "    /* TODO: Implement */");
         if C_Return_Type = "void" then
            Put_Line (File, "    return;");
         else
            Put_Line (File, "    return " & C_Default_Return (Name_Strings.To_String (Func.Return_Type)) & ";");
         end if;
      end if;
      
      Put_Line (File, "}");
      New_Line (File);
   end Emit_C_Function;

   --  Emit code for a single function
   procedure Emit_Function
     (Func     : Function_Definition;
      Target   : Target_Language;
      Output   : out Path_String;
      Success  : out Boolean)
   is
      pragma Unreferenced (Output);
   begin
      --  This procedure is for generating function code to a string
      --  For now, mark as successful (actual emission done in Emit_Code)
      Success := True;
   end Emit_Function;

   --  Main emission procedure
   procedure Emit_Code
     (Config : Emission_Config;
      Result : out Emission_Result)
   is
      Module     : IR_Module;
      Parse_OK   : Boolean;
      Output_File : File_Type;
      Out_Name   : constant String := Path_Strings.To_String (Config.Output_Path);
      Out_Dir    : String (1 .. 512);
      Out_Dir_Len : Natural;
      Template   : Path_String;
      Templ_OK   : Boolean;
   begin
      Result := (Status            => Success,
                 Output_Path       => Config.Output_Path,
                 Output_Size       => 0,
                 Functions_Emitted => 0);

      --  Step 1: Parse IR
      Put_Line ("[INFO] Parsing IR from " & Path_Strings.To_String (Config.IR_Path));
      Parse_IR (Config.IR_Path, Module, Parse_OK);

      if not Parse_OK then
         Result.Status := Error_IR_Parse_Failed;
         return;
      end if;

      --  Step 2: Load templates
      Load_Template (Config.Template_Path, Config.Target, Template, Templ_OK);
      if not Templ_OK then
         Put_Line ("[WARN] Using built-in templates");
      end if;

      --  Step 3: Extract and create output directory if needed
      --  Handle empty paths or paths without directories
      if Out_Name'Length = 0 then
         Put_Line ("[ERROR] Output path is empty");
         Result.Status := Error_Output_Write_Failed;
         return;
      end if;
      
      --  Try to extract containing directory, with error handling
      Out_Dir_Len := 0;
      begin
         declare
            Temp_Dir : constant String := Containing_Directory (Out_Name);
         begin
            if Temp_Dir'Length > 0 and Temp_Dir'Length <= Out_Dir'Length then
               Out_Dir (1 .. Temp_Dir'Length) := Temp_Dir;
               Out_Dir_Len := Temp_Dir'Length;
               
               --  Create directory if it doesn't exist
               if not Exists (Out_Dir (1 .. Out_Dir_Len)) then
                  Put_Line ("[INFO] Creating output directory: " & Out_Dir (1 .. Out_Dir_Len));
                  Create_Directory (Out_Dir (1 .. Out_Dir_Len));
               end if;
            end if;
         end;
      exception
         when Ada.Directories.Name_Error | Ada.Directories.Use_Error =>
            --  No directory component in path (e.g., just "output.py")
            --  This is okay - use current directory
            Put_Line ("[INFO] Output path has no directory component, using current directory");
            Out_Dir_Len := 0;
      end;

      --  Step 4: Open output file
      Create (Output_File, Out_File, Out_Name);

      --  Step 5: Emit header
      if Config.Emit_Comments then
         case Config.Target is
            when Target_Python =>
               Put_Line (Output_File, "#!/usr/bin/env python3");
               Put_Line (Output_File, """""""STUNIR Generated Code");
               Put_Line (Output_File, "Generated by: " & Tool_ID & " v" & Version);
               Put_Line (Output_File, "Module: " & Name_Strings.To_String (Module.Module_Name));
               Put_Line (Output_File, """""""");

            when Target_Rust =>
               Put_Line (Output_File, "//! STUNIR Generated Code");
               Put_Line (Output_File, "//! Generated by: " & Tool_ID & " v" & Version);
               Put_Line (Output_File, "//! Module: " & Name_Strings.To_String (Module.Module_Name));

            when Target_C | Target_Cpp =>
               Put_Line (Output_File, "/* STUNIR Generated Code");
               Put_Line (Output_File, " * Generated by: " & Tool_ID & " v" & Version);
               Put_Line (Output_File, " * Module: " & Name_Strings.To_String (Module.Module_Name));
               Put_Line (Output_File, " */");
               New_Line (Output_File);
               Put_Line (Output_File, "#include <stdint.h>");
               Put_Line (Output_File, "#include <stdbool.h>");

            when others =>
               Put_Line (Output_File, "// STUNIR Generated Code");
               Put_Line (Output_File, "// Generated by: " & Tool_ID & " v" & Version);
         end case;
         New_Line (Output_File);
      end if;

      --  Step 6: Emit functions
      for I in 1 .. Module.Func_Count loop
         case Config.Target is
            when Target_Python =>
               Emit_Python_Function (Module.Functions (I), Output_File);
            when Target_Rust =>
               Emit_Rust_Function (Module.Functions (I), Output_File);
            when Target_C | Target_Cpp =>
               Emit_C_Function (Module.Functions (I), Output_File);
            when others =>
               --  Default to C-style for now
               Emit_C_Function (Module.Functions (I), Output_File);
         end case;
         Result.Functions_Emitted := Result.Functions_Emitted + 1;
      end loop;

      Close (Output_File);
      Put_Line ("[INFO] Emitted " & Natural'Image (Result.Functions_Emitted) &
                " functions to " & Out_Name);
      Result.Status := Success;

   exception
      when others =>
         if Is_Open (Output_File) then
            Close (Output_File);
         end if;
         Result.Status := Error_Output_Write_Failed;
   end Emit_Code;

   --  Entry point for command-line execution
   procedure Run_IR_To_Code
   is
      use Ada.Command_Line;
      Config        : Emission_Config;
      Result        : Emission_Result;
      IR_Path       : Path_String;
      Template_Path : Path_String := Path_Strings.To_Bounded_String ("templates");
      Output_Path   : Path_String;
      Target        : Target_Language := Target_Python;
   begin
      --  Parse command line arguments
      if Argument_Count < 4 then
         Put_Line ("STUNIR IR to Code Emitter (Ada SPARK) v" & Version);
         Put_Line ("Usage: ir_to_code --input <ir.json> --output <file> --target <lang>");
         Put_Line ("Targets: python, rust, c, cpp, go, javascript, typescript, java, csharp, wasm, x86, arm");
         Set_Exit_Status (Failure);
         return;
      end if;

      --  Simple argument parsing
      for I in 1 .. Argument_Count loop
         if Argument (I) = "--input" and I < Argument_Count then
            IR_Path := Path_Strings.To_Bounded_String (Argument (I + 1));
         elsif Argument (I) = "--output" and I < Argument_Count then
            Output_Path := Path_Strings.To_Bounded_String (Argument (I + 1));
         elsif Argument (I) = "--target" and I < Argument_Count then
            Target := Parse_Target (Argument (I + 1));
         elsif Argument (I) = "--templates" and I < Argument_Count then
            Template_Path := Path_Strings.To_Bounded_String (Argument (I + 1));
         end if;
      end loop;

      Initialize_Config
        (Config,
         Path_Strings.To_String (IR_Path),
         Path_Strings.To_String (Template_Path),
         Path_Strings.To_String (Output_Path),
         Target);

      Emit_Code (Config, Result);

      case Result.Status is
         when Success =>
            Set_Exit_Status (Ada.Command_Line.Success);
         when others =>
            Put_Line ("[ERROR] Emission failed with status: " & Emission_Status'Image (Result.Status));
            Set_Exit_Status (Ada.Command_Line.Failure);
      end case;
   end Run_IR_To_Code;

end STUNIR_IR_To_Code;