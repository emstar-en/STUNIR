--  STUNIR Code Emitter Package Body
--  Converts ir.json to target language code
--  DO-333 / SPARK 2014 compliant
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

with Semantic_IR.JSON;
with Semantic_IR.Modules;
with Semantic_IR.Types;
pragma Unreferenced (Semantic_IR.Types);  --  Needed transitively by Semantic_IR.JSON
with STUNIR.Emitters.Node_Table;
with STUNIR.Emitters.CodeGen;
with STUNIR.Emitters.Lisp;
with STUNIR.Emitters.Python;
with STUNIR.Emitters.CFamily;
with STUNIR.Emitters.Prolog_Family;
with STUNIR.Emitters.Futhark_Family;
with STUNIR.Emitters.Lean4_Family;

with Ada.Text_IO;
use Ada.Text_IO;

package body Code_Emitter is

   --  =======================================================================
   --  Internal Helper Functions
   --  =======================================================================

   procedure Append_To_Code
     (Code : in out Code_String;
      Text : in     String;
      Status : out Status_Code)
   with
      Post => (if Status = Success then
                 Code_Strings.Length (Code) = Code_Strings.Length (Code'Old) + Text'Length)
   is
   begin
      if Code_Strings.Length (Code) + Text'Length > Max_Code_Length then
         Status := Error_Too_Large;
         return;
      end if;
      Code_Strings.Append (Code, Text);
      Status := Success;
   end Append_To_Code;

   --  NOTE: To_Lower_Case / To_Upper_Case helpers removed (unused)

   --  =======================================================================
   --  Type Mapping Implementation
   --  =======================================================================

   procedure Map_Type_To_Target
     (IR_Type : in     Type_Name_String;
      Target  : in     Target_Language;
      Mapped  :    out Type_Name_String;
      Status  :    out Status_Code)
   is
      Type_Str : constant String := Type_Name_Strings.To_String (IR_Type);
   begin
      Status := Success;

      case Target is
         when Target_CPP =>
            if Type_Str = "int" then
               Mapped := Type_Name_Strings.To_Bounded_String ("int");
            elsif Type_Str = "float" then
               Mapped := Type_Name_Strings.To_Bounded_String ("double");
            elsif Type_Str = "string" then
               Mapped := Type_Name_Strings.To_Bounded_String ("std::string");
            elsif Type_Str = "bool" then
               Mapped := Type_Name_Strings.To_Bounded_String ("bool");
            elsif Type_Str = "void" then
               Mapped := Type_Name_Strings.To_Bounded_String ("void");
            elsif Type_Str = "program_counter" then
               --  Custom C type - map to appropriate C++ type
               Mapped := Type_Name_Strings.To_Bounded_String ("size_t");
            elsif Type_Str = "bc_num" then
               Mapped := Type_Name_Strings.To_Bounded_String ("void*");
            elsif Type_Str = "sigjmp_buf" then
               Mapped := Type_Name_Strings.To_Bounded_String ("jmp_buf");
            elsif Type_Str = "FILE*" then
               Mapped := Type_Name_Strings.To_Bounded_String ("FILE*");
            elsif Type_Str = "char*" then
               Mapped := Type_Name_Strings.To_Bounded_String ("char*");
            elsif Type_Str = "const char*" then
               Mapped := Type_Name_Strings.To_Bounded_String ("const char*");
            else
               --  For unknown custom types, try to map pointer syntax
               declare
                  Type_Len : constant Natural := Type_Str'Length;
               begin
                  if Type_Len > 2 and then
                     Type_Str (Type_Len - 1 .. Type_Len) = " *" then
                     --  Custom pointer type - keep as void* for safety
                     Mapped := Type_Name_Strings.To_Bounded_String ("void*");
                  else
                     Mapped := IR_Type;
                  end if;
               end;
            end if;

         when Target_Python =>
            --  Python is dynamically typed, but we keep type hints
            --  Strip C pointer syntax and map to Python types
            declare
               Type_Len : constant Natural := Type_Str'Length;
            begin
               if Type_Str = "int" or Type_Str = "i32" then
                  Mapped := Type_Name_Strings.To_Bounded_String ("int");
               elsif Type_Str = "float" or Type_Str = "double" or Type_Str = "f64" then
                  Mapped := Type_Name_Strings.To_Bounded_String ("float");
               elsif Type_Str = "string" or Type_Str = "String" then
                  Mapped := Type_Name_Strings.To_Bounded_String ("str");
               elsif Type_Str = "bool" or Type_Str = "boolean" then
                  Mapped := Type_Name_Strings.To_Bounded_String ("bool");
               elsif Type_Str = "void" then
                  Mapped := Type_Name_Strings.To_Bounded_String ("None");
               elsif Type_Str = "program_counter" then
                  Mapped := Type_Name_Strings.To_Bounded_String ("int");
               elsif Type_Len > 2 and then
                     Type_Str (Type_Len - 1 .. Type_Len) = " *" then
                  --  Pointer type - use Any from typing module
                  Mapped := Type_Name_Strings.To_Bounded_String ("Any");
               elsif Type_Str = "char*" or Type_Str = "const char*" then
                  Mapped := Type_Name_Strings.To_Bounded_String ("str");
               elsif Type_Str = "FILE*" or Type_Str = "bc_num" or Type_Str = "sigjmp_buf" then
                  Mapped := Type_Name_Strings.To_Bounded_String ("Any");
               else
                  Mapped := Type_Name_Strings.To_Bounded_String ("Any");
               end if;
            end;

         when Target_C =>
            if Type_Str = "int" then
               Mapped := Type_Name_Strings.To_Bounded_String ("int");
            elsif Type_Str = "float" then
               Mapped := Type_Name_Strings.To_Bounded_String ("double");
            elsif Type_Str = "string" then
               Mapped := Type_Name_Strings.To_Bounded_String ("char*");
            elsif Type_Str = "bool" then
               Mapped := Type_Name_Strings.To_Bounded_String ("int");
            elsif Type_Str = "void" then
               Mapped := Type_Name_Strings.To_Bounded_String ("void");
            elsif Type_Str = "program_counter" then
               Mapped := Type_Name_Strings.To_Bounded_String ("size_t");
            elsif Type_Str = "bc_num" or Type_Str = "sigjmp_buf" or Type_Str = "FILE*" then
               Mapped := IR_Type;
            else
               Mapped := IR_Type;
            end if;


         when Target_Rust =>
            if Type_Str = "int" then
               Mapped := Type_Name_Strings.To_Bounded_String ("i32");
            elsif Type_Str = "float" then
               Mapped := Type_Name_Strings.To_Bounded_String ("f64");
            elsif Type_Str = "string" then
               Mapped := Type_Name_Strings.To_Bounded_String ("String");
            elsif Type_Str = "bool" then
               Mapped := Type_Name_Strings.To_Bounded_String ("bool");
            elsif Type_Str = "void" then
               Mapped := Type_Name_Strings.To_Bounded_String ("()");
            elsif Type_Str = "program_counter" then
               Mapped := Type_Name_Strings.To_Bounded_String ("usize");
            elsif Type_Str = "bc_num" then
               Mapped := Type_Name_Strings.To_Bounded_String ("*mut c_void");
            elsif Type_Str = "sigjmp_buf" then
               Mapped := Type_Name_Strings.To_Bounded_String ("jmp_buf");
            elsif Type_Str = "FILE*" then
               Mapped := Type_Name_Strings.To_Bounded_String ("*mut FILE");
            elsif Type_Str = "char*" then
               Mapped := Type_Name_Strings.To_Bounded_String ("*mut c_char");
            elsif Type_Str = "const char*" then
               Mapped := Type_Name_Strings.To_Bounded_String ("*const c_char");
            else
               --  For unknown custom types, check for pointer syntax
               declare
                  Type_Len : constant Natural := Type_Str'Length;
               begin
                  if Type_Len > 2 and then
                     Type_Str (Type_Len - 1 .. Type_Len) = " *" then
                     --  Custom pointer type
                     Mapped := Type_Name_Strings.To_Bounded_String ("*mut c_void");
                  else
                     Mapped := IR_Type;
                  end if;
               end;
            end if;

         when Target_Go =>
            if Type_Str = "int" then
               Mapped := Type_Name_Strings.To_Bounded_String ("int");
            elsif Type_Str = "float" then
               Mapped := Type_Name_Strings.To_Bounded_String ("float64");
            elsif Type_Str = "string" then
               Mapped := Type_Name_Strings.To_Bounded_String ("string");
            elsif Type_Str = "bool" then
               Mapped := Type_Name_Strings.To_Bounded_String ("bool");
            elsif Type_Str = "void" then
               Mapped := Type_Name_Strings.To_Bounded_String ("");
            else
               Mapped := IR_Type;
            end if;

         when Target_Java =>
            if Type_Str = "int" then
               Mapped := Type_Name_Strings.To_Bounded_String ("int");
            elsif Type_Str = "float" then
               Mapped := Type_Name_Strings.To_Bounded_String ("double");
            elsif Type_Str = "string" then
               Mapped := Type_Name_Strings.To_Bounded_String ("String");
            elsif Type_Str = "bool" then
               Mapped := Type_Name_Strings.To_Bounded_String ("boolean");
            elsif Type_Str = "void" then
               Mapped := Type_Name_Strings.To_Bounded_String ("void");
            else
               Mapped := IR_Type;
            end if;

         when Target_JavaScript =>
            if Type_Str = "int" then
               Mapped := Type_Name_Strings.To_Bounded_String ("number");
            elsif Type_Str = "float" then
               Mapped := Type_Name_Strings.To_Bounded_String ("number");
            elsif Type_Str = "string" then
               Mapped := Type_Name_Strings.To_Bounded_String ("string");
            elsif Type_Str = "bool" then
               Mapped := Type_Name_Strings.To_Bounded_String ("boolean");
            elsif Type_Str = "void" then
               Mapped := Type_Name_Strings.To_Bounded_String ("void");
            else
               Mapped := IR_Type;
            end if;

         when Target_CSharp =>
            if Type_Str = "int" then
               Mapped := Type_Name_Strings.To_Bounded_String ("int");
            elsif Type_Str = "float" then
               Mapped := Type_Name_Strings.To_Bounded_String ("double");
            elsif Type_Str = "string" then
               Mapped := Type_Name_Strings.To_Bounded_String ("string");
            elsif Type_Str = "bool" then
               Mapped := Type_Name_Strings.To_Bounded_String ("bool");
            elsif Type_Str = "void" then
               Mapped := Type_Name_Strings.To_Bounded_String ("void");
            else
               Mapped := IR_Type;
            end if;

         when Target_Swift =>
            if Type_Str = "int" then
               Mapped := Type_Name_Strings.To_Bounded_String ("Int");
            elsif Type_Str = "float" then
               Mapped := Type_Name_Strings.To_Bounded_String ("Double");
            elsif Type_Str = "string" then
               Mapped := Type_Name_Strings.To_Bounded_String ("String");
            elsif Type_Str = "bool" then
               Mapped := Type_Name_Strings.To_Bounded_String ("Bool");
            elsif Type_Str = "void" then
               Mapped := Type_Name_Strings.To_Bounded_String ("Void");
            else
               Mapped := IR_Type;
            end if;

         when Target_Kotlin =>
            if Type_Str = "int" then
               Mapped := Type_Name_Strings.To_Bounded_String ("Int");
            elsif Type_Str = "float" then
               Mapped := Type_Name_Strings.To_Bounded_String ("Double");
            elsif Type_Str = "string" then
               Mapped := Type_Name_Strings.To_Bounded_String ("String");
            elsif Type_Str = "bool" then
               Mapped := Type_Name_Strings.To_Bounded_String ("Boolean");
            elsif Type_Str = "void" then
               Mapped := Type_Name_Strings.To_Bounded_String ("Unit");
            else
               Mapped := IR_Type;
            end if;

         when Target_SPARK =>
            if Type_Str = "int" then
               Mapped := Type_Name_Strings.To_Bounded_String ("Integer");
            elsif Type_Str = "float" then
               Mapped := Type_Name_Strings.To_Bounded_String ("Float");
            elsif Type_Str = "string" then
               Mapped := Type_Name_Strings.To_Bounded_String ("String");
            elsif Type_Str = "bool" then
               Mapped := Type_Name_Strings.To_Bounded_String ("Boolean");
            elsif Type_Str = "void" then
               Mapped := Type_Name_Strings.To_Bounded_String ("");
            else
               Mapped := IR_Type;
            end if;

         --  AST-emitter targets: type mapping is handled by the emitter layer
         --  (STUNIR.Emitters.CFamily, Python, Lisp, Prolog, Futhark, Lean4).
         --  Return the IR type unchanged; the emitter will map it.
         when Target_Clojure | Target_ClojureScript | Target_Prolog |
              Target_Futhark | Target_Lean4 =>
            Mapped := IR_Type;

      end case;
   end Map_Type_To_Target;

   --  =======================================================================
   --  Code Generation Implementation
   --  =======================================================================

   procedure Generate_Function_Code
     (Func      : in     IR_Function;
      Target    : in     Target_Language;
      Code      :    out Code_String;
      Status    :    out Status_Code)
   is
      Temp_Status : Status_Code;
   begin
      Code := Code_Strings.Null_Bounded_String;
      Status := Success;

      case Target is
         when Target_CPP =>
            --  Return type
            declare
               Mapped_Type : Type_Name_String;
            begin
               Map_Type_To_Target (Func.Return_Type, Target_CPP, Mapped_Type, Temp_Status);
               if Temp_Status /= Success then
                  Status := Temp_Status;
                  return;
               end if;
               Append_To_Code (Code, Type_Name_Strings.To_String (Mapped_Type) & " ", Temp_Status);
               if Temp_Status /= Success then
                  Status := Temp_Status;
                  return;
               end if;
            end;

            --  Function name
            Append_To_Code (Code, Identifier_Strings.To_String (Func.Name), Temp_Status);
            if Temp_Status /= Success then
               Status := Temp_Status;
               return;
            end if;

            --  Parameters
            Append_To_Code (Code, "(", Temp_Status);
            for I in 1 .. Func.Parameters.Count loop
               declare
                  Param : constant Parameter := Func.Parameters.Params (I);
                  Mapped_Type : Type_Name_String;
               begin
                  Map_Type_To_Target (Param.Param_Type, Target_CPP, Mapped_Type, Temp_Status);
                  if Temp_Status /= Success then
                     Status := Temp_Status;
                     return;
                  end if;
                  Append_To_Code (Code, Type_Name_Strings.To_String (Mapped_Type) & " " &
                                        Identifier_Strings.To_String (Param.Name), Temp_Status);
                  if Temp_Status /= Success then
                     Status := Temp_Status;
                     return;
                  end if;
                  if I < Func.Parameters.Count then
                     Append_To_Code (Code, ", ", Temp_Status);
                     if Temp_Status /= Success then
                        Status := Temp_Status;
                        return;
                     end if;
                  end if;
               end;
            end loop;
            Append_To_Code (Code, ") {" & ASCII.LF, Temp_Status);
            if Temp_Status /= Success then
               Status := Temp_Status;
               return;
            end if;

            --  Function body (placeholder)
            Append_To_Code (Code, "    // TODO: Implement function body" & ASCII.LF, Temp_Status);
            Append_To_Code (Code, "    return 0;" & ASCII.LF, Temp_Status);
            Append_To_Code (Code, "}" & ASCII.LF & ASCII.LF, Temp_Status);

         when Target_Python =>
            Append_To_Code (Code, "def ", Temp_Status);
            Append_To_Code (Code, Identifier_Strings.To_String (Func.Name), Temp_Status);
            Append_To_Code (Code, "(", Temp_Status);
            for I in 1 .. Func.Parameters.Count loop
               declare
                  Param : constant Parameter := Func.Parameters.Params (I);
               begin
                  Append_To_Code (Code, Identifier_Strings.To_String (Param.Name), Temp_Status);
                  if I < Func.Parameters.Count then
                     Append_To_Code (Code, ", ", Temp_Status);
                  end if;
               end;
            end loop;
            Append_To_Code (Code, "):" & ASCII.LF, Temp_Status);
            Append_To_Code (Code, "    # TODO: Implement function body" & ASCII.LF, Temp_Status);
            Append_To_Code (Code, "    pass" & ASCII.LF & ASCII.LF, Temp_Status);

         when Target_Rust =>
            --  Function signature: pub fn name(param: type, ...) -> ret_type
            Append_To_Code (Code, "pub fn ", Temp_Status);
            Append_To_Code (Code, Identifier_Strings.To_String (Func.Name), Temp_Status);
            Append_To_Code (Code, "(", Temp_Status);

            --  Parameters with Rust syntax: name: type
            for I in 1 .. Func.Parameters.Count loop
               declare
                  Param : constant Parameter := Func.Parameters.Params (I);
                  Mapped_Type : Type_Name_String;
               begin
                  --  Parameter name first
                  Append_To_Code (Code, Identifier_Strings.To_String (Param.Name), Temp_Status);
                  if Temp_Status /= Success then
                     Status := Temp_Status;
                     return;
                  end if;

                  --  Then type with colon separator
                  Map_Type_To_Target (Param.Param_Type, Target_Rust, Mapped_Type, Temp_Status);
                  if Temp_Status /= Success then
                     Status := Temp_Status;
                     return;
                  end if;
                  Append_To_Code (Code, ": " & Type_Name_Strings.To_String (Mapped_Type), Temp_Status);
                  if Temp_Status /= Success then
                     Status := Temp_Status;
                     return;
                  end if;

                  if I < Func.Parameters.Count then
                     Append_To_Code (Code, ", ", Temp_Status);
                     if Temp_Status /= Success then
                        Status := Temp_Status;
                        return;
                     end if;
                  end if;
               end;
            end loop;
            Append_To_Code (Code, ")", Temp_Status);
            if Temp_Status /= Success then
               Status := Temp_Status;
               return;
            end if;

            --  Return type
            declare
               Mapped_Type : Type_Name_String;
            begin
               Map_Type_To_Target (Func.Return_Type, Target_Rust, Mapped_Type, Temp_Status);
               if Temp_Status /= Success then
                  Status := Temp_Status;
                  return;
               end if;
               declare
                  Ret_Str : constant String := Type_Name_Strings.To_String (Mapped_Type);
               begin
                  if Ret_Str /= "()" then
                     Append_To_Code (Code, " -> " & Ret_Str, Temp_Status);
                     if Temp_Status /= Success then
                        Status := Temp_Status;
                        return;
                     end if;
                  end if;
               end;
            end;

            Append_To_Code (Code, " {" & ASCII.LF, Temp_Status);
            if Temp_Status /= Success then
               Status := Temp_Status;
               return;
            end if;

            --  Function body with return statement
            Append_To_Code (Code, "    // TODO: Implement function body" & ASCII.LF, Temp_Status);
            declare
               Mapped_Type : Type_Name_String;
            begin
               Map_Type_To_Target (Func.Return_Type, Target_Rust, Mapped_Type, Temp_Status);
               declare
                  Ret_Str : constant String := Type_Name_Strings.To_String (Mapped_Type);
               begin
                  if Ret_Str /= "()" then
                     Append_To_Code (Code, "    ", Temp_Status);
                     if Ret_Str = "i32" or Ret_Str = "i64" or Ret_Str = "isize" then
                        Append_To_Code (Code, "return 0;" & ASCII.LF, Temp_Status);
                     elsif Ret_Str = "f32" or Ret_Str = "f64" then
                        Append_To_Code (Code, "return 0.0;" & ASCII.LF, Temp_Status);
                     elsif Ret_Str = "bool" then
                        Append_To_Code (Code, "return false;" & ASCII.LF, Temp_Status);
                     elsif Ret_Str = "String" then
                        Append_To_Code (Code, "return String::new();" & ASCII.LF, Temp_Status);
                     else
                        Append_To_Code (Code, "return Default::default();" & ASCII.LF, Temp_Status);
                     end if;
                  end if;
               end;
            end;
            Append_To_Code (Code, "}" & ASCII.LF & ASCII.LF, Temp_Status);

         when Target_Go =>
            --  Function signature: func name(param type) ret_type
            Append_To_Code (Code, "func ", Temp_Status);
            Append_To_Code (Code, Identifier_Strings.To_String (Func.Name), Temp_Status);
            Append_To_Code (Code, "(", Temp_Status);

            --  Parameters
            for I in 1 .. Func.Parameters.Count loop
               declare
                  Param : constant Parameter := Func.Parameters.Params (I);
                  Mapped_Type : Type_Name_String;
               begin
                  Append_To_Code (Code, Identifier_Strings.To_String (Param.Name) & " ", Temp_Status);
                  if Temp_Status /= Success then
                     Status := Temp_Status;
                     return;
                  end if;
                  Map_Type_To_Target (Param.Param_Type, Target_Go, Mapped_Type, Temp_Status);
                  if Temp_Status /= Success then
                     Status := Temp_Status;
                     return;
                  end if;
                  Append_To_Code (Code, Type_Name_Strings.To_String (Mapped_Type), Temp_Status);
                  if Temp_Status /= Success then
                     Status := Temp_Status;
                     return;
                  end if;
                  if I < Func.Parameters.Count then
                     Append_To_Code (Code, ", ", Temp_Status);
                     if Temp_Status /= Success then
                        Status := Temp_Status;
                        return;
                     end if;
                  end if;
               end;
            end loop;
            Append_To_Code (Code, ")", Temp_Status);
            if Temp_Status /= Success then
               Status := Temp_Status;
               return;
            end if;

            --  Return type
            declare
               Mapped_Type : Type_Name_String;
            begin
               Map_Type_To_Target (Func.Return_Type, Target_Go, Mapped_Type, Temp_Status);
               if Temp_Status /= Success then
                  Status := Temp_Status;
                  return;
               end if;
               declare
                  Ret_Str : constant String := Type_Name_Strings.To_String (Mapped_Type);
               begin
                  if Ret_Str /= "" then
                     Append_To_Code (Code, " " & Ret_Str, Temp_Status);
                     if Temp_Status /= Success then
                        Status := Temp_Status;
                        return;
                     end if;
                  end if;
               end;
            end;

            Append_To_Code (Code, " {" & ASCII.LF, Temp_Status);
            if Temp_Status /= Success then
               Status := Temp_Status;
               return;
            end if;

            --  Function body
            Append_To_Code (Code, "    // TODO: Implement function body" & ASCII.LF, Temp_Status);
            declare
               Mapped_Type : Type_Name_String;
            begin
               Map_Type_To_Target (Func.Return_Type, Target_Go, Mapped_Type, Temp_Status);
               declare
                  Ret_Str : constant String := Type_Name_Strings.To_String (Mapped_Type);
               begin
                  if Ret_Str /= "" then
                     Append_To_Code (Code, "    return ", Temp_Status);
                     if Ret_Str = "int" or Ret_Str = "int32" or Ret_Str = "int64" then
                        Append_To_Code (Code, "0" & ASCII.LF, Temp_Status);
                     elsif Ret_Str = "float64" or Ret_Str = "float32" then
                        Append_To_Code (Code, "0.0" & ASCII.LF, Temp_Status);
                     elsif Ret_Str = "bool" then
                        Append_To_Code (Code, "false" & ASCII.LF, Temp_Status);
                     elsif Ret_Str = "string" then
                        Append_To_Code (Code, """" & ASCII.LF, Temp_Status);
                     else
                        Append_To_Code (Code, "nil" & ASCII.LF, Temp_Status);
                     end if;
                  end if;
               end;
            end;
            Append_To_Code (Code, "}" & ASCII.LF & ASCII.LF, Temp_Status);

         when others =>
            --  Simplified for other languages
            Append_To_Code (Code, "// Function: " & Identifier_Strings.To_String (Func.Name) & ASCII.LF, Temp_Status);
            Append_To_Code (Code, "// TODO: Generate code for target language" & ASCII.LF & ASCII.LF, Temp_Status);
      end case;

      Status := Success;
   end Generate_Function_Code;

   procedure Generate_Header
     (Target : in     Target_Language;
      Header :    out Code_String;
      Status :    out Status_Code)
   is
      Temp_Status : Status_Code;
   begin
      Header := Code_Strings.Null_Bounded_String;

      case Target is
         when Target_CPP =>
            Append_To_Code (Header, "// Generated by STUNIR Code Emitter" & ASCII.LF, Temp_Status);
            Append_To_Code (Header, "// DO NOT EDIT MANUALLY" & ASCII.LF & ASCII.LF, Temp_Status);
            Append_To_Code (Header, "#include <string>" & ASCII.LF & ASCII.LF, Temp_Status);

         when Target_C =>
            Append_To_Code (Header, "/* Generated by STUNIR Code Emitter */" & ASCII.LF, Temp_Status);
            Append_To_Code (Header, "/* DO NOT EDIT MANUALLY */" & ASCII.LF & ASCII.LF, Temp_Status);
            Append_To_Code (Header, "#include <stdio.h>" & ASCII.LF & ASCII.LF, Temp_Status);

         when Target_Python =>
            Append_To_Code (Header, "# Generated by STUNIR Code Emitter" & ASCII.LF, Temp_Status);
            Append_To_Code (Header, "# DO NOT EDIT MANUALLY" & ASCII.LF & ASCII.LF, Temp_Status);

         when Target_Rust =>
            Append_To_Code (Header, "// Generated by STUNIR Code Emitter" & ASCII.LF, Temp_Status);
            Append_To_Code (Header, "// DO NOT EDIT MANUALLY" & ASCII.LF & ASCII.LF, Temp_Status);

         when Target_Go =>
            Append_To_Code (Header, "// Generated by STUNIR Code Emitter" & ASCII.LF, Temp_Status);
            Append_To_Code (Header, "// DO NOT EDIT MANUALLY" & ASCII.LF & ASCII.LF, Temp_Status);
            Append_To_Code (Header, "package main" & ASCII.LF & ASCII.LF, Temp_Status);

         when others =>
            Append_To_Code (Header, "// Generated by STUNIR Code Emitter" & ASCII.LF, Temp_Status);
            Append_To_Code (Header, "// DO NOT EDIT MANUALLY" & ASCII.LF & ASCII.LF, Temp_Status);
      end case;

      Status := Temp_Status;
   end Generate_Header;

   procedure Generate_Footer
     (Target : in     Target_Language;
      Footer :    out Code_String;
      Status :    out Status_Code)
   is
      Temp_Status : Status_Code;
   begin
      Footer := Code_Strings.Null_Bounded_String;

      case Target is
         when Target_Go =>
            Append_To_Code (Footer, ASCII.LF & "func main() {" & ASCII.LF, Temp_Status);
            Append_To_Code (Footer, "    // Entry point" & ASCII.LF, Temp_Status);
            Append_To_Code (Footer, "}" & ASCII.LF, Temp_Status);

         when others =>
            null;  --  No footer needed
      end case;

      Status := Temp_Status;
   end Generate_Footer;

   procedure Generate_All_Code
     (IR       : in     IR_Data;
      Target   : in     Target_Language;
      Complete :    out Code_String;
      Status   :    out Status_Code)
   is
      Header : Code_String;
      Footer : Code_String;
      Func_Code : Code_String;
      Temp_Status : Status_Code;
   begin
      Complete := Code_Strings.Null_Bounded_String;
      Status := Success;

      --  Generate header
      Generate_Header (Target, Header, Temp_Status);
      if Temp_Status /= Success then
         Status := Temp_Status;
         return;
      end if;

      Append_To_Code (Complete, Code_Strings.To_String (Header), Temp_Status);
      if Temp_Status /= Success then
         Status := Temp_Status;
         return;
      end if;

      --  Generate code for each function
      for I in 1 .. IR.Functions.Count loop
         Generate_Function_Code (IR.Functions.Functions (I), Target, Func_Code, Temp_Status);
         if Temp_Status /= Success then
            Status := Temp_Status;
            return;
         end if;
         Append_To_Code (Complete, Code_Strings.To_String (Func_Code), Temp_Status);
         if Temp_Status /= Success then
            Status := Temp_Status;
            return;
         end if;
      end loop;

      --  Generate footer
      Generate_Footer (Target, Footer, Temp_Status);
      if Temp_Status /= Success then
         Status := Temp_Status;
         return;
      end if;
      Append_To_Code (Complete, Code_Strings.To_String (Footer), Temp_Status);

      Status := Success;
   end Generate_All_Code;

   procedure Generate_CPP_Code
     (IR     : in     IR_Data;
      Code   :    out Code_String;
      Status :    out Status_Code)
   is
   begin
      Generate_All_Code (IR, Target_CPP, Code, Status);
   end Generate_CPP_Code;

   procedure Generate_C_Code
     (IR     : in     IR_Data;
      Code   :    out Code_String;
      Status :    out Status_Code)
   is
   begin
      Generate_All_Code (IR, Target_C, Code, Status);
   end Generate_C_Code;

   procedure Generate_Python_Code
     (IR     : in     IR_Data;
      Code   :    out Code_String;
      Status :    out Status_Code)
   is
   begin
      Generate_All_Code (IR, Target_Python, Code, Status);
   end Generate_Python_Code;

   procedure Generate_Rust_Code
     (IR     : in     IR_Data;
      Code   :    out Code_String;
      Status :    out Status_Code)
   is
   begin
      Generate_All_Code (IR, Target_Rust, Code, Status);
   end Generate_Rust_Code;

   procedure Generate_Go_Code
     (IR     : in     IR_Data;
      Code   :    out Code_String;
      Status :    out Status_Code)
   is
   begin
      Generate_All_Code (IR, Target_Go, Code, Status);
   end Generate_Go_Code;

   --  =======================================================================
   --  File Operations
   --  =======================================================================

   function Get_File_Extension
     (Target : Target_Language) return Identifier_String
   is
   begin
      case Target is
         when Target_CPP       => return Identifier_Strings.To_Bounded_String (".cpp");
         when Target_C         => return Identifier_Strings.To_Bounded_String (".c");
         when Target_Python    => return Identifier_Strings.To_Bounded_String (".py");
         when Target_Rust      => return Identifier_Strings.To_Bounded_String (".rs");
         when Target_Go        => return Identifier_Strings.To_Bounded_String (".go");
         when Target_Java      => return Identifier_Strings.To_Bounded_String (".java");
         when Target_JavaScript=> return Identifier_Strings.To_Bounded_String (".js");
         when Target_CSharp    => return Identifier_Strings.To_Bounded_String (".cs");
         when Target_Swift     => return Identifier_Strings.To_Bounded_String (".swift");
         when Target_Kotlin    => return Identifier_Strings.To_Bounded_String (".kt");
         when Target_SPARK     => return Identifier_Strings.To_Bounded_String (".adb");
         when Target_Clojure   => return Identifier_Strings.To_Bounded_String (".clj");
         when Target_ClojureScript => return Identifier_Strings.To_Bounded_String (".cljs");
         when Target_Prolog    => return Identifier_Strings.To_Bounded_String (".pl");
         when Target_Futhark   => return Identifier_Strings.To_Bounded_String (".fut");
         when Target_Lean4     => return Identifier_Strings.To_Bounded_String (".lean");
      end case;
   end Get_File_Extension;

   --  =======================================================================
   --  IR Parsing (simplified)
   --  =======================================================================

   procedure Parse_IR_JSON
     (JSON_Content : in     JSON_String;
      IR           :    out IR_Data;
      Status       :    out Status_Code)
   is
      Module : Semantic_IR.Modules.IR_Module;
      Nodes  : STUNIR.Emitters.Node_Table.Node_Table;
   begin
      --  Field-by-field init to avoid Dynamic_Predicate violation on IR_Function
      IR.Schema_Version  := Identifier_Strings.Null_Bounded_String;
      IR.IR_Version      := Identifier_Strings.Null_Bounded_String;
      IR.Module_Name     := Identifier_Strings.Null_Bounded_String;
      IR.Functions.Count := 0;

      Semantic_IR.JSON.Parse_IR_JSON (JSON_Content, Module, Nodes, Status);
      if Status /= Success then
         return;
      end if;

      -- Minimal bridge: fill module name for output path
      IR.Module_Name := Identifier_Strings.To_Bounded_String (Semantic_IR.Types.Name_Strings.To_String (Module.Module_Name));
      IR.IR_Version := Identifier_Strings.To_Bounded_String (Semantic_IR.Schema_Version);
      IR.Schema_Version := Identifier_Strings.To_Bounded_String ("semantic_ir_v1");
      Status := Success;
   end Parse_IR_JSON;

   procedure Emit_AST_Module
     (Module : in     Semantic_IR.Modules.IR_Module;
      Nodes  : in     STUNIR.Emitters.Node_Table.Node_Table;
      Target : in     Target_Language;
      Output :    out Code_String;
      Status :    out Status_Code)
   is
      Buffer  : STUNIR.Emitters.CodeGen.IR_Code_Buffer;
      Success : Boolean := False;
   begin
      case Target is
         when Target_Python =>
            declare
               E : STUNIR.Emitters.Python.Python_Emitter;
            begin
               STUNIR.Emitters.Python.Emit_Module (E, Module, Nodes, Buffer, Success);
            end;
         when Target_C =>
            declare
               E : STUNIR.Emitters.CFamily.C_Emitter;
            begin
               STUNIR.Emitters.CFamily.Emit_Module (E, Module, Nodes, Buffer, Success);
            end;
         when Target_Clojure =>
            declare
               E : STUNIR.Emitters.Lisp.Lisp_Emitter;
            begin
               E.Config.Dialect := STUNIR.Emitters.Lisp.Clojure;
               STUNIR.Emitters.Lisp.Emit_Module (E, Module, Nodes, Buffer, Success);
            end;
         when Target_ClojureScript =>
            declare
               E : STUNIR.Emitters.Lisp.Lisp_Emitter;
            begin
               E.Config.Dialect := STUNIR.Emitters.Lisp.ClojureScript;
               STUNIR.Emitters.Lisp.Emit_Module (E, Module, Nodes, Buffer, Success);
            end;
         when Target_Prolog =>
            declare
               E : STUNIR.Emitters.Prolog_Family.Prolog_Emitter;
            begin
               STUNIR.Emitters.Prolog_Family.Emit_Module (E, Module, Nodes, Buffer, Success);
            end;
         when Target_Futhark =>
            declare
               E : STUNIR.Emitters.Futhark_Family.Futhark_Emitter;
            begin
               STUNIR.Emitters.Futhark_Family.Emit_Module (E, Module, Nodes, Buffer, Success);
            end;
         when Target_Lean4 =>
            declare
               E : STUNIR.Emitters.Lean4_Family.Lean4_Emitter;
            begin
               STUNIR.Emitters.Lean4_Family.Emit_Module (E, Module, Nodes, Buffer, Success);
            end;
         when others =>
            Success := False;
      end case;

      if Success then
         Output := Code_Strings.To_Bounded_String (STUNIR.Emitters.CodeGen.Code_Buffers.To_String (Buffer));
         Status := STUNIR_Types.Success;
      else
         Output := Code_Strings.Null_Bounded_String;
         Status := Error_Not_Implemented;
      end if;
   end Emit_AST_Module;

   --  =======================================================================
   --  Main Entry Points
   --  =======================================================================

   procedure Process_IR_File
     (Input_Path   : in     Path_String;
      Output_Dir   : in     Path_String;
      Target       : in     Target_Language;
      Status       :    out Status_Code)
   is
      pragma SPARK_Mode (Off);  --  File I/O not in SPARK

      Input_File   : File_Type;
      Output_File  : File_Type;
      File_Content : String (1 .. Max_JSON_Length);
      Content_Len  : Natural := 0;
      JSON_Content : JSON_String;
      IR           : IR_Data;
      Code         : Code_String;
      Output_Path  : Path_String;
   begin
      Status := Success;

      --  Read input file
      begin
         Open (Input_File, In_File, Path_Strings.To_String (Input_Path));
         while not End_Of_File (Input_File) and Content_Len < Max_JSON_Length loop
            declare
               Line : constant String := Get_Line (Input_File);
               New_Len : constant Natural := Content_Len + Line'Length;
            begin
               if New_Len <= Max_JSON_Length then
                  File_Content (Content_Len + 1 .. New_Len) := Line;
                  Content_Len := New_Len;
               else
                  Close (Input_File);
                  Status := Error_File_IO;
                  return;
               end if;
            end;
         end loop;
         Close (Input_File);
      exception
         when others =>
            Status := Error_File_IO;
            return;
      end;

      if Content_Len = 0 then
         Status := Error_Invalid_Input;
         return;
      end if;

      JSON_Content := JSON_Strings.To_Bounded_String (File_Content (1 .. Content_Len));

      --  Parse IR JSON into AST and map minimal IR metadata
      Parse_IR_JSON (JSON_Content, IR, Status);
      if Status /= Success then
         return;
      end if;

      --  Re-parse AST for code emission (AST emitters)
      declare
         Module : Semantic_IR.Modules.IR_Module;
         Nodes  : STUNIR.Emitters.Node_Table.Node_Table;
      begin
         Semantic_IR.JSON.Parse_IR_JSON (JSON_Content, Module, Nodes, Status);
         if Status /= Success then
            return;
         end if;

         if Target in Target_C | Target_Python | Target_Clojure | Target_ClojureScript | Target_Prolog | Target_Futhark | Target_Lean4 then
            Emit_AST_Module (Module, Nodes, Target, Code, Status);
         else
            Generate_All_Code (IR, Target, Code, Status);
         end if;
      end;

      --  Construct output path
      declare
         Out_Dir  : constant String := Path_Strings.To_String (Output_Dir);
         Mod_Name : constant String := Identifier_Strings.To_String (IR.Module_Name);
         Ext      : constant String := Identifier_Strings.To_String (Get_File_Extension (Target));
         Full_Path : constant String := Out_Dir & "/" & Mod_Name & Ext;
      begin
         if Full_Path'Length <= Max_Path_Length then
            Output_Path := Path_Strings.To_Bounded_String (Full_Path);
         else
            Status := Error_Too_Large;
            return;
         end if;
      end;

      --  Write output file
      begin
         Create (Output_File, Out_File, Path_Strings.To_String (Output_Path));
         Put (Output_File, Code_Strings.To_String (Code));
         Close (Output_File);
      exception
         when others =>
            Status := Error_File_IO;
            return;
      end;

      Status := Success;
   end Process_IR_File;

   procedure Process_IR_File_All_Targets
     (Input_Path : in     Path_String;
      Output_Dir : in     Path_String;
      Status     :    out Status_Code)
   is
      Target_Status : Status_Code;
   begin
      Status := Success;

      --  Process for each target language
      for Target in Target_Language loop
         Process_IR_File (Input_Path, Output_Dir, Target, Target_Status);
         if Target_Status /= Success then
            Status := Target_Status;
            return;
         end if;
      end loop;
   end Process_IR_File_All_Targets;

end Code_Emitter;