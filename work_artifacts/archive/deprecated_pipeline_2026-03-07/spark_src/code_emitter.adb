--  STUNIR Code Emitter Package Body
--  Converts ir.json to target language code
--  DO-333 / SPARK 2014 compliant
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

with STUNIR_JSON_Parser;
use STUNIR_JSON_Parser;
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
with Ada.Directories;
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

         when Target_Futhark =>
            if Type_Str = "int" then
               Mapped := Type_Name_Strings.To_Bounded_String ("i32");
            elsif Type_Str = "float" then
               Mapped := Type_Name_Strings.To_Bounded_String ("f64");
            elsif Type_Str = "string" then
               Mapped := Type_Name_Strings.To_Bounded_String ("string");
            elsif Type_Str = "bool" then
               Mapped := Type_Name_Strings.To_Bounded_String ("bool");
            elsif Type_Str = "void" then
               Mapped := Type_Name_Strings.To_Bounded_String ("()");
            else
               Mapped := IR_Type;
            end if;

         when Target_Lean4 =>
            if Type_Str = "int" then
               Mapped := Type_Name_Strings.To_Bounded_String ("Int");
            elsif Type_Str = "float" then
               Mapped := Type_Name_Strings.To_Bounded_String ("Float");
            elsif Type_Str = "string" then
               Mapped := Type_Name_Strings.To_Bounded_String ("String");
            elsif Type_Str = "bool" then
               Mapped := Type_Name_Strings.To_Bounded_String ("Bool");
            elsif Type_Str = "void" then
               Mapped := Type_Name_Strings.To_Bounded_String ("Unit");
            else
               Mapped := IR_Type;
            end if;

         --  Other targets use IR types directly for now.
         when Target_Clojure | Target_ClojureScript | Target_Prolog =>
            Mapped := IR_Type;

      end case;
   end Map_Type_To_Target;

   --  =======================================================================
   --  Code Generation Implementation
   --  =======================================================================

   function Step_Type_To_String (Step : Step_Type_Enum) return String is
   begin
      case Step is
         when Step_Nop           => return "nop";
         when Step_Assign        => return "assign";
         when Step_Call          => return "call";
         when Step_Return        => return "return";
         when Step_Error         => return "error";
         when Step_If            => return "if";
         when Step_While         => return "while";
         when Step_For           => return "for";
         when Step_Break         => return "break";
         when Step_Continue      => return "continue";
         when Step_Switch        => return "switch";
         when Step_Try           => return "try";
         when Step_Throw         => return "throw";
         when Step_Array_New     => return "array_new";
         when Step_Array_Get     => return "array_get";
         when Step_Array_Set     => return "array_set";
         when Step_Array_Push    => return "array_push";
         when Step_Array_Pop     => return "array_pop";
         when Step_Array_Len     => return "array_len";
         when Step_Map_New       => return "map_new";
         when Step_Map_Get       => return "map_get";
         when Step_Map_Set       => return "map_set";
         when Step_Map_Delete    => return "map_delete";
         when Step_Map_Has       => return "map_has";
         when Step_Map_Keys      => return "map_keys";
         when Step_Set_New       => return "set_new";
         when Step_Set_Add       => return "set_add";
         when Step_Set_Remove    => return "set_remove";
         when Step_Set_Has       => return "set_has";
         when Step_Set_Union     => return "set_union";
         when Step_Set_Intersect => return "set_intersect";
         when Step_Struct_New    => return "struct_new";
         when Step_Struct_Get    => return "struct_get";
         when Step_Struct_Set    => return "struct_set";
         when Step_Generic_Call  => return "generic_call";
         when Step_Type_Cast     => return "type_cast";
      end case;
   end Step_Type_To_String;

   procedure Emit_Step_Comments
     (Code   : in out Code_String;
      Steps  : in     Step_Collection;
      Indent : in     String;
      Status :    out Status_Code)
   is
      Temp_Status : Status_Code;
   begin
      Status := Success;
      if Steps.Count = 0 then
         return;
      end if;

      for I in 1 .. Steps.Count loop
         declare
            Step : constant IR_Step := Steps.Steps (I);
         begin
            Append_To_Code (Code, Indent & "// step: " & Step_Type_To_String (Step.Step_Type), Temp_Status);
            if Temp_Status /= Success then
               Status := Temp_Status;
               return;
            end if;

            if Identifier_Strings.Length (Step.Target) > 0 then
               Append_To_Code (Code, " target=" & Identifier_Strings.To_String (Step.Target), Temp_Status);
               if Temp_Status /= Success then
                  Status := Temp_Status;
                  return;
               end if;
            end if;
            if Identifier_Strings.Length (Step.Value) > 0 then
               Append_To_Code (Code, " value=" & Identifier_Strings.To_String (Step.Value), Temp_Status);
               if Temp_Status /= Success then
                  Status := Temp_Status;
                  return;
               end if;
            end if;
            if Identifier_Strings.Length (Step.Value) > 0 then
               Append_To_Code (Code, " value=" & Identifier_Strings.To_String (Step.Value), Temp_Status);
               if Temp_Status /= Success then
                  Status := Temp_Status;
                  return;
               end if;
            end if;

            Append_To_Code (Code, "" & ASCII.LF, Temp_Status);
            if Temp_Status /= Success then
               Status := Temp_Status;
               return;
            end if;
         end;
      end loop;
   end Emit_Step_Comments;

   procedure Find_Return_Step
     (Steps        : in     Step_Collection;
      Has_Return   :    out Boolean;
      Return_Value :    out Identifier_String)
   is
   begin
      Has_Return := False;
      Return_Value := Null_Identifier;
      for I in 1 .. Steps.Count loop
         if Steps.Steps (I).Step_Type = Step_Return then
            Has_Return := True;
            Return_Value := Steps.Steps (I).Value;
            exit;
         end if;
      end loop;
   end Find_Return_Step;

   procedure Emit_Assign_Steps_CLike
     (Code  : in out Code_String;
      Steps : in     Step_Collection;
      Status:    out Status_Code)
   is
      Temp_Status : Status_Code;
   begin
      Status := Success;
      for I in 1 .. Steps.Count loop
         if Steps.Steps (I).Step_Type = Step_Assign then
            declare
               Target_Str : constant String := Identifier_Strings.To_String (Steps.Steps (I).Target);
               Value_Str  : constant String := Identifier_Strings.To_String (Steps.Steps (I).Value);
            begin
               if Target_Str'Length > 0 and then Value_Str'Length > 0 then
                  Append_To_Code (Code, "    " & Target_Str & " = " & Value_Str & ";" & ASCII.LF, Temp_Status);
                  if Temp_Status /= Success then
                     Status := Temp_Status;
                     return;
                  end if;
               end if;
            end;
         end if;
      end loop;
   end Emit_Assign_Steps_CLike;

   procedure Emit_Assign_Steps
     (Code      : in out Code_String;
      Steps     : in     Step_Collection;
      Operator  : in     String;
      Line_End  : in     String;
      Status    :    out Status_Code)
   is
      Temp_Status : Status_Code;
   begin
      Status := Success;
      for I in 1 .. Steps.Count loop
         if Steps.Steps (I).Step_Type = Step_Assign then
            declare
               Target_Str : constant String := Identifier_Strings.To_String (Steps.Steps (I).Target);
               Value_Str  : constant String := Identifier_Strings.To_String (Steps.Steps (I).Value);
            begin
               if Target_Str'Length > 0 and then Value_Str'Length > 0 then
                  Append_To_Code (Code, "    " & Target_Str & Operator & Value_Str & Line_End, Temp_Status);
                  if Temp_Status /= Success then
                     Status := Temp_Status;
                     return;
                  end if;
               end if;
            end;
         end if;
      end loop;
   end Emit_Assign_Steps;

   procedure Emit_Call_Steps_CLike
     (Code  : in out Code_String;
      Steps : in     Step_Collection;
      Status:    out Status_Code)
   is
      Temp_Status : Status_Code;
   begin
      Status := Success;
      for I in 1 .. Steps.Count loop
         if Steps.Steps (I).Step_Type = Step_Call then
            declare
               Target_Str : constant String := Identifier_Strings.To_String (Steps.Steps (I).Target);
               Value_Str  : constant String := Identifier_Strings.To_String (Steps.Steps (I).Value);
            begin
               if Value_Str'Length > 0 then
                  if Target_Str'Length > 0 then
                     Append_To_Code (Code, "    " & Target_Str & " = " & Value_Str & ";" & ASCII.LF, Temp_Status);
                  else
                     Append_To_Code (Code, "    " & Value_Str & ";" & ASCII.LF, Temp_Status);
                  end if;
                  if Temp_Status /= Success then
                     Status := Temp_Status;
                     return;
                  end if;
               end if;
            end;
         end if;
      end loop;
   end Emit_Call_Steps_CLike;

   procedure Emit_Call_Steps
     (Code      : in out Code_String;
      Steps     : in     Step_Collection;
      Operator  : in     String;
      Line_End  : in     String;
      Status    :    out Status_Code)
   is
      Temp_Status : Status_Code;
   begin
      Status := Success;
      for I in 1 .. Steps.Count loop
         if Steps.Steps (I).Step_Type = Step_Call then
            declare
               Target_Str : constant String := Identifier_Strings.To_String (Steps.Steps (I).Target);
               Value_Str  : constant String := Identifier_Strings.To_String (Steps.Steps (I).Value);
            begin
               if Value_Str'Length > 0 then
                  if Target_Str'Length > 0 then
                     Append_To_Code (Code, "    " & Target_Str & Operator & Value_Str & Line_End, Temp_Status);
                  else
                     Append_To_Code (Code, "    " & Value_Str & Line_End, Temp_Status);
                  end if;
                  if Temp_Status /= Success then
                     Status := Temp_Status;
                     return;
                  end if;
               end if;
            end;
         end if;
      end loop;
   end Emit_Call_Steps;

   --  =======================================================================
   --  Control Flow Emission
   --  =======================================================================

   procedure Emit_If_Step_CLike
     (Code   : in out Code_String;
      Step   : in     IR_Step;
      Steps  : in     Step_Collection;
      Indent : in     String;
      Status :    out Status_Code)
   is
      Temp_Status : Status_Code;
      Cond_Str    : constant String := Identifier_Strings.To_String (Step.Condition);
   begin
      Status := Success;
      if Cond_Str'Length > 0 then
         Append_To_Code (Code, Indent & "if (" & Cond_Str & ") {" & ASCII.LF, Temp_Status);
         if Temp_Status /= Success then
            Status := Temp_Status;
            return;
         end if;
         --  Emit then block steps
         for I in Step.Then_Start .. Step.Then_Start + Step.Then_Count - 1 loop
            if I <= Steps.Count then
               declare
                  S : constant IR_Step := Steps.Steps (I);
               begin
                  case S.Step_Type is
                     when Step_Assign =>
                        Append_To_Code (Code, Indent & "    " & 
                           Identifier_Strings.To_String (S.Target) & " = " & 
                           Identifier_Strings.To_String (S.Value) & ";" & ASCII.LF, Temp_Status);
                     when Step_Call =>
                        if Identifier_Strings.Length (S.Target) > 0 then
                           Append_To_Code (Code, Indent & "    " & 
                              Identifier_Strings.To_String (S.Target) & " = " & 
                              Identifier_Strings.To_String (S.Value) & ";" & ASCII.LF, Temp_Status);
                        else
                           Append_To_Code (Code, Indent & "    " & 
                              Identifier_Strings.To_String (S.Value) & ";" & ASCII.LF, Temp_Status);
                        end if;
                     when others =>
                        Append_To_Code (Code, Indent & "    // step: " & Step_Type_To_String (S.Step_Type) & ASCII.LF, Temp_Status);
                  end case;
               end;
            end if;
         end loop;
         Append_To_Code (Code, Indent & "}" & ASCII.LF, Temp_Status);
         --  Emit else block if present
         if Step.Else_Count > 0 then
            Append_To_Code (Code, Indent & "else {" & ASCII.LF, Temp_Status);
            for I in Step.Else_Start .. Step.Else_Start + Step.Else_Count - 1 loop
               if I <= Steps.Count then
                  declare
                     S : constant IR_Step := Steps.Steps (I);
                  begin
                     case S.Step_Type is
                        when Step_Assign =>
                           Append_To_Code (Code, Indent & "    " & 
                              Identifier_Strings.To_String (S.Target) & " = " & 
                              Identifier_Strings.To_String (S.Value) & ";" & ASCII.LF, Temp_Status);
                        when others =>
                           Append_To_Code (Code, Indent & "    // step: " & Step_Type_To_String (S.Step_Type) & ASCII.LF, Temp_Status);
                     end case;
                  end;
               end if;
            end loop;
            Append_To_Code (Code, Indent & "}" & ASCII.LF, Temp_Status);
         end if;
      end if;
   end Emit_If_Step_CLike;

   procedure Emit_While_Step_CLike
     (Code   : in out Code_String;
      Step   : in     IR_Step;
      Steps  : in     Step_Collection;
      Indent : in     String;
      Status :    out Status_Code)
   is
      Temp_Status : Status_Code;
      Cond_Str    : constant String := Identifier_Strings.To_String (Step.Condition);
   begin
      Status := Success;
      if Cond_Str'Length > 0 then
         Append_To_Code (Code, Indent & "while (" & Cond_Str & ") {" & ASCII.LF, Temp_Status);
         if Temp_Status /= Success then
            Status := Temp_Status;
            return;
         end if;
         --  Emit body steps
         for I in Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
            if I <= Steps.Count then
               declare
                  S : constant IR_Step := Steps.Steps (I);
               begin
                  case S.Step_Type is
                     when Step_Assign =>
                        Append_To_Code (Code, Indent & "    " & 
                           Identifier_Strings.To_String (S.Target) & " = " & 
                           Identifier_Strings.To_String (S.Value) & ";" & ASCII.LF, Temp_Status);
                     when Step_Break =>
                        Append_To_Code (Code, Indent & "    break;" & ASCII.LF, Temp_Status);
                     when Step_Continue =>
                        Append_To_Code (Code, Indent & "    continue;" & ASCII.LF, Temp_Status);
                     when others =>
                        Append_To_Code (Code, Indent & "    // step: " & Step_Type_To_String (S.Step_Type) & ASCII.LF, Temp_Status);
                  end case;
               end;
            end if;
         end loop;
         Append_To_Code (Code, Indent & "}" & ASCII.LF, Temp_Status);
      end if;
   end Emit_While_Step_CLike;

   procedure Emit_For_Step_CLike
     (Code   : in out Code_String;
      Step   : in     IR_Step;
      Steps  : in     Step_Collection;
      Indent : in     String;
      Status :    out Status_Code)
   is
      Temp_Status : Status_Code;
      Init_Str    : constant String := Identifier_Strings.To_String (Step.Init);
      Cond_Str    : constant String := Identifier_Strings.To_String (Step.Condition);
      Incr_Str    : constant String := Identifier_Strings.To_String (Step.Increment);
   begin
      Status := Success;
      Append_To_Code (Code, Indent & "for (" & Init_Str & "; " & Cond_Str & "; " & Incr_Str & ") {" & ASCII.LF, Temp_Status);
      if Temp_Status /= Success then
         Status := Temp_Status;
         return;
      end if;
      --  Emit body steps
      for I in Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
         if I <= Steps.Count then
            declare
               S : constant IR_Step := Steps.Steps (I);
            begin
               case S.Step_Type is
                  when Step_Assign =>
                     Append_To_Code (Code, Indent & "    " & 
                        Identifier_Strings.To_String (S.Target) & " = " & 
                        Identifier_Strings.To_String (S.Value) & ";" & ASCII.LF, Temp_Status);
                  when Step_Break =>
                     Append_To_Code (Code, Indent & "    break;" & ASCII.LF, Temp_Status);
                  when Step_Continue =>
                     Append_To_Code (Code, Indent & "    continue;" & ASCII.LF, Temp_Status);
                  when others =>
                     Append_To_Code (Code, Indent & "    // step: " & Step_Type_To_String (S.Step_Type) & ASCII.LF, Temp_Status);
               end case;
            end;
         end if;
      end loop;
      Append_To_Code (Code, Indent & "}" & ASCII.LF, Temp_Status);
   end Emit_For_Step_CLike;

   --  =======================================================================
   --  Data Structure Emission
   --  =======================================================================

   procedure Emit_Array_Ops_CLike
     (Code   : in out Code_String;
      Steps  : in     Step_Collection;
      Indent : in     String;
      Status :    out Status_Code)
   is
      Temp_Status : Status_Code;
   begin
      Status := Success;
      for I in 1 .. Steps.Count loop
         declare
            S : constant IR_Step := Steps.Steps (I);
         begin
            case S.Step_Type is
               when Step_Array_New =>
                  Append_To_Code (Code, Indent & 
                     Identifier_Strings.To_String (S.Target) & " = [];" & ASCII.LF, Temp_Status);
               when Step_Array_Get =>
                  Append_To_Code (Code, Indent & 
                     Identifier_Strings.To_String (S.Target) & " = " &
                     Identifier_Strings.To_String (S.Value) & "[" &
                     Identifier_Strings.To_String (S.Index) & "];" & ASCII.LF, Temp_Status);
               when Step_Array_Set =>
                  Append_To_Code (Code, Indent & 
                     Identifier_Strings.To_String (S.Target) & "[" &
                     Identifier_Strings.To_String (S.Index) & "] = " &
                     Identifier_Strings.To_String (S.Value) & ";" & ASCII.LF, Temp_Status);
               when Step_Array_Push =>
                  Append_To_Code (Code, Indent & 
                     Identifier_Strings.To_String (S.Target) & ".append(" &
                     Identifier_Strings.To_String (S.Value) & ");" & ASCII.LF, Temp_Status);
               when Step_Array_Len =>
                  Append_To_Code (Code, Indent & 
                     Identifier_Strings.To_String (S.Target) & " = len(" &
                     Identifier_Strings.To_String (S.Value) & ");" & ASCII.LF, Temp_Status);
               when others =>
                  null;
            end case;
            if Temp_Status /= Success then
               Status := Temp_Status;
               return;
            end if;
         end;
      end loop;
   end Emit_Array_Ops_CLike;

   procedure Emit_Map_Ops_CLike
     (Code   : in out Code_String;
      Steps  : in     Step_Collection;
      Indent : in     String;
      Status :    out Status_Code)
   is
      Temp_Status : Status_Code;
   begin
      Status := Success;
      for I in 1 .. Steps.Count loop
         declare
            S : constant IR_Step := Steps.Steps (I);
         begin
            case S.Step_Type is
               when Step_Map_New =>
                  Append_To_Code (Code, Indent & 
                     Identifier_Strings.To_String (S.Target) & " = {};" & ASCII.LF, Temp_Status);
               when Step_Map_Get =>
                  Append_To_Code (Code, Indent & 
                     Identifier_Strings.To_String (S.Target) & " = " &
                     Identifier_Strings.To_String (S.Value) & "[" &
                     Identifier_Strings.To_String (S.Key) & "];" & ASCII.LF, Temp_Status);
               when Step_Map_Set =>
                  Append_To_Code (Code, Indent & 
                     Identifier_Strings.To_String (S.Target) & "[" &
                     Identifier_Strings.To_String (S.Key) & "] = " &
                     Identifier_Strings.To_String (S.Value) & ";" & ASCII.LF, Temp_Status);
               when Step_Map_Has =>
                  Append_To_Code (Code, Indent & 
                     Identifier_Strings.To_String (S.Target) & " = (" &
                     Identifier_Strings.To_String (S.Key) & " in " &
                     Identifier_Strings.To_String (S.Value) & ");" & ASCII.LF, Temp_Status);
               when others =>
                  null;
            end case;
            if Temp_Status /= Success then
               Status := Temp_Status;
               return;
            end if;
         end;
      end loop;
   end Emit_Map_Ops_CLike;

   --  =======================================================================
   --  Exception Emission
   --  =======================================================================

   procedure Emit_Try_Step_CLike
     (Code   : in out Code_String;
      Step   : in     IR_Step;
      Steps  : in     Step_Collection;
      Indent : in     String;
      Status :    out Status_Code)
   is
      Temp_Status : Status_Code;
   begin
      Status := Success;
      --  C-like languages don't have try/catch, emit as comments
      Append_To_Code (Code, Indent & "/* try block */" & ASCII.LF, Temp_Status);
      if Temp_Status /= Success then
         Status := Temp_Status;
         return;
      end if;
      --  Emit try body
      if Step.Try_Count > 0 then
         for I in Step.Try_Start .. Step.Try_Start + Step.Try_Count - 1 loop
            if I >= 1 and then I <= Steps.Count then
               Append_To_Code (Code, Indent & "  // " & Step_Type_To_String (Steps.Steps (I).Step_Type) & ASCII.LF, Temp_Status);
            end if;
         end loop;
      end if;
      --  Emit catch blocks
      for C in Positive range 1 .. Positive (Step.Catch_Count) loop
         Append_To_Code (Code, Indent & "/* catch: " & 
            Identifier_Strings.To_String (Step.Catch_Blocks (C).Exception_Type) & " */" & ASCII.LF, Temp_Status);
      end loop;
   end Emit_Try_Step_CLike;

   procedure Emit_Throw_Step_CLike
     (Code   : in out Code_String;
      Step   : in     IR_Step;
      Indent : in     String;
      Status :    out Status_Code)
   is
      Temp_Status : Status_Code;
   begin
      Status := Success;
      Append_To_Code (Code, Indent & "/* throw: " & 
         Identifier_Strings.To_String (Step.Value) & " */" & ASCII.LF, Temp_Status);
   end Emit_Throw_Step_CLike;

   --  =======================================================================
   --  Generic Call and Type Cast Emission
   --  =======================================================================

   procedure Emit_Generic_Call_Step_CLike
     (Code   : in out Code_String;
      Step   : in     IR_Step;
      Indent : in     String;
      Status :    out Status_Code)
   is
      Temp_Status : Status_Code;
   begin
      Status := Success;
      if Identifier_Strings.Length (Step.Target) > 0 then
         Append_To_Code (Code, Indent & 
            Identifier_Strings.To_String (Step.Target) & " = " &
            Identifier_Strings.To_String (Step.Value) & "(" &
            Identifier_Strings.To_String (Step.Args) & ");" & ASCII.LF, Temp_Status);
      else
         Append_To_Code (Code, Indent & 
            Identifier_Strings.To_String (Step.Value) & "(" &
            Identifier_Strings.To_String (Step.Args) & ");" & ASCII.LF, Temp_Status);
      end if;
   end Emit_Generic_Call_Step_CLike;

   procedure Emit_Type_Cast_Step_CLike
     (Code   : in out Code_String;
      Step   : in     IR_Step;
      Indent : in     String;
      Status :    out Status_Code)
   is
      Temp_Status : Status_Code;
   begin
      Status := Success;
      Append_To_Code (Code, Indent & 
         Identifier_Strings.To_String (Step.Target) & " = (" &
         Identifier_Strings.To_String (Step.Type_Args) & ")" &
         Identifier_Strings.To_String (Step.Value) & ";" & ASCII.LF, Temp_Status);
   end Emit_Type_Cast_Step_CLike;

   --  =======================================================================
   --  Main Step Emission Dispatcher
   --  =======================================================================

   procedure Emit_All_Steps_CLike
     (Code   : in out Code_String;
      Steps  : in     Step_Collection;
      Indent : in     String;
      Status :    out Status_Code)
   is
      Temp_Status : Status_Code;
   begin
      Status := Success;
      for I in 1 .. Steps.Count loop
         declare
            S : constant IR_Step := Steps.Steps (I);
         begin
            case S.Step_Type is
               when Step_Nop =>
                  null;
               when Step_Assign =>
                  Append_To_Code (Code, Indent & 
                     Identifier_Strings.To_String (S.Target) & " = " & 
                     Identifier_Strings.To_String (S.Value) & ";" & ASCII.LF, Temp_Status);
               when Step_Call =>
                  if Identifier_Strings.Length (S.Target) > 0 then
                     Append_To_Code (Code, Indent & 
                        Identifier_Strings.To_String (S.Target) & " = " & 
                        Identifier_Strings.To_String (S.Value) & ";" & ASCII.LF, Temp_Status);
                  else
                     Append_To_Code (Code, Indent & 
                        Identifier_Strings.To_String (S.Value) & ";" & ASCII.LF, Temp_Status);
                  end if;
               when Step_Return =>
                  null;  --  Handled separately
               when Step_Error =>
                  Append_To_Code (Code, Indent & "/* error: " & 
                     Identifier_Strings.To_String (S.Error_Msg) & " */" & ASCII.LF, Temp_Status);
               when Step_If =>
                  Emit_If_Step_CLike (Code, S, Steps, Indent, Temp_Status);
               when Step_While =>
                  Emit_While_Step_CLike (Code, S, Steps, Indent, Temp_Status);
               when Step_For =>
                  Emit_For_Step_CLike (Code, S, Steps, Indent, Temp_Status);
               when Step_Break =>
                  Append_To_Code (Code, Indent & "break;" & ASCII.LF, Temp_Status);
               when Step_Continue =>
                  Append_To_Code (Code, Indent & "continue;" & ASCII.LF, Temp_Status);
               when Step_Switch =>
                  Append_To_Code (Code, Indent & "/* switch: " & 
                     Identifier_Strings.To_String (S.Expr) & " */" & ASCII.LF, Temp_Status);
               when Step_Try =>
                  Emit_Try_Step_CLike (Code, S, Steps, Indent, Temp_Status);
               when Step_Throw =>
                  Emit_Throw_Step_CLike (Code, S, Indent, Temp_Status);
               when Step_Array_New | Step_Array_Get | Step_Array_Set | 
                    Step_Array_Push | Step_Array_Pop | Step_Array_Len =>
                  null;  --  Handled in Emit_Array_Ops_CLike
               when Step_Map_New | Step_Map_Get | Step_Map_Set |
                    Step_Map_Delete | Step_Map_Has | Step_Map_Keys =>
                  null;  --  Handled in Emit_Map_Ops_CLike
               when Step_Set_New | Step_Set_Add | Step_Set_Remove |
                    Step_Set_Has | Step_Set_Union | Step_Set_Intersect =>
                  Append_To_Code (Code, Indent & "/* set op */" & ASCII.LF, Temp_Status);
               when Step_Struct_New | Step_Struct_Get | Step_Struct_Set =>
                  Append_To_Code (Code, Indent & "/* struct op */" & ASCII.LF, Temp_Status);
               when Step_Generic_Call =>
                  Emit_Generic_Call_Step_CLike (Code, S, Indent, Temp_Status);
               when Step_Type_Cast =>
                  Emit_Type_Cast_Step_CLike (Code, S, Indent, Temp_Status);
            end case;
            if Temp_Status /= Success then
               Status := Temp_Status;
               return;
            end if;
         end;
      end loop;
      --  Emit array and map ops
      Emit_Array_Ops_CLike (Code, Steps, Indent, Temp_Status);
      if Temp_Status /= Success then
         Status := Temp_Status;
         return;
      end if;
      Emit_Map_Ops_CLike (Code, Steps, Indent, Temp_Status);
   end Emit_All_Steps_CLike;

   procedure Emit_Return_For_CLike
     (Code         : in out Code_String;
      Target       : in     Target_Language;
      Return_Type  : in     Type_Name_String;
      Return_Value : in     Identifier_String;
      Status       :    out Status_Code)
   is
      Temp_Status : Status_Code;
      Mapped_Type : Type_Name_String;
      Val_Str     : constant String := Identifier_Strings.To_String (Return_Value);
   begin
      Status := Success;
      Map_Type_To_Target (Return_Type, Target, Mapped_Type, Temp_Status);
      if Temp_Status /= Success then
         Status := Temp_Status;
         return;
      end if;
      declare
         Ret_Str : constant String := Type_Name_Strings.To_String (Mapped_Type);
      begin
         if Ret_Str'Length = 0 or else Ret_Str = "void" then
            return;
         end if;

         if Val_Str'Length > 0 then
            Append_To_Code (Code, "    return " & Val_Str & ";" & ASCII.LF, Temp_Status);
         elsif Ret_Str = "double" or else Ret_Str = "float" then
            Append_To_Code (Code, "    return 0.0;" & ASCII.LF, Temp_Status);
         elsif Ret_Str = "bool" then
            Append_To_Code (Code, "    return false;" & ASCII.LF, Temp_Status);
         elsif Ret_Str = "char*" then
            Append_To_Code (Code, "    return NULL;" & ASCII.LF, Temp_Status);
         elsif Ret_Str = "std::string" then
            Append_To_Code (Code, "    return std::string{};" & ASCII.LF, Temp_Status);
         else
            Append_To_Code (Code, "    return 0;" & ASCII.LF, Temp_Status);
         end if;

         if Temp_Status /= Success then
            Status := Temp_Status;
            return;
         end if;
      end;
   end Emit_Return_For_CLike;

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

            --  Function body (from steps)
            declare
               Has_Return   : Boolean;
               Return_Value : Identifier_String;
            begin
               Emit_Step_Comments (Code, Func.Steps, "    ", Temp_Status);
               if Temp_Status /= Success then
                  Status := Temp_Status;
                  return;
               end if;

               Emit_Assign_Steps_CLike (Code, Func.Steps, Temp_Status);
               if Temp_Status /= Success then
                  Status := Temp_Status;
                  return;
               end if;

               Emit_Call_Steps_CLike (Code, Func.Steps, Temp_Status);
               if Temp_Status /= Success then
                  Status := Temp_Status;
                  return;
               end if;

               Find_Return_Step (Func.Steps, Has_Return, Return_Value);
               if Has_Return then
                  Emit_Return_For_CLike (Code, Target_CPP, Func.Return_Type, Return_Value, Temp_Status);
                  if Temp_Status /= Success then
                     Status := Temp_Status;
                     return;
                  end if;
               else
                  Append_To_Code (Code, "    // TODO: Implement function body" & ASCII.LF, Temp_Status);
                  Emit_Return_For_CLike (Code, Target_CPP, Func.Return_Type, Null_Identifier, Temp_Status);
                  if Temp_Status /= Success then
                     Status := Temp_Status;
                     return;
                  end if;
               end if;
            end;
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
            declare
               Has_Return   : Boolean;
               Return_Value : Identifier_String;
            begin
               Emit_Step_Comments (Code, Func.Steps, "    # ", Temp_Status);
               if Temp_Status /= Success then
                  Status := Temp_Status;
                  return;
               end if;
               Emit_Assign_Steps (Code, Func.Steps, " = ", "" & ASCII.LF, Temp_Status);
               if Temp_Status /= Success then
                  Status := Temp_Status;
                  return;
               end if;

               Emit_Call_Steps (Code, Func.Steps, " = ", "" & ASCII.LF, Temp_Status);
               if Temp_Status /= Success then
                  Status := Temp_Status;
                  return;
               end if;
               Find_Return_Step (Func.Steps, Has_Return, Return_Value);
               if Has_Return then
                  if Identifier_Strings.Length (Return_Value) > 0 then
                     Append_To_Code (Code, "    return " & Identifier_Strings.To_String (Return_Value) & ASCII.LF, Temp_Status);
                  else
                     Append_To_Code (Code, "    return None" & ASCII.LF, Temp_Status);
                  end if;
               else
                  Append_To_Code (Code, "    # TODO: Implement function body" & ASCII.LF, Temp_Status);
                  Append_To_Code (Code, "    pass" & ASCII.LF, Temp_Status);
               end if;
               Append_To_Code (Code, "" & ASCII.LF, Temp_Status);
            end;

         when Target_Clojure | Target_ClojureScript =>
            --  Clojure-style function
            Append_To_Code (Code, "(defn ", Temp_Status);
            Append_To_Code (Code, Identifier_Strings.To_String (Func.Name), Temp_Status);
            Append_To_Code (Code, " [", Temp_Status);
            for I in 1 .. Func.Parameters.Count loop
               declare
                  Param : constant Parameter := Func.Parameters.Params (I);
               begin
                  Append_To_Code (Code, Identifier_Strings.To_String (Param.Name), Temp_Status);
                  if I < Func.Parameters.Count then
                     Append_To_Code (Code, " ", Temp_Status);
                  end if;
               end;
            end loop;
            Append_To_Code (Code, "]" & ASCII.LF, Temp_Status);
            declare
               Has_Return   : Boolean;
               Return_Value : Identifier_String;
            begin
               Emit_Step_Comments (Code, Func.Steps, "  ;; ", Temp_Status);
               if Temp_Status /= Success then
                  Status := Temp_Status;
                  return;
               end if;
               Emit_Assign_Steps (Code, Func.Steps, " ", "" & ASCII.LF, Temp_Status);
               if Temp_Status /= Success then
                  Status := Temp_Status;
                  return;
               end if;

               Emit_Call_Steps (Code, Func.Steps, " ", "" & ASCII.LF, Temp_Status);
               if Temp_Status /= Success then
                  Status := Temp_Status;
                  return;
               end if;
               Find_Return_Step (Func.Steps, Has_Return, Return_Value);
               if Has_Return then
                  if Identifier_Strings.Length (Return_Value) > 0 then
                     Append_To_Code (Code, "  " & Identifier_Strings.To_String (Return_Value) & ASCII.LF, Temp_Status);
                  else
                     Append_To_Code (Code, "  nil" & ASCII.LF, Temp_Status);
                  end if;
               else
                  Append_To_Code (Code, "  ;; TODO: Implement function body" & ASCII.LF, Temp_Status);
                  Append_To_Code (Code, "  nil" & ASCII.LF, Temp_Status);
               end if;
            end;
            Append_To_Code (Code, ")" & ASCII.LF & ASCII.LF, Temp_Status);

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
            declare
               Has_Return   : Boolean;
               Return_Value : Identifier_String;
            begin
               Emit_Step_Comments (Code, Func.Steps, "    ", Temp_Status);
               if Temp_Status /= Success then
                  Status := Temp_Status;
                  return;
               end if;

               Emit_Assign_Steps_CLike (Code, Func.Steps, Temp_Status);
               if Temp_Status /= Success then
                  Status := Temp_Status;
                  return;
               end if;

               Emit_Call_Steps_CLike (Code, Func.Steps, Temp_Status);
               if Temp_Status /= Success then
                  Status := Temp_Status;
                  return;
               end if;

               Find_Return_Step (Func.Steps, Has_Return, Return_Value);
               if Has_Return then
                  if Identifier_Strings.Length (Return_Value) > 0 then
                     Append_To_Code (Code, "    return " & Identifier_Strings.To_String (Return_Value) & ";" & ASCII.LF, Temp_Status);
                  else
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
                  end if;
               else
                  Append_To_Code (Code, "    // TODO: Implement function body" & ASCII.LF, Temp_Status);
               end if;
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
            declare
               Has_Return   : Boolean;
               Return_Value : Identifier_String;
            begin
               Emit_Step_Comments (Code, Func.Steps, "    ", Temp_Status);
               if Temp_Status /= Success then
                  Status := Temp_Status;
                  return;
               end if;

               Emit_Assign_Steps_CLike (Code, Func.Steps, Temp_Status);
               if Temp_Status /= Success then
                  Status := Temp_Status;
                  return;
               end if;

               Emit_Call_Steps_CLike (Code, Func.Steps, Temp_Status);
               if Temp_Status /= Success then
                  Status := Temp_Status;
                  return;
               end if;

               Find_Return_Step (Func.Steps, Has_Return, Return_Value);
               if Has_Return then
                  if Identifier_Strings.Length (Return_Value) > 0 then
                     Append_To_Code (Code, "    return " & Identifier_Strings.To_String (Return_Value) & ASCII.LF, Temp_Status);
                  else
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
                  end if;
               else
                  Append_To_Code (Code, "    // TODO: Implement function body" & ASCII.LF, Temp_Status);
               end if;
            end;
            Append_To_Code (Code, "}" & ASCII.LF & ASCII.LF, Temp_Status);

         when Target_Futhark =>
            --  Futhark: let name (args) : type = expr
            Append_To_Code (Code, "let ", Temp_Status);
            Append_To_Code (Code, Identifier_Strings.To_String (Func.Name), Temp_Status);
            for I in 1 .. Func.Parameters.Count loop
               declare
                  Param : constant Parameter := Func.Parameters.Params (I);
               begin
                  Append_To_Code (Code, " ", Temp_Status);
                  Append_To_Code (Code, Identifier_Strings.To_String (Param.Name), Temp_Status);
               end;
            end loop;
            declare
               Mapped_Type : Type_Name_String;
            begin
               Map_Type_To_Target (Func.Return_Type, Target_Futhark, Mapped_Type, Temp_Status);
               if Temp_Status /= Success then
                  Status := Temp_Status;
                  return;
               end if;
               Append_To_Code (Code, " : " & Type_Name_Strings.To_String (Mapped_Type), Temp_Status);
            end;
            Append_To_Code (Code, " =" & ASCII.LF, Temp_Status);
            declare
               Has_Return   : Boolean;
               Return_Value : Identifier_String;
            begin
               Emit_Step_Comments (Code, Func.Steps, "  -- ", Temp_Status);
               if Temp_Status /= Success then
                  Status := Temp_Status;
                  return;
               end if;
               Emit_Assign_Steps (Code, Func.Steps, " = ", "" & ASCII.LF, Temp_Status);
               if Temp_Status /= Success then
                  Status := Temp_Status;
                  return;
               end if;

               Emit_Call_Steps (Code, Func.Steps, " = ", "" & ASCII.LF, Temp_Status);
               if Temp_Status /= Success then
                  Status := Temp_Status;
                  return;
               end if;
               Find_Return_Step (Func.Steps, Has_Return, Return_Value);
               if Has_Return then
                  if Identifier_Strings.Length (Return_Value) > 0 then
                     Append_To_Code (Code, "  " & Identifier_Strings.To_String (Return_Value) & ASCII.LF, Temp_Status);
                  else
                     Append_To_Code (Code, "  0" & ASCII.LF, Temp_Status);
                  end if;
               else
                  Append_To_Code (Code, "  0" & ASCII.LF, Temp_Status);
               end if;
            end;
            Append_To_Code (Code, "" & ASCII.LF, Temp_Status);

         when Target_Lean4 =>
            Append_To_Code (Code, "def ", Temp_Status);
            Append_To_Code (Code, Identifier_Strings.To_String (Func.Name), Temp_Status);
            for I in 1 .. Func.Parameters.Count loop
               declare
                  Param : constant Parameter := Func.Parameters.Params (I);
                  Mapped_Type : Type_Name_String;
               begin
                  Map_Type_To_Target (Param.Param_Type, Target_Lean4, Mapped_Type, Temp_Status);
                  if Temp_Status /= Success then
                     Status := Temp_Status;
                     return;
                  end if;
                  Append_To_Code (Code, " (" & Identifier_Strings.To_String (Param.Name) & " : " & Type_Name_Strings.To_String (Mapped_Type) & ")", Temp_Status);
               end;
            end loop;
            declare
               Mapped_Type : Type_Name_String;
            begin
               Map_Type_To_Target (Func.Return_Type, Target_Lean4, Mapped_Type, Temp_Status);
               if Temp_Status /= Success then
                  Status := Temp_Status;
                  return;
               end if;
               Append_To_Code (Code, " : " & Type_Name_Strings.To_String (Mapped_Type) & " :=" & ASCII.LF, Temp_Status);
            end;
            declare
               Has_Return   : Boolean;
               Return_Value : Identifier_String;
            begin
               Emit_Step_Comments (Code, Func.Steps, "  -- ", Temp_Status);
               if Temp_Status /= Success then
                  Status := Temp_Status;
                  return;
               end if;
               Emit_Assign_Steps (Code, Func.Steps, " := ", "" & ASCII.LF, Temp_Status);
               if Temp_Status /= Success then
                  Status := Temp_Status;
                  return;
               end if;

               Emit_Call_Steps (Code, Func.Steps, " := ", "" & ASCII.LF, Temp_Status);
               if Temp_Status /= Success then
                  Status := Temp_Status;
                  return;
               end if;
               Find_Return_Step (Func.Steps, Has_Return, Return_Value);
               if Has_Return then
                  if Identifier_Strings.Length (Return_Value) > 0 then
                     Append_To_Code (Code, "  " & Identifier_Strings.To_String (Return_Value) & ASCII.LF, Temp_Status);
                  else
                     Append_To_Code (Code, "  0" & ASCII.LF, Temp_Status);
                  end if;
               else
                  Append_To_Code (Code, "  0" & ASCII.LF, Temp_Status);
               end if;
            end;
            Append_To_Code (Code, "}" & ASCII.LF & ASCII.LF, Temp_Status);

         when Target_JavaScript =>
            --  JavaScript function generation
            Append_To_Code (Code, "function ", Temp_Status);
            if Temp_Status /= Success then
               Status := Temp_Status;
               return;
            end if;

            Append_To_Code (Code, Identifier_Strings.To_String (Func.Name), Temp_Status);
            if Temp_Status /= Success then
               Status := Temp_Status;
               return;
            end if;

            Append_To_Code (Code, "(", Temp_Status);
            for I in 1 .. Func.Parameters.Count loop
               declare
                  Param : constant Parameter := Func.Parameters.Params (I);
               begin
                  Append_To_Code (Code, Identifier_Strings.To_String (Param.Name), Temp_Status);
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

            declare
               Has_Return   : Boolean;
               Return_Value : Identifier_String;
            begin
               Emit_Step_Comments (Code, Func.Steps, "    ", Temp_Status);
               if Temp_Status /= Success then
                  Status := Temp_Status;
                  return;
               end if;

               Emit_Assign_Steps (Code, Func.Steps, " = ", ";" & ASCII.LF, Temp_Status);
               if Temp_Status /= Success then
                  Status := Temp_Status;
                  return;
               end if;

               Emit_Call_Steps (Code, Func.Steps, " = ", ";" & ASCII.LF, Temp_Status);
               if Temp_Status /= Success then
                  Status := Temp_Status;
                  return;
               end if;

               Find_Return_Step (Func.Steps, Has_Return, Return_Value);
               if Has_Return then
                  if Identifier_Strings.Length (Return_Value) > 0 then
                     Append_To_Code (Code, "    return " & Identifier_Strings.To_String (Return_Value) & ";" & ASCII.LF, Temp_Status);
                  else
                     Append_To_Code (Code, "    return;" & ASCII.LF, Temp_Status);
                  end if;
               else
                  Append_To_Code (Code, "    // TODO: Implement function body" & ASCII.LF, Temp_Status);
               end if;
               if Temp_Status /= Success then
                  Status := Temp_Status;
                  return;
               end if;
            end;
            Append_To_Code (Code, "}" & ASCII.LF & ASCII.LF, Temp_Status);

         when Target_CSharp =>
            --  C# function generation
            declare
               Mapped_Type : Type_Name_String;
            begin
               Map_Type_To_Target (Func.Return_Type, Target_CSharp, Mapped_Type, Temp_Status);
               if Temp_Status /= Success then
                  Status := Temp_Status;
                  return;
               end if;
               Append_To_Code (Code, "public static " & Type_Name_Strings.To_String (Mapped_Type) & " ", Temp_Status);
               if Temp_Status /= Success then
                  Status := Temp_Status;
                  return;
               end if;
            end;

            Append_To_Code (Code, Identifier_Strings.To_String (Func.Name), Temp_Status);
            if Temp_Status /= Success then
               Status := Temp_Status;
               return;
            end if;

            Append_To_Code (Code, "(", Temp_Status);
            for I in 1 .. Func.Parameters.Count loop
               declare
                  Param : constant Parameter := Func.Parameters.Params (I);
                  Mapped_Type : Type_Name_String;
               begin
                  Map_Type_To_Target (Param.Param_Type, Target_CSharp, Mapped_Type, Temp_Status);
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

            declare
               Has_Return   : Boolean;
               Return_Value : Identifier_String;
            begin
               Emit_Step_Comments (Code, Func.Steps, "    ", Temp_Status);
               if Temp_Status /= Success then
                  Status := Temp_Status;
                  return;
               end if;

               Emit_Assign_Steps_CLike (Code, Func.Steps, Temp_Status);
               if Temp_Status /= Success then
                  Status := Temp_Status;
                  return;
               end if;

               Emit_Call_Steps_CLike (Code, Func.Steps, Temp_Status);
               if Temp_Status /= Success then
                  Status := Temp_Status;
                  return;
               end if;

               Find_Return_Step (Func.Steps, Has_Return, Return_Value);
               if Has_Return then
                  Emit_Return_For_CLike (Code, Target_CSharp, Func.Return_Type, Return_Value, Temp_Status);
                  if Temp_Status /= Success then
                     Status := Temp_Status;
                     return;
                  end if;
               else
                  Append_To_Code (Code, "    // TODO: Implement function body" & ASCII.LF, Temp_Status);
                  Emit_Return_For_CLike (Code, Target_CSharp, Func.Return_Type, Null_Identifier, Temp_Status);
                  if Temp_Status /= Success then
                     Status := Temp_Status;
                     return;
                  end if;
               end if;
            end;
            Append_To_Code (Code, "}" & ASCII.LF & ASCII.LF, Temp_Status);

         when Target_Java =>
            --  Java function generation
            declare
               Mapped_Type : Type_Name_String;
            begin
               Map_Type_To_Target (Func.Return_Type, Target_Java, Mapped_Type, Temp_Status);
               if Temp_Status /= Success then
                  Status := Temp_Status;
                  return;
               end if;
               Append_To_Code (Code, "public static " & Type_Name_Strings.To_String (Mapped_Type) & " ", Temp_Status);
               if Temp_Status /= Success then
                  Status := Temp_Status;
                  return;
               end if;
            end;

            Append_To_Code (Code, Identifier_Strings.To_String (Func.Name), Temp_Status);
            if Temp_Status /= Success then
               Status := Temp_Status;
               return;
            end if;

            Append_To_Code (Code, "(", Temp_Status);
            for I in 1 .. Func.Parameters.Count loop
               declare
                  Param : constant Parameter := Func.Parameters.Params (I);
                  Mapped_Type : Type_Name_String;
               begin
                  Map_Type_To_Target (Param.Param_Type, Target_Java, Mapped_Type, Temp_Status);
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

            declare
               Has_Return   : Boolean;
               Return_Value : Identifier_String;
            begin
               Emit_Step_Comments (Code, Func.Steps, "    ", Temp_Status);
               if Temp_Status /= Success then
                  Status := Temp_Status;
                  return;
               end if;

               Emit_Assign_Steps_CLike (Code, Func.Steps, Temp_Status);
               if Temp_Status /= Success then
                  Status := Temp_Status;
                  return;
               end if;

               Emit_Call_Steps_CLike (Code, Func.Steps, Temp_Status);
               if Temp_Status /= Success then
                  Status := Temp_Status;
                  return;
               end if;

               Find_Return_Step (Func.Steps, Has_Return, Return_Value);
               if Has_Return then
                  Emit_Return_For_CLike (Code, Target_Java, Func.Return_Type, Return_Value, Temp_Status);
                  if Temp_Status /= Success then
                     Status := Temp_Status;
                     return;
                  end if;
               else
                  Append_To_Code (Code, "    // TODO: Implement function body" & ASCII.LF, Temp_Status);
                  Emit_Return_For_CLike (Code, Target_Java, Func.Return_Type, Null_Identifier, Temp_Status);
                  if Temp_Status /= Success then
                     Status := Temp_Status;
                     return;
                  end if;
               end if;
            end;
            Append_To_Code (Code, "}" & ASCII.LF & ASCII.LF, Temp_Status);

         when Target_C =>
            --  C function generation (similar to C++ but C-style)
            declare
               Mapped_Type : Type_Name_String;
            begin
               Map_Type_To_Target (Func.Return_Type, Target_C, Mapped_Type, Temp_Status);
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

            Append_To_Code (Code, Identifier_Strings.To_String (Func.Name), Temp_Status);
            if Temp_Status /= Success then
               Status := Temp_Status;
               return;
            end if;

            Append_To_Code (Code, "(", Temp_Status);
            for I in 1 .. Func.Parameters.Count loop
               declare
                  Param : constant Parameter := Func.Parameters.Params (I);
                  Mapped_Type : Type_Name_String;
               begin
                  Map_Type_To_Target (Param.Param_Type, Target_C, Mapped_Type, Temp_Status);
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

            declare
               Has_Return   : Boolean;
               Return_Value : Identifier_String;
            begin
               Emit_Step_Comments (Code, Func.Steps, "    ", Temp_Status);
               if Temp_Status /= Success then
                  Status := Temp_Status;
                  return;
               end if;

               Emit_Assign_Steps_CLike (Code, Func.Steps, Temp_Status);
               if Temp_Status /= Success then
                  Status := Temp_Status;
                  return;
               end if;

               Emit_Call_Steps_CLike (Code, Func.Steps, Temp_Status);
               if Temp_Status /= Success then
                  Status := Temp_Status;
                  return;
               end if;

               Find_Return_Step (Func.Steps, Has_Return, Return_Value);
               if Has_Return then
                  Emit_Return_For_CLike (Code, Target_C, Func.Return_Type, Return_Value, Temp_Status);
                  if Temp_Status /= Success then
                     Status := Temp_Status;
                     return;
                  end if;
               else
                  Append_To_Code (Code, "    /* TODO: Implement function body */" & ASCII.LF, Temp_Status);
                  Emit_Return_For_CLike (Code, Target_C, Func.Return_Type, Null_Identifier, Temp_Status);
                  if Temp_Status /= Success then
                     Status := Temp_Status;
                     return;
                  end if;
               end if;
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
      if IR.Functions.Count > 0 then
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
      else
         --  No functions - generate a placeholder
         case Target is
            when Target_Python =>
               Append_To_Code (Complete, "# No functions to generate" & ASCII.LF, Temp_Status);
            when Target_Clojure | Target_ClojureScript =>
               Append_To_Code (Complete, ";; No functions to generate" & ASCII.LF, Temp_Status);
            when Target_Futhark =>
               Append_To_Code (Complete, "-- No functions to generate" & ASCII.LF, Temp_Status);
            when Target_Lean4 =>
               Append_To_Code (Complete, "-- No functions to generate" & ASCII.LF, Temp_Status);
            when Target_C =>
               Append_To_Code (Complete, "/* No functions to generate */" & ASCII.LF, Temp_Status);
            when others =>
               Append_To_Code (Complete, "// No functions to generate" & ASCII.LF, Temp_Status);
         end case;
      end if;

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
      Parser : Parser_State;
      Temp_Status : Status_Code;
   begin
      --  Field-by-field init to avoid Dynamic_Predicate violation on IR_Function
      IR.Schema_Version  := Identifier_Strings.Null_Bounded_String;
      IR.IR_Version      := Identifier_Strings.Null_Bounded_String;
      IR.Module_Name     := Identifier_Strings.Null_Bounded_String;
      IR.Functions.Count := 0;

      --  Use simple JSON parsing to extract module name
      Initialize_Parser (Parser, JSON_Content, Temp_Status);
      if Temp_Status /= Success then
         Status := Error_Parse;
         return;
      end if;

      --  Expect object start
      Next_Token (Parser, Temp_Status);
      if Temp_Status /= Success or else
         Current_Token (Parser) /= Token_Object_Start then
         Status := Error_Parse;
         return;
      end if;

      --  Parse root object members
      Next_Token (Parser, Temp_Status);
      while Temp_Status = Success and then
            Current_Token (Parser) /= Token_Object_End loop
         declare
            Member_Name  : Identifier_String;
            Member_Value : JSON_String;
         begin
            Parse_String_Member (Parser, Member_Name, Member_Value, Temp_Status);
            if Temp_Status /= Success then
               Status := Error_Parse;
               return;
            end if;

            declare
               Name_Str : constant String := Identifier_Strings.To_String (Member_Name);
            begin
               if Name_Str = "module_name" then
                  IR.Module_Name := Identifier_Strings.To_Bounded_String (JSON_Strings.To_String (Member_Value));
                  Next_Token (Parser, Temp_Status);
               elsif Name_Str = "ir_version" then
                  IR.IR_Version := Identifier_Strings.To_Bounded_String (JSON_Strings.To_String (Member_Value));
                  Next_Token (Parser, Temp_Status);
               elsif Name_Str = "schema_version" then
                  IR.Schema_Version := Identifier_Strings.To_Bounded_String (JSON_Strings.To_String (Member_Value));
                  Next_Token (Parser, Temp_Status);
               elsif Name_Str = "schema" then
                  --  Accept schema tag but ignore for emission
                  Next_Token (Parser, Temp_Status);
               elsif Name_Str = "functions" and then Current_Token (Parser) = Token_Array_Start then
                  --  Parse functions array
                  IR.Functions.Count := 0;
                  Next_Token (Parser, Temp_Status);
                  while Temp_Status = Success and then Current_Token (Parser) /= Token_Array_End loop
                     if Current_Token (Parser) /= Token_Object_Start then
                        Status := Error_Parse;
                        return;
                     end if;
                     if IR.Functions.Count >= Max_Functions then
                        Status := Error_Too_Large;
                        return;
                     end if;

                     IR.Functions.Count := IR.Functions.Count + 1;
                     declare
                        Func : IR_Function;
                     begin
                        --  Initialize function record with all fields
                        Func.Name := Null_Identifier;
                        Func.Return_Type := Null_Type_Name;
                        Func.Parameters := (Count => 0, Params => (others => (Name => Null_Identifier, Param_Type => Null_Type_Name)));
                        --  Initialize steps procedurally to avoid stack overflow
                        Init_Step_Collection (Func.Steps);

                        Next_Token (Parser, Temp_Status);
                        while Temp_Status = Success and then Current_Token (Parser) /= Token_Object_End loop
                           declare
                              F_Name  : Identifier_String;
                              F_Value : JSON_String;
                           begin
                              Parse_String_Member (Parser, F_Name, F_Value, Temp_Status);
                              exit when Temp_Status /= Success;

                              declare
                                 Key : constant String := Identifier_Strings.To_String (F_Name);
                              begin
                                 if Key = "name" then
                                    Func.Name := Identifier_Strings.To_Bounded_String (JSON_Strings.To_String (F_Value));
                                    Next_Token (Parser, Temp_Status);
                                 elsif Key = "return_type" then
                                    Func.Return_Type := Type_Name_Strings.To_Bounded_String (JSON_Strings.To_String (F_Value));
                                    Next_Token (Parser, Temp_Status);
                                 elsif (Key = "parameters" or else Key = "args") and then Current_Token (Parser) = Token_Array_Start then
                                    Next_Token (Parser, Temp_Status);
                                    while Temp_Status = Success and then Current_Token (Parser) /= Token_Array_End loop
                                       if Current_Token (Parser) /= Token_Object_Start then
                                          Status := Error_Parse;
                                          return;
                                       end if;
                                       if Func.Parameters.Count >= Max_Parameters then
                                          Status := Error_Too_Large;
                                          return;
                                       end if;
                                       Func.Parameters.Count := Func.Parameters.Count + 1;
                                       declare
                                          Param_Name : Identifier_String := Null_Identifier;
                                          Param_Type : Type_Name_String := Null_Type_Name;
                                       begin
                                          Next_Token (Parser, Temp_Status);
                                          while Temp_Status = Success and then Current_Token (Parser) /= Token_Object_End loop
                                             declare
                                                P_Name  : Identifier_String;
                                                P_Value : JSON_String;
                                             begin
                                                Parse_String_Member (Parser, P_Name, P_Value, Temp_Status);
                                                exit when Temp_Status /= Success;
                                                declare
                                                   P_Key : constant String := Identifier_Strings.To_String (P_Name);
                                                begin
                                                   if P_Key = "name" then
                                                      Param_Name := Identifier_Strings.To_Bounded_String (JSON_Strings.To_String (P_Value));
                                                      Next_Token (Parser, Temp_Status);
                                                   elsif P_Key = "type" then
                                                      Param_Type := Type_Name_Strings.To_Bounded_String (JSON_Strings.To_String (P_Value));
                                                      Next_Token (Parser, Temp_Status);
                                                   else
                                                      if Current_Token (Parser) = Token_Array_Start
                                                         or Current_Token (Parser) = Token_Object_Start
                                                      then
                                                         Skip_Value (Parser, Temp_Status);
                                                      else
                                                         Next_Token (Parser, Temp_Status);
                                                      end if;
                                                   end if;
                                                end;
                                                if Current_Token (Parser) = Token_Comma then
                                                   Next_Token (Parser, Temp_Status);
                                                end if;
                                             end;
                                          end loop;
                                          if Current_Token (Parser) = Token_Object_End then
                                             Next_Token (Parser, Temp_Status);
                                          end if;
                                          Func.Parameters.Params (Func.Parameters.Count) := (Name => Param_Name, Param_Type => Param_Type);
                                       end;
                                       if Current_Token (Parser) = Token_Comma then
                                          Next_Token (Parser, Temp_Status);
                                       end if;
                                    end loop;
                                    if Current_Token (Parser) = Token_Array_End then
                                       Next_Token (Parser, Temp_Status);
                                    end if;
                                 elsif Key = "steps" and then Current_Token (Parser) = Token_Array_Start then
                                    Next_Token (Parser, Temp_Status);
                                    while Temp_Status = Success and then Current_Token (Parser) /= Token_Array_End loop
                                       if Current_Token (Parser) /= Token_Object_Start then
                                          Status := Error_Parse;
                                          return;
                                       end if;
                                       if Func.Steps.Count >= Max_Steps then
                                          Status := Error_Too_Large;
                                          return;
                                       end if;
                                       Func.Steps.Count := Func.Steps.Count + 1;
                                       declare
                                          --  Initialize step with Default_Step
                                          Step : IR_Step := Make_Default_Step;
                                       begin
                                          Next_Token (Parser, Temp_Status);
                                          while Temp_Status = Success and then Current_Token (Parser) /= Token_Object_End loop
                                             declare
                                                S_Name  : Identifier_String;
                                                S_Value : JSON_String;
                                             begin
                                                Parse_String_Member (Parser, S_Name, S_Value, Temp_Status);
                                                exit when Temp_Status /= Success;
                                                declare
                                                   S_Key : constant String := Identifier_Strings.To_String (S_Name);
                                                   Val  : constant String := JSON_Strings.To_String (S_Value);
                                                begin
                                                   if S_Key = "op" then
                                                      --  Map op string to Step_Type_Enum
                                                      if Val = "nop" then
                                                         Step.Step_Type := Step_Nop;
                                                      elsif Val = "assign" then
                                                         Step.Step_Type := Step_Assign;
                                                      elsif Val = "call" then
                                                         Step.Step_Type := Step_Call;
                                                      elsif Val = "return" then
                                                         Step.Step_Type := Step_Return;
                                                      elsif Val = "error" then
                                                         Step.Step_Type := Step_Error;
                                                      --  Control flow
                                                      elsif Val = "if" then
                                                         Step.Step_Type := Step_If;
                                                      elsif Val = "while" then
                                                         Step.Step_Type := Step_While;
                                                      elsif Val = "for" then
                                                         Step.Step_Type := Step_For;
                                                      elsif Val = "break" then
                                                         Step.Step_Type := Step_Break;
                                                      elsif Val = "continue" then
                                                         Step.Step_Type := Step_Continue;
                                                      elsif Val = "switch" then
                                                         Step.Step_Type := Step_Switch;
                                                      --  Exceptions
                                                      elsif Val = "try" then
                                                         Step.Step_Type := Step_Try;
                                                      elsif Val = "throw" then
                                                         Step.Step_Type := Step_Throw;
                                                      --  Array ops
                                                      elsif Val = "array_new" then
                                                         Step.Step_Type := Step_Array_New;
                                                      elsif Val = "array_get" then
                                                         Step.Step_Type := Step_Array_Get;
                                                      elsif Val = "array_set" then
                                                         Step.Step_Type := Step_Array_Set;
                                                      elsif Val = "array_push" then
                                                         Step.Step_Type := Step_Array_Push;
                                                      elsif Val = "array_pop" then
                                                         Step.Step_Type := Step_Array_Pop;
                                                      elsif Val = "array_len" then
                                                         Step.Step_Type := Step_Array_Len;
                                                      --  Map ops
                                                      elsif Val = "map_new" then
                                                         Step.Step_Type := Step_Map_New;
                                                      elsif Val = "map_get" then
                                                         Step.Step_Type := Step_Map_Get;
                                                      elsif Val = "map_set" then
                                                         Step.Step_Type := Step_Map_Set;
                                                      elsif Val = "map_delete" then
                                                         Step.Step_Type := Step_Map_Delete;
                                                      elsif Val = "map_has" then
                                                         Step.Step_Type := Step_Map_Has;
                                                      elsif Val = "map_keys" then
                                                         Step.Step_Type := Step_Map_Keys;
                                                      --  Set ops
                                                      elsif Val = "set_new" then
                                                         Step.Step_Type := Step_Set_New;
                                                      elsif Val = "set_add" then
                                                         Step.Step_Type := Step_Set_Add;
                                                      elsif Val = "set_remove" then
                                                         Step.Step_Type := Step_Set_Remove;
                                                      elsif Val = "set_has" then
                                                         Step.Step_Type := Step_Set_Has;
                                                      elsif Val = "set_union" then
                                                         Step.Step_Type := Step_Set_Union;
                                                      elsif Val = "set_intersect" then
                                                         Step.Step_Type := Step_Set_Intersect;
                                                      --  Struct ops
                                                      elsif Val = "struct_new" then
                                                         Step.Step_Type := Step_Struct_New;
                                                      elsif Val = "struct_get" then
                                                         Step.Step_Type := Step_Struct_Get;
                                                      elsif Val = "struct_set" then
                                                         Step.Step_Type := Step_Struct_Set;
                                                      --  Generic call / type cast
                                                      elsif Val = "generic_call" then
                                                         Step.Step_Type := Step_Generic_Call;
                                                      elsif Val = "type_cast" then
                                                         Step.Step_Type := Step_Type_Cast;
                                                      else
                                                         Step.Step_Type := Step_Nop;
                                                      end if;
                                                      Next_Token (Parser, Temp_Status);
                                                   elsif S_Key = "target" then
                                                      Step.Target := Identifier_Strings.To_Bounded_String (Val);
                                                      Next_Token (Parser, Temp_Status);
                                                   elsif S_Key = "value" then
                                                      Step.Value := Identifier_Strings.To_Bounded_String (Val);
                                                      Next_Token (Parser, Temp_Status);
                                                   elsif S_Key = "error_msg" or S_Key = "message" then
                                                      Step.Error_Msg := Identifier_Strings.To_Bounded_String (Val);
                                                      Next_Token (Parser, Temp_Status);
                                                   elsif S_Key = "condition" or S_Key = "cond" then
                                                      Step.Condition := Identifier_Strings.To_Bounded_String (Val);
                                                      Next_Token (Parser, Temp_Status);
                                                   elsif S_Key = "init" then
                                                      Step.Init := Identifier_Strings.To_Bounded_String (Val);
                                                      Next_Token (Parser, Temp_Status);
                                                   elsif S_Key = "increment" or S_Key = "incr" then
                                                      Step.Increment := Identifier_Strings.To_Bounded_String (Val);
                                                      Next_Token (Parser, Temp_Status);
                                                   elsif S_Key = "expr" then
                                                      Step.Expr := Identifier_Strings.To_Bounded_String (Val);
                                                      Next_Token (Parser, Temp_Status);
                                                   elsif S_Key = "index" then
                                                      Step.Index := Identifier_Strings.To_Bounded_String (Val);
                                                      Next_Token (Parser, Temp_Status);
                                                   elsif S_Key = "key" then
                                                      Step.Key := Identifier_Strings.To_Bounded_String (Val);
                                                      Next_Token (Parser, Temp_Status);
                                                   elsif S_Key = "field" then
                                                      Step.Field := Identifier_Strings.To_Bounded_String (Val);
                                                      Next_Token (Parser, Temp_Status);
                                                   elsif S_Key = "args" then
                                                      Step.Args := Identifier_Strings.To_Bounded_String (Val);
                                                      Next_Token (Parser, Temp_Status);
                                                   elsif S_Key = "type_args" or S_Key = "type" then
                                                      Step.Type_Args := Identifier_Strings.To_Bounded_String (Val);
                                                      Next_Token (Parser, Temp_Status);
                                                   else
                                                      if Current_Token (Parser) = Token_Array_Start
                                                         or Current_Token (Parser) = Token_Object_Start
                                                      then
                                                         Skip_Value (Parser, Temp_Status);
                                                      else
                                                         Next_Token (Parser, Temp_Status);
                                                      end if;
                                                   end if;
                                                end;
                                                if Current_Token (Parser) = Token_Comma then
                                                   Next_Token (Parser, Temp_Status);
                                                end if;
                                             end;
                                          end loop;
                                          if Current_Token (Parser) = Token_Object_End then
                                             Next_Token (Parser, Temp_Status);
                                          end if;
                                          Func.Steps.Steps (Func.Steps.Count) := Step;
                                       end;
                                       if Current_Token (Parser) = Token_Comma then
                                          Next_Token (Parser, Temp_Status);
                                       end if;
                                    end loop;
                                    if Current_Token (Parser) = Token_Array_End then
                                       Next_Token (Parser, Temp_Status);
                                    end if;
                                 else
                                    if Current_Token (Parser) = Token_Array_Start
                                       or Current_Token (Parser) = Token_Object_Start
                                    then
                                       Skip_Value (Parser, Temp_Status);
                                    else
                                       Next_Token (Parser, Temp_Status);
                                    end if;
                                 end if;
                              end;

                              if Current_Token (Parser) = Token_Comma then
                                 Next_Token (Parser, Temp_Status);
                              end if;
                           end;
                        end loop;

                        if Current_Token (Parser) = Token_Object_End then
                           Next_Token (Parser, Temp_Status);
                        end if;
                        IR.Functions.Functions (IR.Functions.Count) := Func;
                     end;

                     if Current_Token (Parser) = Token_Comma then
                        Next_Token (Parser, Temp_Status);
                     end if;
                  end loop;
                  if Current_Token (Parser) = Token_Array_End then
                     Next_Token (Parser, Temp_Status);
                  end if;
               else
                  --  Unknown member - skip its value
                  if Current_Token (Parser) = Token_Array_Start
                     or Current_Token (Parser) = Token_Object_Start
                  then
                     Skip_Value (Parser, Temp_Status);
                  else
                     Next_Token (Parser, Temp_Status);
                  end if;
               end if;
            end;

            --  Check for comma
            if Current_Token (Parser) = Token_Comma then
               Next_Token (Parser, Temp_Status);
            end if;
         end;
      end loop;

      --  Set defaults if not found
      if Identifier_Strings.Length (IR.Module_Name) = 0 then
         IR.Module_Name := Identifier_Strings.To_Bounded_String ("module");
      end if;
      if Identifier_Strings.Length (IR.IR_Version) = 0 then
         IR.IR_Version := Identifier_Strings.To_Bounded_String ("1.0");
      end if;
      if Identifier_Strings.Length (IR.Schema_Version) = 0 then
         IR.Schema_Version := Identifier_Strings.To_Bounded_String ("1.0");
      end if;

      --  Note: Functions are not parsed from JSON for simplicity.
      --  The code generator will handle empty function list.

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

      --  Use IR-based emission for all targets to ensure step bodies are honored.
      Generate_All_Code (IR, Target, Code, Status);

      --  Ensure output directory exists and construct output path
      declare
         Out_Dir  : constant String := Path_Strings.To_String (Output_Dir);
         Mod_Name : constant String := Identifier_Strings.To_String (IR.Module_Name);
         Ext      : constant String := Identifier_Strings.To_String (Get_File_Extension (Target));
         Full_Path : constant String := Out_Dir & "/" & Mod_Name & Ext;
      begin
         if Out_Dir'Length = 0 then
            Status := Error_Invalid_Input;
            return;
         end if;

         begin
            if not Ada.Directories.Exists (Out_Dir) then
               Ada.Directories.Create_Path (Out_Dir);
            end if;
         exception
            when others =>
               Status := Error_File_IO;
               return;
         end;

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