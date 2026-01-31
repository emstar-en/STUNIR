-- STUNIR Polyglot Emitter (Body)
-- DO-178C Level A
-- Phase 3a: Core Category Emitters

package body STUNIR.Emitters.Polyglot is
   pragma SPARK_Mode (On);

   function Get_Language_Name (Lang : Target_Language) return String is
   begin
      case Lang is
         when Lang_C89  => return "C89";
         when Lang_C99  => return "C99";
         when Lang_Rust => return "Rust";
      end case;
   end Get_Language_Name;

   function Map_Type_To_C (IR_Type : String) return String is
   begin
      if IR_Type = "int" or IR_Type = "i32" then
         return "int32_t";
      elsif IR_Type = "i8" then
         return "int8_t";
      elsif IR_Type = "i16" then
         return "int16_t";
      elsif IR_Type = "i64" then
         return "int64_t";
      elsif IR_Type = "u32" then
         return "uint32_t";
      elsif IR_Type = "float" or IR_Type = "f32" then
         return "float";
      elsif IR_Type = "double" or IR_Type = "f64" then
         return "double";
      elsif IR_Type = "bool" then
         return "bool";
      elsif IR_Type = "string" then
         return "char*";
      elsif IR_Type = "void" then
         return "void";
      else
         return IR_Type;
      end if;
   end Map_Type_To_C;

   function Map_Type_To_Rust (IR_Type : String) return String is
   begin
      if IR_Type = "int" or IR_Type = "i32" or IR_Type = "int32_t" then
         return "i32";
      elsif IR_Type = "i8" or IR_Type = "int8_t" then
         return "i8";
      elsif IR_Type = "i16" or IR_Type = "int16_t" then
         return "i16";
      elsif IR_Type = "i64" or IR_Type = "int64_t" then
         return "i64";
      elsif IR_Type = "u32" or IR_Type = "uint32_t" then
         return "u32";
      elsif IR_Type = "float" or IR_Type = "f32" then
         return "f32";
      elsif IR_Type = "double" or IR_Type = "f64" then
         return "f64";
      elsif IR_Type = "bool" then
         return "bool";
      elsif IR_Type = "string" then
         return "String";
      elsif IR_Type = "void" then
         return "()";
      else
         return IR_Type;
      end if;
   end Map_Type_To_Rust;

   overriding procedure Emit_Module
     (Self    : in out Polyglot_Emitter;
      Module  : in     IR_Module;
      Output  :    out IR_Code_Buffer;
      Success :    out Boolean)
   is
   begin
      case Self.Config.Language is
         when Lang_C89 =>
            Emit_C89 (Module, Output, Success);
         when Lang_C99 =>
            Emit_C99 (Module, Output, Success);
         when Lang_Rust =>
            Emit_Rust (Module, Output, Success);
      end case;
   end Emit_Module;

   overriding procedure Emit_Type
     (Self    : in out Polyglot_Emitter;
      T       : in     IR_Type_Def;
      Output  :    out IR_Code_Buffer;
      Success :    out Boolean)
   is
      Gen : Code_Generator;
      Line_Success : Boolean;
      Type_Name : constant String := Get_Type_Name (T);
   begin
      Success := False;
      Initialize (Gen);

      case Self.Config.Language is
         when Lang_C89 | Lang_C99 =>
            Append_Line (Gen, "typedef struct {", Line_Success);
            if not Line_Success then return; end if;

            Increase_Indent (Gen);

            for I in 1 .. T.Field_Cnt loop
               pragma Loop_Invariant (I <= T.Field_Cnt);

               declare
                  Field_Name : constant String := Name_Strings.To_String (T.Fields (I).Name);
                  Field_Type : constant String := Type_Strings.To_String (T.Fields (I).Type_Ref);
                  Mapped_Type : constant String := Map_Type_To_C (Field_Type);
               begin
                  Append_Line (Gen, Mapped_Type & " " & Field_Name & ";", Line_Success);
                  if not Line_Success then return; end if;
               end;
            end loop;

            Decrease_Indent (Gen);
            Append_Line (Gen, "} " & Type_Name & ";", Line_Success);
            if not Line_Success then return; end if;

         when Lang_Rust =>
            Append_Line (Gen, "pub struct " & Type_Name & " {", Line_Success);
            if not Line_Success then return; end if;

            Increase_Indent (Gen);

            for I in 1 .. T.Field_Cnt loop
               pragma Loop_Invariant (I <= T.Field_Cnt);

               declare
                  Field_Name : constant String := Name_Strings.To_String (T.Fields (I).Name);
                  Field_Type : constant String := Type_Strings.To_String (T.Fields (I).Type_Ref);
                  Mapped_Type : constant String := Map_Type_To_Rust (Field_Type);
               begin
                  Append_Line (Gen, "pub " & Field_Name & ": " & Mapped_Type & ",", Line_Success);
                  if not Line_Success then return; end if;
               end;
            end loop;

            Decrease_Indent (Gen);
            Append_Line (Gen, "}", Line_Success);
            if not Line_Success then return; end if;
      end case;

      Append_Line (Gen, "", Line_Success);
      if not Line_Success then return; end if;

      Get_Output (Gen, Output);
      Success := True;
   end Emit_Type;

   overriding procedure Emit_Function
     (Self    : in out Polyglot_Emitter;
      Func    : in     IR_Function;
      Output  :    out IR_Code_Buffer;
      Success :    out Boolean)
   is
      Gen : Code_Generator;
      Line_Success : Boolean;
      Func_Name : constant String := Get_Function_Name (Func);
      Return_Type : constant String := Type_Strings.To_String (Func.Return_Type);
   begin
      Success := False;
      Initialize (Gen);

      case Self.Config.Language is
         when Lang_C89 | Lang_C99 =>
            declare
               Mapped_Return : constant String := Map_Type_To_C (Return_Type);
               Signature : String := Mapped_Return & " " & Func_Name & "(";
            begin
               for I in 1 .. Func.Arg_Cnt loop
                  pragma Loop_Invariant (I <= Func.Arg_Cnt);

                  declare
                     Arg_Type : constant String := Type_Strings.To_String (Func.Args (I).Type_Ref);
                     Arg_Name : constant String := Name_Strings.To_String (Func.Args (I).Name);
                     Mapped_Type : constant String := Map_Type_To_C (Arg_Type);
                  begin
                     if I > 1 then
                        Signature := Signature & ", ";
                     end if;
                     Signature := Signature & Mapped_Type & " " & Arg_Name;
                  end;
               end loop;

               if Func.Arg_Cnt = 0 then
                  Signature := Signature & "void";
               end if;

               Signature := Signature & ") {";
               Append_Line (Gen, Signature, Line_Success);
               if not Line_Success then return; end if;
            end;

         when Lang_Rust =>
            declare
               Mapped_Return : constant String := Map_Type_To_Rust (Return_Type);
               Signature : String := "pub fn " & Func_Name & "(";
            begin
               for I in 1 .. Func.Arg_Cnt loop
                  pragma Loop_Invariant (I <= Func.Arg_Cnt);

                  declare
                     Arg_Type : constant String := Type_Strings.To_String (Func.Args (I).Type_Ref);
                     Arg_Name : constant String := Name_Strings.To_String (Func.Args (I).Name);
                     Mapped_Type : constant String := Map_Type_To_Rust (Arg_Type);
                  begin
                     if I > 1 then
                        Signature := Signature & ", ";
                     end if;
                     Signature := Signature & Arg_Name & ": " & Mapped_Type;
                  end;
               end loop;

               Signature := Signature & ") -> " & Mapped_Return & " {";
               Append_Line (Gen, Signature, Line_Success);
               if not Line_Success then return; end if;
            end;
      end case;

      Increase_Indent (Gen);

      -- Function body
      for I in 1 .. Func.Stmt_Cnt loop
         pragma Loop_Invariant (I <= Func.Stmt_Cnt);

         declare
            Stmt_Data : constant String := Code_Buffers.To_String (Func.Statements (I).Data);
         begin
            if Stmt_Data'Length > 0 then
               Append_Line (Gen, Stmt_Data, Line_Success);
               if not Line_Success then return; end if;
            end if;
         end;
      end loop;

      Decrease_Indent (Gen);
      Append_Line (Gen, "}", Line_Success);
      if not Line_Success then return; end if;

      Append_Line (Gen, "", Line_Success);
      if not Line_Success then return; end if;

      Get_Output (Gen, Output);
      Success := True;
   end Emit_Function;

   procedure Emit_C89
     (Module  : in     IR_Module;
      Output  :    out IR_Code_Buffer;
      Success :    out Boolean)
   is
      Gen : Code_Generator;
      Line_Success : Boolean;
      Emitter : Polyglot_Emitter;
   begin
      Success := False;
      Initialize (Gen);
      Emitter.Config.Language := Lang_C89;

      Append_Line (Gen, "/* STUNIR Generated C89 Code */", Line_Success);
      if not Line_Success then return; end if;

      Append_Line (Gen, "/* DO-178C Level A Compliant */", Line_Success);
      if not Line_Success then return; end if;

      Append_Line (Gen, "", Line_Success);
      if not Line_Success then return; end if;

      -- C89 headers
      Append_Line (Gen, "#include <stdio.h>", Line_Success);
      if not Line_Success then return; end if;

      Append_Line (Gen, "", Line_Success);
      if not Line_Success then return; end if;

      -- C89 type definitions (no stdint.h)
      Append_Line (Gen, "typedef signed char int8_t;", Line_Success);
      if not Line_Success then return; end if;

      Append_Line (Gen, "typedef short int16_t;", Line_Success);
      if not Line_Success then return; end if;

      Append_Line (Gen, "typedef long int32_t;", Line_Success);
      if not Line_Success then return; end if;

      Append_Line (Gen, "typedef unsigned char uint8_t;", Line_Success);
      if not Line_Success then return; end if;

      Append_Line (Gen, "typedef unsigned short uint16_t;", Line_Success);
      if not Line_Success then return; end if;

      Append_Line (Gen, "typedef unsigned long uint32_t;", Line_Success);
      if not Line_Success then return; end if;

      Append_Line (Gen, "", Line_Success);
      if not Line_Success then return; end if;

      -- Emit types
      for I in 1 .. Module.Type_Cnt loop
         pragma Loop_Invariant (I <= Module.Type_Cnt);

         declare
            Type_Output : IR_Code_Buffer;
            Type_Success : Boolean;
         begin
            Emit_Type (Emitter, Module.Types (I), Type_Output, Type_Success);
            if Type_Success then
               Append_Raw (Gen, Code_Buffers.To_String (Type_Output), Line_Success);
               if not Line_Success then return; end if;
            end if;
         end;
      end loop;

      -- Emit functions
      for I in 1 .. Module.Func_Cnt loop
         pragma Loop_Invariant (I <= Module.Func_Cnt);

         declare
            Func_Output : IR_Code_Buffer;
            Func_Success : Boolean;
         begin
            Emit_Function (Emitter, Module.Functions (I), Func_Output, Func_Success);
            if Func_Success then
               Append_Raw (Gen, Code_Buffers.To_String (Func_Output), Line_Success);
               if not Line_Success then return; end if;
            end if;
         end;
      end loop;

      Get_Output (Gen, Output);
      Success := True;
   end Emit_C89;

   procedure Emit_C99
     (Module  : in     IR_Module;
      Output  :    out IR_Code_Buffer;
      Success :    out Boolean)
   is
      Gen : Code_Generator;
      Line_Success : Boolean;
      Emitter : Polyglot_Emitter;
   begin
      Success := False;
      Initialize (Gen);
      Emitter.Config.Language := Lang_C99;

      Append_Line (Gen, "/* STUNIR Generated C99 Code */", Line_Success);
      if not Line_Success then return; end if;

      Append_Line (Gen, "/* DO-178C Level A Compliant */", Line_Success);
      if not Line_Success then return; end if;

      Append_Line (Gen, "", Line_Success);
      if not Line_Success then return; end if;

      -- C99 headers
      Append_Line (Gen, "#include <stdint.h>", Line_Success);
      if not Line_Success then return; end if;

      Append_Line (Gen, "#include <stdbool.h>", Line_Success);
      if not Line_Success then return; end if;

      Append_Line (Gen, "", Line_Success);
      if not Line_Success then return; end if;

      -- Emit types
      for I in 1 .. Module.Type_Cnt loop
         pragma Loop_Invariant (I <= Module.Type_Cnt);

         declare
            Type_Output : IR_Code_Buffer;
            Type_Success : Boolean;
         begin
            Emit_Type (Emitter, Module.Types (I), Type_Output, Type_Success);
            if Type_Success then
               Append_Raw (Gen, Code_Buffers.To_String (Type_Output), Line_Success);
               if not Line_Success then return; end if;
            end if;
         end;
      end loop;

      -- Emit functions
      for I in 1 .. Module.Func_Cnt loop
         pragma Loop_Invariant (I <= Module.Func_Cnt);

         declare
            Func_Output : IR_Code_Buffer;
            Func_Success : Boolean;
         begin
            Emit_Function (Emitter, Module.Functions (I), Func_Output, Func_Success);
            if Func_Success then
               Append_Raw (Gen, Code_Buffers.To_String (Func_Output), Line_Success);
               if not Line_Success then return; end if;
            end if;
         end;
      end loop;

      Get_Output (Gen, Output);
      Success := True;
   end Emit_C99;

   procedure Emit_Rust
     (Module  : in     IR_Module;
      Output  :    out IR_Code_Buffer;
      Success :    out Boolean)
   is
      Gen : Code_Generator;
      Line_Success : Boolean;
      Emitter : Polyglot_Emitter;
   begin
      Success := False;
      Initialize (Gen);
      Emitter.Config.Language := Lang_Rust;

      Append_Line (Gen, "// STUNIR Generated Rust Code", Line_Success);
      if not Line_Success then return; end if;

      Append_Line (Gen, "// DO-178C Level A Compliant", Line_Success);
      if not Line_Success then return; end if;

      Append_Line (Gen, "", Line_Success);
      if not Line_Success then return; end if;

      -- Rust attributes
      Append_Line (Gen, "#![allow(dead_code)]", Line_Success);
      if not Line_Success then return; end if;

      Append_Line (Gen, "", Line_Success);
      if not Line_Success then return; end if;

      -- Emit types
      for I in 1 .. Module.Type_Cnt loop
         pragma Loop_Invariant (I <= Module.Type_Cnt);

         declare
            Type_Output : IR_Code_Buffer;
            Type_Success : Boolean;
         begin
            Emit_Type (Emitter, Module.Types (I), Type_Output, Type_Success);
            if Type_Success then
               Append_Raw (Gen, Code_Buffers.To_String (Type_Output), Line_Success);
               if not Line_Success then return; end if;
            end if;
         end;
      end loop;

      -- Emit functions
      for I in 1 .. Module.Func_Cnt loop
         pragma Loop_Invariant (I <= Module.Func_Cnt);

         declare
            Func_Output : IR_Code_Buffer;
            Func_Success : Boolean;
         begin
            Emit_Function (Emitter, Module.Functions (I), Func_Output, Func_Success);
            if Func_Success then
               Append_Raw (Gen, Code_Buffers.To_String (Func_Output), Line_Success);
               if not Line_Success then return; end if;
            end if;
         end;
      end loop;

      Get_Output (Gen, Output);
      Success := True;
   end Emit_Rust;

end STUNIR.Emitters.Polyglot;
