-- STUNIR WebAssembly Emitter (Body)
-- DO-178C Level A
-- Phase 3a: Core Category Emitters

package body STUNIR.Emitters.WASM is
   pragma SPARK_Mode (On);

   function Get_WASM_Type (IR_Type : String) return String is
   begin
      if IR_Type = "int" or IR_Type = "i32" or IR_Type = "int32_t" then
         return "i32";
      elsif IR_Type = "i64" or IR_Type = "int64_t" then
         return "i64";
      elsif IR_Type = "float" or IR_Type = "f32" then
         return "f32";
      elsif IR_Type = "double" or IR_Type = "f64" then
         return "f64";
      else
         return "i32";
      end if;
   end Get_WASM_Type;

   overriding procedure Emit_Module
     (Self    : in out WASM_Emitter;
      Module  : in     IR_Module;
      Output  :    out IR_Code_Buffer;
      Success :    out Boolean)
   is
      Gen : Code_Generator;
      Line_Success : Boolean;
   begin
      Success := False;
      Initialize (Gen);

      Append_Line (Gen, "/* STUNIR Generated WASM Code */", Line_Success);
      if not Line_Success then return; end if;

      if Self.Config.Use_WASI then
         Append_Line (Gen, "/* WASI Enabled */", Line_Success);
         if not Line_Success then return; end if;
      end if;

      Append_Line (Gen, "", Line_Success);
      if not Line_Success then return; end if;

      -- Generate C code that will be compiled to WASM
      Append_Line (Gen, "#include <stdint.h>", Line_Success);
      if not Line_Success then return; end if;

      if Self.Config.Use_WASI then
         Append_Line (Gen, "#include <wasi/api.h>", Line_Success);
         if not Line_Success then return; end if;
      end if;

      Append_Line (Gen, "", Line_Success);
      if not Line_Success then return; end if;

      -- Export macro
      Append_Line (Gen, "#define WASM_EXPORT __attribute__((visibility(""default"")))", Line_Success);
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
            Emit_Type (Self, Module.Types (I), Type_Output, Type_Success);
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
            Emit_Function (Self, Module.Functions (I), Func_Output, Func_Success);
            if Func_Success then
               Append_Raw (Gen, Code_Buffers.To_String (Func_Output), Line_Success);
               if not Line_Success then return; end if;
            end if;
         end;
      end loop;

      Get_Output (Gen, Output);
      Success := True;
   end Emit_Module;

   overriding procedure Emit_Type
     (Self    : in out WASM_Emitter;
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

      Append_Line (Gen, "typedef struct {", Line_Success);
      if not Line_Success then return; end if;

      Increase_Indent (Gen);

      for I in 1 .. T.Field_Cnt loop
         pragma Loop_Invariant (I <= T.Field_Cnt);

         declare
            Field_Name : constant String := Name_Strings.To_String (T.Fields (I).Name);
            Field_Type : constant String := Type_Strings.To_String (T.Fields (I).Type_Ref);
         begin
            Append_Line (Gen, Field_Type & " " & Field_Name & ";", Line_Success);
            if not Line_Success then return; end if;
         end;
      end loop;

      Decrease_Indent (Gen);

      Append_Line (Gen, "} " & Type_Name & ";", Line_Success);
      if not Line_Success then return; end if;

      Append_Line (Gen, "", Line_Success);
      if not Line_Success then return; end if;

      Get_Output (Gen, Output);
      Success := True;
   end Emit_Type;

   overriding procedure Emit_Function
     (Self    : in out WASM_Emitter;
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

      -- Function signature with export
      declare
         Export_Prefix : constant String := (if Self.Config.Export_All then "WASM_EXPORT " else "");
         Signature : String := Export_Prefix & Return_Type & " " & Func_Name & "(";
      begin
         for I in 1 .. Func.Arg_Cnt loop
            pragma Loop_Invariant (I <= Func.Arg_Cnt);

            declare
               Arg_Type : constant String := Type_Strings.To_String (Func.Args (I).Type_Ref);
               Arg_Name : constant String := Name_Strings.To_String (Func.Args (I).Name);
            begin
               if I > 1 then
                  Signature := Signature & ", ";
               end if;
               Signature := Signature & Arg_Type & " " & Arg_Name;
            end;
         end loop;

         if Func.Arg_Cnt = 0 then
            Signature := Signature & "void";
         end if;

         Signature := Signature & ") {";
         Append_Line (Gen, Signature, Line_Success);
         if not Line_Success then return; end if;
      end;

      Increase_Indent (Gen);

      Append_Line (Gen, "/* WASM Function Body */", Line_Success);
      if not Line_Success then return; end if;

      -- Emit statements
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

   procedure Emit_WAT_Module
     (Self    : in out WASM_Emitter;
      Module  : in     IR_Module;
      Output  :    out IR_Code_Buffer;
      Success :    out Boolean)
   is
      Gen : Code_Generator;
      Line_Success : Boolean;
      Module_Name : constant String := Name_Strings.To_String (Module.Module_Name);
   begin
      Success := False;
      Initialize (Gen);

      Append_Line (Gen, "(module", Line_Success);
      if not Line_Success then return; end if;

      Increase_Indent (Gen);

      Append_Line (Gen, ";; STUNIR Generated WebAssembly Text Format", Line_Success);
      if not Line_Success then return; end if;

      Append_Line (Gen, ";; Module: " & Module_Name, Line_Success);
      if not Line_Success then return; end if;

      -- Emit function exports
      for I in 1 .. Module.Func_Cnt loop
         pragma Loop_Invariant (I <= Module.Func_Cnt);

         declare
            Func_Name : constant String := Get_Function_Name (Module.Functions (I));
         begin
            Append_Line (Gen, "(export """ & Func_Name & """ (func $" & Func_Name & "))", Line_Success);
            if not Line_Success then return; end if;
         end;
      end loop;

      Decrease_Indent (Gen);
      Append_Line (Gen, ")", Line_Success);
      if not Line_Success then return; end if;

      Get_Output (Gen, Output);
      Success := True;
   end Emit_WAT_Module;

end STUNIR.Emitters.WASM;
