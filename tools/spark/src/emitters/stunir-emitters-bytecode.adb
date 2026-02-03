-- STUNIR Bytecode Emitter (SPARK Body)
with Ada.Strings; use Ada.Strings;

package body STUNIR.Emitters.Bytecode is
   pragma SPARK_Mode (On);

   overriding procedure Emit_Module (Self : in out Bytecode_Emitter; Module : in IR_Module; Output : out IR_Code_Buffer; Success: out Boolean) is
   begin
      Code_Buffers.Set_Bounded_String (Target => Output, Source => "", Drop => Right);
      case Self.Config.Format is
         when JVM_Bytecode =>
            Code_Buffers.Append (Source => Output, New_Item => "; STUNIR JVM Bytecode" & ASCII.LF);
            Code_Buffers.Append (Source => Output, New_Item => ".class public " & Name_Strings.To_String (Module.Module_Name) & ASCII.LF);
            Code_Buffers.Append (Source => Output, New_Item => ".super java/lang/Object" & ASCII.LF);
         when DOTNET_IL =>
            Code_Buffers.Append (Source => Output, New_Item => "// STUNIR .NET IL" & ASCII.LF);
            Code_Buffers.Append (Source => Output, New_Item => ".class public " & Name_Strings.To_String (Module.Module_Name) & " {" & ASCII.LF);
            Code_Buffers.Append (Source => Output, New_Item => "}" & ASCII.LF);
         when Python_Bytecode =>
            Code_Buffers.Append (Source => Output, New_Item => "# STUNIR Python Bytecode" & ASCII.LF);
         when LLVM_IR =>
            Code_Buffers.Append (Source => Output, New_Item => "; STUNIR LLVM IR" & ASCII.LF);
            Code_Buffers.Append (Source => Output, New_Item => "define void @main() {" & ASCII.LF & "  ret void" & ASCII.LF & "}" & ASCII.LF);
         when WebAssembly_Bytecode =>
            Code_Buffers.Append (Source => Output, New_Item => ";; STUNIR WebAssembly" & ASCII.LF);
            Code_Buffers.Append (Source => Output, New_Item => "(module)" & ASCII.LF);
      end case;
      Success := True;
   end Emit_Module;

   overriding procedure Emit_Type (Self : in out Bytecode_Emitter; T : in IR_Type_Def; Output : out IR_Code_Buffer; Success: out Boolean) is
   begin
      Code_Buffers.Set_Bounded_String (Target => Output, Source => "", Drop => Right);
      Code_Buffers.Append (Source => Output, New_Item => "; Type: " & Name_Strings.To_String (T.Name) & ASCII.LF);
      Success := True;
   end Emit_Type;

   overriding procedure Emit_Function (Self : in out Bytecode_Emitter; Func : in IR_Function; Output : out IR_Code_Buffer; Success: out Boolean) is
   begin
      Code_Buffers.Set_Bounded_String (Target => Output, Source => "", Drop => Right);
      case Self.Config.Format is
         when JVM_Bytecode =>
            Code_Buffers.Append (Source => Output, New_Item => ".method public static " & Name_Strings.To_String (Func.Name) & "()V" & ASCII.LF);
            Code_Buffers.Append (Source => Output, New_Item => "  return" & ASCII.LF);
            Code_Buffers.Append (Source => Output, New_Item => ".end method" & ASCII.LF);
         when DOTNET_IL =>
            Code_Buffers.Append (Source => Output, New_Item => ".method public static void " & Name_Strings.To_String (Func.Name) & "() {" & ASCII.LF);
            Code_Buffers.Append (Source => Output, New_Item => "  ret" & ASCII.LF & "}" & ASCII.LF);
         when others =>
            Code_Buffers.Append (Source => Output, New_Item => "; Function: " & Name_Strings.To_String (Func.Name) & ASCII.LF);
      end case;
      Success := True;
   end Emit_Function;

end STUNIR.Emitters.Bytecode;
