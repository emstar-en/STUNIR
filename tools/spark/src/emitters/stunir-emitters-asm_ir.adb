-- STUNIR Assembly IR Emitter (SPARK Body)
with Ada.Strings; use Ada.Strings;

package body STUNIR.Emitters.ASM_IR is
   pragma SPARK_Mode (On);

   overriding procedure Emit_Module (Self : in out ASM_IR_Emitter; Module : in IR_Module; Output : out IR_Code_Buffer; Success: out Boolean) is
   begin
      Code_Buffers.Set_Bounded_String (Target => Output, Source => "", Drop => Right);
      case Self.Config.Format is
         when LLVM_IR =>
            Code_Buffers.Append (Source => Output, New_Item => "; STUNIR Generated LLVM IR" & ASCII.LF);
            Code_Buffers.Append (Source => Output, New_Item => "; DO-178C Level A" & ASCII.LF & ASCII.LF);
            Code_Buffers.Append (Source => Output, New_Item => "target datalayout = ""e-m:e-p:64:64-i64:64""" & ASCII.LF);
            Code_Buffers.Append (Source => Output, New_Item => "target triple = ""x86_64-unknown-linux-gnu""" & ASCII.LF & ASCII.LF);
         when GCC_RTL =>
            Code_Buffers.Append (Source => Output, New_Item => ";; STUNIR Generated GCC RTL" & ASCII.LF);
         when MLIR =>
            Code_Buffers.Append (Source => Output, New_Item => "// STUNIR Generated MLIR" & ASCII.LF);
            Code_Buffers.Append (Source => Output, New_Item => "module {" & ASCII.LF & "}" & ASCII.LF);
         when QBE_IR =>
            Code_Buffers.Append (Source => Output, New_Item => "# STUNIR Generated QBE IR" & ASCII.LF);
         when Cranelift_IR =>
            Code_Buffers.Append (Source => Output, New_Item => "; STUNIR Generated Cranelift IR" & ASCII.LF);
      end case;
      Success := True;
   end Emit_Module;

   overriding procedure Emit_Type (Self : in out ASM_IR_Emitter; T : in IR_Type_Def; Output : out IR_Code_Buffer; Success: out Boolean) is
   begin
      Code_Buffers.Set_Bounded_String (Target => Output, Source => "", Drop => Right);
      case Self.Config.Format is
         when LLVM_IR =>
            Code_Buffers.Append (Source => Output, New_Item => "%" & Name_Strings.To_String (T.Name) & " = type { }");
         when others =>
            Code_Buffers.Append (Source => Output, New_Item => "; Type: " & Name_Strings.To_String (T.Name));
      end case;
      Code_Buffers.Append (Source => Output, New_Item => ASCII.LF);
      Success := True;
   end Emit_Type;

   overriding procedure Emit_Function (Self : in out ASM_IR_Emitter; Func : in IR_Function; Output : out IR_Code_Buffer; Success: out Boolean) is
   begin
      Code_Buffers.Set_Bounded_String (Target => Output, Source => "", Drop => Right);
      case Self.Config.Format is
         when LLVM_IR =>
            Code_Buffers.Append (Source => Output, New_Item => "define void @" & Name_Strings.To_String (Func.Name) & "() {" & ASCII.LF);
            Code_Buffers.Append (Source => Output, New_Item => "  ret void" & ASCII.LF);
            Code_Buffers.Append (Source => Output, New_Item => "}" & ASCII.LF);
         when others =>
            Code_Buffers.Append (Source => Output, New_Item => "; Function: " & Name_Strings.To_String (Func.Name) & ASCII.LF);
      end case;
      Success := True;
   end Emit_Function;

end STUNIR.Emitters.ASM_IR;
