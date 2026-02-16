-- STUNIR Systems Programming Emitter (SPARK Body)
with Ada.Strings; use Ada.Strings;

package body STUNIR.Emitters.Systems is
   pragma SPARK_Mode (On);

   overriding procedure Emit_Module (Self : in out Systems_Emitter; Module : in IR_Module; Output : out IR_Code_Buffer; Success: out Boolean) is
   begin
      Code_Buffers.Set_Bounded_String (Target => Output, Source => "", Drop => Right);
      case Self.Config.Language is
         when Ada_2012 | Ada_2022 =>
            Code_Buffers.Append (Source => Output, New_Item => "-- STUNIR Generated Ada" & ASCII.LF);
            Code_Buffers.Append (Source => Output, New_Item => "-- DO-178C Level A" & ASCII.LF & ASCII.LF);
            Code_Buffers.Append (Source => Output, New_Item => "package " & Name_Strings.To_String (Module.Module_Name) & " is" & ASCII.LF);
            if Self.Config.Use_SPARK then
               Code_Buffers.Append (Source => Output, New_Item => "   pragma SPARK_Mode (On);" & ASCII.LF);
            end if;
            Code_Buffers.Append (Source => Output, New_Item => "end " & Name_Strings.To_String (Module.Module_Name) & ";" & ASCII.LF);
         when D_Lang =>
            Code_Buffers.Append (Source => Output, New_Item => "// STUNIR Generated D" & ASCII.LF);
            Code_Buffers.Append (Source => Output, New_Item => "module " & Name_Strings.To_String (Module.Module_Name) & ";" & ASCII.LF);
         when Nim =>
            Code_Buffers.Append (Source => Output, New_Item => "# STUNIR Generated Nim" & ASCII.LF);
            Code_Buffers.Append (Source => Output, New_Item => "# DO-178C Level A" & ASCII.LF & ASCII.LF);
         when Zig =>
            Code_Buffers.Append (Source => Output, New_Item => "// STUNIR Generated Zig" & ASCII.LF);
            Code_Buffers.Append (Source => Output, New_Item => "const std = @import(""std"");" & ASCII.LF & ASCII.LF);
         when Carbon =>
            Code_Buffers.Append (Source => Output, New_Item => "// STUNIR Generated Carbon" & ASCII.LF);
      end case;
      Success := True;
   end Emit_Module;

   overriding procedure Emit_Type (Self : in out Systems_Emitter; T : in IR_Type_Def; Output : out IR_Code_Buffer; Success: out Boolean) is
   begin
      Code_Buffers.Set_Bounded_String (Target => Output, Source => "", Drop => Right);
      case Self.Config.Language is
         when Ada_2012 | Ada_2022 =>
            Code_Buffers.Append (Source => Output, New_Item => "   type " & Name_Strings.To_String (T.Name) & " is record" & ASCII.LF);
            for I in 1 .. T.Field_Cnt loop
               Code_Buffers.Append (Source => Output, New_Item => "      " & Name_Strings.To_String (T.Fields (I).Name) & " : Integer;" & ASCII.LF);
            end loop;
            Code_Buffers.Append (Source => Output, New_Item => "   end record;" & ASCII.LF);
         when D_Lang =>
            Code_Buffers.Append (Source => Output, New_Item => "struct " & Name_Strings.To_String (T.Name) & " { }" & ASCII.LF);
         when Nim =>
            Code_Buffers.Append (Source => Output, New_Item => "type " & Name_Strings.To_String (T.Name) & " = object" & ASCII.LF);
         when Zig =>
            Code_Buffers.Append (Source => Output, New_Item => "const " & Name_Strings.To_String (T.Name) & " = struct {};" & ASCII.LF);
         when others =>
            Code_Buffers.Append (Source => Output, New_Item => "type " & Name_Strings.To_String (T.Name) & ";" & ASCII.LF);
      end case;
      Success := True;
   end Emit_Type;

   overriding procedure Emit_Function (Self : in out Systems_Emitter; Func : in IR_Function; Output : out IR_Code_Buffer; Success: out Boolean) is
   begin
      Code_Buffers.Set_Bounded_String (Target => Output, Source => "", Drop => Right);
      case Self.Config.Language is
         when Ada_2012 | Ada_2022 =>
            Code_Buffers.Append (Source => Output, New_Item => "   procedure " & Name_Strings.To_String (Func.Name));
            if Func.Arg_Cnt > 0 then
               Code_Buffers.Append (Source => Output, New_Item => " (...)");
            end if;
            Code_Buffers.Append (Source => Output, New_Item => ";" & ASCII.LF);
         when D_Lang =>
            Code_Buffers.Append (Source => Output, New_Item => "void " & Name_Strings.To_String (Func.Name) & "() { }" & ASCII.LF);
         when Nim =>
            Code_Buffers.Append (Source => Output, New_Item => "proc " & Name_Strings.To_String (Func.Name) & "() = discard" & ASCII.LF);
         when Zig =>
            Code_Buffers.Append (Source => Output, New_Item => "pub fn " & Name_Strings.To_String (Func.Name) & "() void {}" & ASCII.LF);
         when others =>
            Code_Buffers.Append (Source => Output, New_Item => "fn " & Name_Strings.To_String (Func.Name) & "() {}" & ASCII.LF);
      end case;
      Success := True;
   end Emit_Function;

end STUNIR.Emitters.Systems;
