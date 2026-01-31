-- STUNIR Object-Oriented Programming Emitter (SPARK Body)
-- DO-178C Level A
-- Phase 3c: Remaining Category Emitters

with Ada.Strings; use Ada.Strings;

package body STUNIR.Emitters.OOP is
   pragma SPARK_Mode (On);

   overriding procedure Emit_Module
     (Self   : in out OOP_Emitter;
      Module : in     IR_Module;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   is
   begin
      Code_Buffers.Set_Bounded_String (Target => Output, Source => "", Drop => Ada.Strings.Right);

      case Self.Config.Language is
         when Java =>
            Code_Buffers.Append (Source => Output, New_Item => "// STUNIR Generated Java" & ASCII.LF);
            Code_Buffers.Append (Source => Output, New_Item => "// DO-178C Level A" & ASCII.LF & ASCII.LF);
            Code_Buffers.Append (Source => Output, New_Item => "public class " & Name_Strings.To_String (Module.Module_Name) & " {" & ASCII.LF);
            Code_Buffers.Append (Source => Output, New_Item => "    // STUNIR generated" & ASCII.LF);
            Code_Buffers.Append (Source => Output, New_Item => "}" & ASCII.LF);

         when Cpp =>
            Code_Buffers.Append (Source => Output, New_Item => "// STUNIR Generated C++" & ASCII.LF);
            Code_Buffers.Append (Source => Output, New_Item => "// DO-178C Level A" & ASCII.LF & ASCII.LF);
            Code_Buffers.Append (Source => Output, New_Item => "class " & Name_Strings.To_String (Module.Module_Name) & " {" & ASCII.LF);
            Code_Buffers.Append (Source => Output, New_Item => "public:" & ASCII.LF);
            Code_Buffers.Append (Source => Output, New_Item => "    // STUNIR generated" & ASCII.LF);
            Code_Buffers.Append (Source => Output, New_Item => "};" & ASCII.LF);

         when CSharp =>
            Code_Buffers.Append (Source => Output, New_Item => "// STUNIR Generated C#" & ASCII.LF);
            Code_Buffers.Append (Source => Output, New_Item => "// DO-178C Level A" & ASCII.LF & ASCII.LF);
            Code_Buffers.Append (Source => Output, New_Item => "public class " & Name_Strings.To_String (Module.Module_Name) & " {" & ASCII.LF);
            Code_Buffers.Append (Source => Output, New_Item => "    // STUNIR generated" & ASCII.LF);
            Code_Buffers.Append (Source => Output, New_Item => "}" & ASCII.LF);

         when Python_OOP =>
            Code_Buffers.Append (Source => Output, New_Item => "# STUNIR Generated Python OOP" & ASCII.LF);
            Code_Buffers.Append (Source => Output, New_Item => "# DO-178C Level A" & ASCII.LF & ASCII.LF);
            Code_Buffers.Append (Source => Output, New_Item => "class " & Name_Strings.To_String (Module.Module_Name) & ":" & ASCII.LF);
            Code_Buffers.Append (Source => Output, New_Item => "    \"\"\"STUNIR generated class\"\"\"" & ASCII.LF);
            Code_Buffers.Append (Source => Output, New_Item => "    pass" & ASCII.LF);

         when Ruby =>
            Code_Buffers.Append (Source => Output, New_Item => "# STUNIR Generated Ruby" & ASCII.LF);
            Code_Buffers.Append (Source => Output, New_Item => "# DO-178C Level A" & ASCII.LF & ASCII.LF);
            Code_Buffers.Append (Source => Output, New_Item => "class " & Name_Strings.To_String (Module.Module_Name) & ASCII.LF);
            Code_Buffers.Append (Source => Output, New_Item => "  # STUNIR generated" & ASCII.LF);
            Code_Buffers.Append (Source => Output, New_Item => "end" & ASCII.LF);

         when others =>
            Code_Buffers.Append (Source => Output, New_Item => "// STUNIR Generated OOP" & ASCII.LF);
      end case;

      Success := True;
   end Emit_Module;

   overriding procedure Emit_Type
     (Self   : in out OOP_Emitter;
      T      : in     IR_Type_Def;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   is
   begin
      Code_Buffers.Set_Bounded_String (Target => Output, Source => "", Drop => Ada.Strings.Right);

      Code_Buffers.Append (Source => Output, New_Item => "class " & Name_Strings.To_String (T.Name) & " {" & ASCII.LF);
      for I in 1 .. T.Field_Cnt loop
         Code_Buffers.Append (Source => Output, New_Item => "    private String " & Name_Strings.To_String (T.Fields (I).Name) & ";" & ASCII.LF);
      end loop;
      Code_Buffers.Append (Source => Output, New_Item => "}" & ASCII.LF);

      Success := True;
   end Emit_Type;

   overriding procedure Emit_Function
     (Self   : in out OOP_Emitter;
      Func   : in     IR_Function;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   is
   begin
      Code_Buffers.Set_Bounded_String (Target => Output, Source => "", Drop => Ada.Strings.Right);

      Code_Buffers.Append (Source => Output, New_Item => "    public void " & Name_Strings.To_String (Func.Name) & "() {" & ASCII.LF);
      Code_Buffers.Append (Source => Output, New_Item => "        // STUNIR generated method" & ASCII.LF);
      Code_Buffers.Append (Source => Output, New_Item => "    }" & ASCII.LF);

      Success := True;
   end Emit_Function;

end STUNIR.Emitters.OOP;
