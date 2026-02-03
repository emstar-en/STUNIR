-- STUNIR Scientific Computing Emitter (SPARK Body)
with Ada.Strings; use Ada.Strings;

package body STUNIR.Emitters.Scientific is
   pragma SPARK_Mode (On);

   overriding procedure Emit_Module (Self : in out Scientific_Emitter; Module : in IR_Module; Output : out IR_Code_Buffer; Success: out Boolean) is
   begin
      Code_Buffers.Set_Bounded_String (Target => Output, Source => "", Drop => Right);
      case Self.Config.Language is
         when MATLAB =>
            Code_Buffers.Append (Source => Output, New_Item => "% STUNIR Generated MATLAB" & ASCII.LF & "function main()" & ASCII.LF & "end" & ASCII.LF);
         when NumPy =>
            Code_Buffers.Append (Source => Output, New_Item => "# STUNIR Generated NumPy" & ASCII.LF & "import numpy as np" & ASCII.LF & ASCII.LF);
         when Julia =>
            Code_Buffers.Append (Source => Output, New_Item => "# STUNIR Generated Julia" & ASCII.LF & "module " & Name_Strings.To_String (Module.Module_Name) & ASCII.LF & "end" & ASCII.LF);
         when R_Lang =>
            Code_Buffers.Append (Source => Output, New_Item => "# STUNIR Generated R" & ASCII.LF & "library(stats)" & ASCII.LF);
         when Fortran_90 | Fortran_95 =>
            Code_Buffers.Append (Source => Output, New_Item => "! STUNIR Generated Fortran" & ASCII.LF & "program main" & ASCII.LF & "end program" & ASCII.LF);
      end case;
      Success := True;
   end Emit_Module;

   overriding procedure Emit_Type (Self : in out Scientific_Emitter; T : in IR_Type_Def; Output : out IR_Code_Buffer; Success: out Boolean) is
   begin
      Code_Buffers.Set_Bounded_String (Target => Output, Source => "", Drop => Right);
      Code_Buffers.Append (Source => Output, New_Item => "% Type: " & Name_Strings.To_String (T.Name) & ASCII.LF);
      Success := True;
   end Emit_Type;

   overriding procedure Emit_Function (Self : in out Scientific_Emitter; Func : in IR_Function; Output : out IR_Code_Buffer; Success: out Boolean) is
   begin
      Code_Buffers.Set_Bounded_String (Target => Output, Source => "", Drop => Right);
      Code_Buffers.Append (Source => Output, New_Item => "function result = " & Name_Strings.To_String (Func.Name) & "()" & ASCII.LF);
      Code_Buffers.Append (Source => Output, New_Item => "end" & ASCII.LF);
      Success := True;
   end Emit_Function;

end STUNIR.Emitters.Scientific;
