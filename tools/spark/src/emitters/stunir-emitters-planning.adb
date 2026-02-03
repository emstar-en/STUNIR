-- STUNIR Planning Language Emitter (SPARK Body)
with Ada.Strings; use Ada.Strings;

package body STUNIR.Emitters.Planning is
   pragma SPARK_Mode (On);

   overriding procedure Emit_Module (Self : in out Planning_Emitter; Module : in IR_Module; Output : out IR_Code_Buffer; Success: out Boolean) is
   begin
      Code_Buffers.Set_Bounded_String (Target => Output, Source => "", Drop => Right);
      Code_Buffers.Append (Source => Output, New_Item => ";; STUNIR Generated PDDL Domain" & ASCII.LF);
      Code_Buffers.Append (Source => Output, New_Item => ";; DO-178C Level A" & ASCII.LF & ASCII.LF);
      Code_Buffers.Append (Source => Output, New_Item => "(define (domain " & Name_Strings.To_String (Module.Module_Name) & ")" & ASCII.LF);
      Code_Buffers.Append (Source => Output, New_Item => "  (:requirements :strips" & ASCII.LF);
      if Self.Config.Use_Temporal then
         Code_Buffers.Append (Source => Output, New_Item => "    :durative-actions" & ASCII.LF);
      end if;
      if Self.Config.Use_Numeric then
         Code_Buffers.Append (Source => Output, New_Item => "    :fluents" & ASCII.LF);
      end if;
      Code_Buffers.Append (Source => Output, New_Item => "  )" & ASCII.LF);
      Code_Buffers.Append (Source => Output, New_Item => ")" & ASCII.LF);
      Success := True;
   end Emit_Module;

   overriding procedure Emit_Type (Self : in out Planning_Emitter; T : in IR_Type_Def; Output : out IR_Code_Buffer; Success: out Boolean) is
   begin
      Code_Buffers.Set_Bounded_String (Target => Output, Source => "", Drop => Right);
      Code_Buffers.Append (Source => Output, New_Item => "  (:types " & Name_Strings.To_String (T.Name) & ")" & ASCII.LF);
      Success := True;
   end Emit_Type;

   overriding procedure Emit_Function (Self : in out Planning_Emitter; Func : in IR_Function; Output : out IR_Code_Buffer; Success: out Boolean) is
   begin
      Code_Buffers.Set_Bounded_String (Target => Output, Source => "", Drop => Right);
      Code_Buffers.Append (Source => Output, New_Item => "  (:action " & Name_Strings.To_String (Func.Name) & ASCII.LF);
      Code_Buffers.Append (Source => Output, New_Item => "    :parameters ()" & ASCII.LF);
      Code_Buffers.Append (Source => Output, New_Item => "    :precondition (and )" & ASCII.LF);
      Code_Buffers.Append (Source => Output, New_Item => "    :effect (and )" & ASCII.LF);
      Code_Buffers.Append (Source => Output, New_Item => "  )" & ASCII.LF);
      Success := True;
   end Emit_Function;

end STUNIR.Emitters.Planning;
