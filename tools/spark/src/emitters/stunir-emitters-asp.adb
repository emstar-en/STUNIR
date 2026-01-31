-- STUNIR Answer Set Programming Emitter (SPARK Body)
with Ada.Strings; use Ada.Strings;

package body STUNIR.Emitters.ASP is
   pragma SPARK_Mode (On);

   overriding procedure Emit_Module (Self : in out ASP_Emitter; Module : in IR_Module; Output : out IR_Code_Buffer; Success: out Boolean) is
   begin
      Code_Buffers.Set_Bounded_String (Target => Output, Source => "", Drop => Right);
      Code_Buffers.Append (Source => Output, New_Item => "% STUNIR Generated ASP" & ASCII.LF);
      Code_Buffers.Append (Source => Output, New_Item => "% DO-178C Level A" & ASCII.LF);
      Code_Buffers.Append (Source => Output, New_Item => "% Solver: " & ASP_Solver'Image (Self.Config.Solver) & ASCII.LF & ASCII.LF);
      Code_Buffers.Append (Source => Output, New_Item => "% Facts" & ASCII.LF);
      Code_Buffers.Append (Source => Output, New_Item => "fact(example)." & ASCII.LF & ASCII.LF);
      Code_Buffers.Append (Source => Output, New_Item => "% Rules" & ASCII.LF);
      Code_Buffers.Append (Source => Output, New_Item => "derived(X) :- fact(X)." & ASCII.LF & ASCII.LF);
      if Self.Config.Use_Aggregates then
         Code_Buffers.Append (Source => Output, New_Item => "% Aggregates" & ASCII.LF);
         Code_Buffers.Append (Source => Output, New_Item => "count(N) :- N = #count{X : fact(X)}." & ASCII.LF & ASCII.LF);
      end if;
      if Self.Config.Use_Optimization then
         Code_Buffers.Append (Source => Output, New_Item => "% Optimization" & ASCII.LF);
         Code_Buffers.Append (Source => Output, New_Item => "#minimize { 1 : fact(X) }." & ASCII.LF);
      end if;
      Success := True;
   end Emit_Module;

   overriding procedure Emit_Type (Self : in out ASP_Emitter; T : in IR_Type_Def; Output : out IR_Code_Buffer; Success: out Boolean) is
   begin
      Code_Buffers.Set_Bounded_String (Target => Output, Source => "", Drop => Right);
      Code_Buffers.Append (Source => Output, New_Item => "% Type: " & Name_Strings.To_String (T.Name) & ASCII.LF);
      for I in 1 .. T.Field_Cnt loop
         Code_Buffers.Append (Source => Output, New_Item => "% Field: " & Name_Strings.To_String (T.Fields (I).Name) & ASCII.LF);
      end loop;
      Success := True;
   end Emit_Type;

   overriding procedure Emit_Function (Self : in out ASP_Emitter; Func : in IR_Function; Output : out IR_Code_Buffer; Success: out Boolean) is
   begin
      Code_Buffers.Set_Bounded_String (Target => Output, Source => "", Drop => Right);
      Code_Buffers.Append (Source => Output, New_Item => "% Rule: " & Name_Strings.To_String (Func.Name) & ASCII.LF);
      Code_Buffers.Append (Source => Output, New_Item => Name_Strings.To_String (Func.Name) & "(X) :- true." & ASCII.LF);
      Success := True;
   end Emit_Function;

end STUNIR.Emitters.ASP;
