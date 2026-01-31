-- STUNIR Visitor Pattern Implementation
-- DO-178C Level A
-- Phase 3a: Core Category Emitters

package body STUNIR.Emitters.Visitor is
   pragma SPARK_Mode (On);

   procedure Traverse_Module
     (Module  : in     IR_Module;
      Context : in out Visitor_Context'Class;
      Result  :    out Visit_Result)
   is
   begin
      Result := Continue;
      Context.Result := Continue;

      -- Start module visit
      On_Module_Start (Context, Module);
      if Context.Result = Abort_Visit then
         Result := Abort_Visit;
         return;
      end if;

      -- Visit all types
      for I in 1 .. Module.Type_Cnt loop
         pragma Loop_Invariant (I <= Module.Type_Cnt);
         pragma Loop_Invariant (Context.Result /= Abort_Visit);

         On_Type_Start (Context, Module.Types (I));
         if Context.Result = Abort_Visit then
            Result := Abort_Visit;
            return;
         end if;

         On_Type_End (Context, Module.Types (I));
         if Context.Result = Abort_Visit then
            Result := Abort_Visit;
            return;
         end if;
      end loop;

      -- Visit all functions
      for I in 1 .. Module.Func_Cnt loop
         pragma Loop_Invariant (I <= Module.Func_Cnt);
         pragma Loop_Invariant (Context.Result /= Abort_Visit);

         On_Function_Start (Context, Module.Functions (I));
         if Context.Result = Abort_Visit then
            Result := Abort_Visit;
            return;
         end if;

         On_Function_End (Context, Module.Functions (I));
         if Context.Result = Abort_Visit then
            Result := Abort_Visit;
            return;
         end if;
      end loop;

      -- End module visit
      On_Module_End (Context, Module);
      if Context.Result = Abort_Visit then
         Result := Abort_Visit;
         return;
      end if;

      Result := Continue;
   end Traverse_Module;

end STUNIR.Emitters.Visitor;
