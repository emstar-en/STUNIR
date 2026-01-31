-- STUNIR Semantic IR Data Model (SPARK Implementation Body)
-- DO-178C Level A
-- Phase 3a: Core Category Emitters

package body STUNIR.Semantic_IR is
   pragma SPARK_Mode (On);

   -- Get the name of a type definition
   function Get_Type_Name (T : IR_Type_Def) return String is
   begin
      return Name_Strings.To_String (T.Name);
   end Get_Type_Name;

   -- Get the name of a function
   function Get_Function_Name (Func : IR_Function) return String is
   begin
      return Name_Strings.To_String (Func.Name);
   end Get_Function_Name;

end STUNIR.Semantic_IR;
