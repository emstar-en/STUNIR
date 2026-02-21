--  STUNIR Test Data Generator Types - Implementation
--  SPARK Migration Phase 3

pragma SPARK_Mode (On);

package body Test_Data_Types is

   --  ===========================================
   --  Initialize Vector Array
   --  ===========================================

   function Init_Vector_Array return Vector_Array is
      Result : Vector_Array;
   begin
      for I in Valid_Vector_Index loop
         Result (I) := Empty_Vector;
      end loop;
      return Result;
   end Init_Vector_Array;

   --  ===========================================
   --  Empty Vector Set
   --  ===========================================

   function Empty_Vector_Set return Vector_Set is
   begin
      return (
         Vectors => Init_Vector_Array,
         Count   => 0,
         Stats   => Empty_Vector_Stats
      );
   end Empty_Vector_Set;

   --  ===========================================
   --  Get Category Count
   --  ===========================================

   function Get_Category_Count (
      Stats : Vector_Stats;
      Cat   : Vector_Category) return Natural is
   begin
      case Cat is
         when Conformance => return Stats.Conformance;
         when Boundary    => return Stats.Boundary;
         when Error_Cat   => return Stats.Error_Cnt;
         when Performance => return Stats.Performance;
         when Regression  => return Stats.Regression;
         when Random_Cat  => return Stats.Random_Cnt;
      end case;
   end Get_Category_Count;

end Test_Data_Types;
