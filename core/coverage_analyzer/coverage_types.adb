--  STUNIR Coverage Analyzer Types - Implementation
--  SPARK Migration Phase 3

pragma SPARK_Mode (On);

package body Coverage_Types is

   --  ===========================================
   --  Initialize Module Array
   --  ===========================================

   function Init_Module_Array return Module_Array is
      Result : Module_Array;
   begin
      for I in Valid_Module_Index loop
         Result (I) := Empty_Module_Coverage;
      end loop;
      return Result;
   end Init_Module_Array;

   --  ===========================================
   --  Empty Report
   --  ===========================================

   function Empty_Report return Coverage_Report is
   begin
      return (
         Modules       => Init_Module_Array,
         Module_Count  => 0,
         Total_Metrics => Empty_Metrics,
         Line_Pct      => 0,
         Branch_Pct    => 0,
         Function_Pct  => 0,
         Is_Valid      => False,
         Meets_Minimum => False
      );
   end Empty_Report;

   --  ===========================================
   --  Get Line Coverage
   --  ===========================================

   function Get_Line_Coverage (M : Coverage_Metrics) return Percentage is
   begin
      if M.Total_Lines = 0 then
         return 100;  --  No lines = full coverage
      elsif M.Covered_Lines >= M.Total_Lines then
         return 100;
      else
         return Percentage ((M.Covered_Lines * 100) / M.Total_Lines);
      end if;
   end Get_Line_Coverage;

   --  ===========================================
   --  Get Branch Coverage
   --  ===========================================

   function Get_Branch_Coverage (M : Coverage_Metrics) return Percentage is
   begin
      if M.Total_Branches = 0 then
         return 100;  --  No branches = full coverage
      elsif M.Covered_Branches >= M.Total_Branches then
         return 100;
      else
         return Percentage ((M.Covered_Branches * 100) / M.Total_Branches);
      end if;
   end Get_Branch_Coverage;

   --  ===========================================
   --  Get Function Coverage
   --  ===========================================

   function Get_Function_Coverage (M : Coverage_Metrics) return Percentage is
   begin
      if M.Total_Functions = 0 then
         return 100;  --  No functions = full coverage
      elsif M.Covered_Functions >= M.Total_Functions then
         return 100;
      else
         return Percentage ((M.Covered_Functions * 100) / M.Total_Functions);
      end if;
   end Get_Function_Coverage;

   --  ===========================================
   --  Classify Coverage
   --  ===========================================

   function Classify_Coverage (Pct : Percentage) return Coverage_Level is
   begin
      if Pct = 100 then
         return Full;
      elsif Pct >= 90 then
         return High;
      elsif Pct >= 70 then
         return Medium;
      elsif Pct >= 50 then
         return Low;
      elsif Pct > 0 then
         return Minimal;
      else
         return None;
      end if;
   end Classify_Coverage;

   --  ===========================================
   --  Meets Minimum Coverage
   --  ===========================================

   function Meets_Minimum_Coverage (M : Coverage_Metrics) return Boolean is
      Line_Pct   : constant Percentage := Get_Line_Coverage (M);
      Branch_Pct : constant Percentage := Get_Branch_Coverage (M);
      Func_Pct   : constant Percentage := Get_Function_Coverage (M);
   begin
      return Line_Pct >= Minimum_Line_Coverage and
             Branch_Pct >= Minimum_Branch_Coverage and
             Func_Pct >= Minimum_Function_Coverage;
   end Meets_Minimum_Coverage;

end Coverage_Types;
