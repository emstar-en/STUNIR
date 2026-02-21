--  STUNIR DO-332 Inheritance Metrics Implementation
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

package body Inheritance_Metrics is

   --  ============================================================
   --  Calculate_Metrics Implementation
   --  ============================================================

   function Calculate_Metrics (
      Classes : Class_Array;
      Methods : Method_Array;
      Links   : Inheritance_Array;
      Results : Inheritance_Result_Array
   ) return Inheritance_Metrics_Record is
      M           : Inheritance_Metrics_Record := Null_Metrics;
      Total_Depth : Natural := 0;
   begin
      M.Total_Classes := Metric_Value (Classes'Length);

      --  Process each result
      for I in Results'Range loop
         if I in Classes'Range then
            --  Track max depth
            if Results (I).Depth > Natural (M.Max_Depth) then
               M.Max_Depth := Metric_Value (Results (I).Depth);
            end if;
            Total_Depth := Total_Depth + Results (I).Depth;

            --  Count diamond patterns
            if Results (I).Has_Diamond then
               M.Diamond_Patterns := M.Diamond_Patterns + 1;
            end if;

            --  Count circular errors
            if Results (I).Has_Circular then
               M.Circular_Errors := M.Circular_Errors + 1;
            end if;

            --  Count multiple inheritance
            if Results (I).Multiple_Count > 1 then
               M.Classes_With_Multiple := M.Classes_With_Multiple + 1;
            end if;

            --  Count overrides
            M.Total_Overrides := M.Total_Overrides + Metric_Value (Results (I).Override_Count);
            if not Results (I).All_Overrides_Valid then
               M.Invalid_Overrides := M.Invalid_Overrides + 1;
            end if;

            --  Count abstract and virtual
            M.Total_Virtual_Methods := M.Total_Virtual_Methods + 
                                       Metric_Value (Results (I).Virtual_Count);
            if Results (I).Abstract_Count > 0 then
               M.Abstract_Classes := M.Abstract_Classes + 1;
            end if;
         end if;
      end loop;

      --  Calculate average depth (x100 for precision without floats)
      if Classes'Length > 0 then
         M.Avg_Depth_X100 := Metric_Value ((Total_Depth * 100) / Classes'Length);
      end if;

      --  Count interfaces
      for I in Classes'Range loop
         if Classes (I).Kind = Interface_Type then
            M.Interface_Count := M.Interface_Count + 1;
         end if;
      end loop;

      return M;
   end Calculate_Metrics;

   --  ============================================================
   --  Calculate_Depth_Metrics Implementation
   --  ============================================================

   function Calculate_Depth_Metrics (
      Results : Inheritance_Result_Array
   ) return Inheritance_Metrics_Record is
      M           : Inheritance_Metrics_Record := Null_Metrics;
      Total_Depth : Natural := 0;
   begin
      M.Total_Classes := Metric_Value (Results'Length);

      for I in Results'Range loop
         if Results (I).Depth > Natural (M.Max_Depth) then
            M.Max_Depth := Metric_Value (Results (I).Depth);
         end if;
         Total_Depth := Total_Depth + Results (I).Depth;
      end loop;

      if Results'Length > 0 then
         M.Avg_Depth_X100 := Metric_Value ((Total_Depth * 100) / Results'Length);
      end if;

      return M;
   end Calculate_Depth_Metrics;

   --  ============================================================
   --  Calculate_Override_Metrics Implementation
   --  ============================================================

   function Calculate_Override_Metrics (
      Classes : Class_Array;
      Methods : Method_Array;
      Links   : Inheritance_Array
   ) return Inheritance_Metrics_Record is
      M : Inheritance_Metrics_Record := Null_Metrics;
   begin
      --  Count total overrides
      for I in Methods'Range loop
         if Methods (I).Has_Override then
            M.Total_Overrides := M.Total_Overrides + 1;
         end if;
      end loop;

      --  Count virtual methods
      for I in Methods'Range loop
         if Is_Virtual (Methods (I).Kind) then
            M.Total_Virtual_Methods := M.Total_Virtual_Methods + 1;
         end if;
      end loop;

      return M;
   end Calculate_Override_Metrics;

   --  ============================================================
   --  Metrics_Pass_Thresholds Implementation
   --  ============================================================

   function Metrics_Pass_Thresholds (
      Metrics      : Inheritance_Metrics_Record;
      Max_Depth    : Natural;
      Max_Multiple : Natural
   ) return Boolean is
   begin
      --  Check depth threshold
      if Natural (Metrics.Max_Depth) > Max_Depth then
         return False;
      end if;

      --  Check multiple inheritance threshold
      if Natural (Metrics.Classes_With_Multiple) > Max_Multiple then
         return False;
      end if;

      --  No circular inheritance allowed
      if Metrics.Circular_Errors > 0 then
         return False;
      end if;

      --  No invalid overrides allowed
      if Metrics.Invalid_Overrides > 0 then
         return False;
      end if;

      return True;
   end Metrics_Pass_Thresholds;

end Inheritance_Metrics;
