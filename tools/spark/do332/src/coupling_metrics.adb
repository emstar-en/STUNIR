--  STUNIR DO-332 Coupling Metrics Implementation
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

package body Coupling_Metrics is

   --  ============================================================
   --  Calculate_CBO Implementation
   --  ============================================================

   function Calculate_CBO (
      Class_ID     : OOP_Types.Class_ID;
      Dependencies : Dependency_Array
   ) return Natural is
      Coupled_Classes : Class_ID_Array (1 .. 1000);
      Count           : Natural := 0;
      Found           : Boolean;
   begin
      --  Count unique classes this class depends on or is depended on by
      for I in Dependencies'Range loop
         if Dependencies (I).Source_Class = Class_ID then
            --  Outgoing dependency
            Found := False;
            for J in 1 .. Count loop
               if Coupled_Classes (J) = Dependencies (I).Target_Class then
                  Found := True;
                  exit;
               end if;
            end loop;
            if not Found and Count < 1000 then
               Count := Count + 1;
               Coupled_Classes (Count) := Dependencies (I).Target_Class;
            end if;
         elsif Dependencies (I).Target_Class = Class_ID then
            --  Incoming dependency
            Found := False;
            for J in 1 .. Count loop
               if Coupled_Classes (J) = Dependencies (I).Source_Class then
                  Found := True;
                  exit;
               end if;
            end loop;
            if not Found and Count < 1000 then
               Count := Count + 1;
               Coupled_Classes (Count) := Dependencies (I).Source_Class;
            end if;
         end if;
      end loop;
      return Count;
   end Calculate_CBO;

   --  ============================================================
   --  Calculate_RFC Implementation
   --  ============================================================

   function Calculate_RFC (
      Class_ID : OOP_Types.Class_ID;
      Methods  : Method_Array
   ) return Natural is
      Own_Methods     : Natural := 0;
      Called_Methods  : Natural := 0;  --  Would need call graph analysis
   begin
      --  Count own methods
      for I in Methods'Range loop
         if Methods (I).Owning_Class = Class_ID then
            Own_Methods := Own_Methods + 1;
         end if;
      end loop;

      --  RFC = own methods + methods called (simplified: just own methods)
      return Own_Methods + Called_Methods;
   end Calculate_RFC;

   --  ============================================================
   --  Calculate_LCOM Implementation
   --  ============================================================

   function Calculate_LCOM (
      Class_ID : OOP_Types.Class_ID;
      Methods  : Method_Array
   ) return Natural is
      Method_Count : Natural := 0;
   begin
      --  LCOM calculation requires attribute usage analysis
      --  Simplified: return 0 (perfectly cohesive) if few methods,
      --  else estimate based on method count

      for I in Methods'Range loop
         if Methods (I).Owning_Class = Class_ID then
            Method_Count := Method_Count + 1;
         end if;
      end loop;

      --  Simple heuristic: LCOM increases with method count
      --  A real implementation would analyze field usage
      if Method_Count <= 3 then
         return 0;
      else
         return (Method_Count * (Method_Count - 1)) / 2;  --  Potential pairs
      end if;
   end Calculate_LCOM;

   --  ============================================================
   --  Calculate_NOC Implementation
   --  ============================================================

   function Calculate_NOC (
      Class_ID : OOP_Types.Class_ID;
      Links    : Inheritance_Array
   ) return Natural is
      Count : Natural := 0;
   begin
      for I in Links'Range loop
         if Links (I).Parent_ID = Class_ID then
            Count := Count + 1;
         end if;
      end loop;
      return Count;
   end Calculate_NOC;

   --  ============================================================
   --  Calculate_WMC Implementation
   --  ============================================================

   function Calculate_WMC (
      Class_ID : OOP_Types.Class_ID;
      Methods  : Method_Array
   ) return Natural is
      Count : Natural := 0;
   begin
      --  Simplified: WMC = method count (assuming complexity 1 per method)
      for I in Methods'Range loop
         if Methods (I).Owning_Class = Class_ID then
            Count := Count + 1;
         end if;
      end loop;
      return Count;
   end Calculate_WMC;

   --  ============================================================
   --  Check_Thresholds Implementation
   --  ============================================================

   function Check_Thresholds (
      CBO        : Natural;
      RFC        : Natural;
      LCOM       : Natural;
      DIT        : Natural;
      NOC        : Natural;
      WMC        : Natural;
      Thresholds : Threshold_Set
   ) return Threshold_Violations is
   begin
      return (
         CBO_Exceeded  => CBO > Thresholds.CBO_Threshold,
         RFC_Exceeded  => RFC > Thresholds.RFC_Threshold,
         LCOM_Exceeded => LCOM > Thresholds.LCOM_Threshold,
         DIT_Exceeded  => DIT > Thresholds.DIT_Threshold,
         NOC_Exceeded  => NOC > Thresholds.NOC_Threshold,
         WMC_Exceeded  => WMC > Thresholds.WMC_Threshold
      );
   end Check_Thresholds;

end Coupling_Metrics;
