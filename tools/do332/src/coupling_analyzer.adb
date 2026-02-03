--  STUNIR DO-332 Coupling Analyzer Implementation
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

with Coupling_Metrics; use Coupling_Metrics;

package body Coupling_Analyzer is

   --  ============================================================
   --  Analyze_Class_Dependencies Implementation
   --  ============================================================

   function Analyze_Class_Dependencies (
      Class   : Class_Info;
      Classes : Class_Array;
      Methods : Method_Array;
      Links   : Inheritance_Array
   ) return Dependency_Array is
      Temp  : Dependency_Array (1 .. Max_Dependencies);
      Count : Natural := 0;
   begin
      --  Add inheritance dependencies
      for I in Links'Range loop
         if Links (I).Child_ID = Class.ID and Count < Max_Dependencies then
            Count := Count + 1;
            Temp (Count) := (
               Source_Class => Class.ID,
               Target_Class => Links (I).Parent_ID,
               Kind         => Inheritance_Dep,
               Count        => 1
            );
         end if;
      end loop;

      --  Add call dependencies from methods
      for I in Methods'Range loop
         if Methods (I).Owning_Class = Class.ID then
            --  Check if method calls other classes (simplified)
            --  In reality, we'd analyze the method body
            null;  --  Placeholder for detailed call analysis
         end if;
      end loop;

      if Count = 0 then
         declare
            Empty : Dependency_Array (1 .. 0);
         begin
            return Empty;
         end;
      end if;

      return Temp (1 .. Count);
   end Analyze_Class_Dependencies;

   --  ============================================================
   --  Build_Dependency_Graph Implementation
   --  ============================================================

   procedure Build_Dependency_Graph (
      Classes      : in     Class_Array;
      Methods      : in     Method_Array;
      Links        : in     Inheritance_Array;
      Dependencies : out    Dependency_Array;
      Dep_Count    : out    Natural;
      Success      : out    Boolean
   ) is
   begin
      Dep_Count := 0;
      Success := True;

      --  Collect all inheritance dependencies
      for I in Links'Range loop
         if Dep_Count < Dependencies'Length then
            Dep_Count := Dep_Count + 1;
            Dependencies (Dependencies'First + Dep_Count - 1) := (
               Source_Class => Links (I).Child_ID,
               Target_Class => Links (I).Parent_ID,
               Kind         => Inheritance_Dep,
               Count        => 1
            );
         else
            Success := False;
            return;
         end if;
      end loop;
   end Build_Dependency_Graph;

   --  ============================================================
   --  Detect_Circular_Dependencies Implementation
   --  ============================================================

   function Detect_Circular_Dependencies (
      Start_Class  : Class_ID;
      Dependencies : Dependency_Array
   ) return Circular_Dep_Info is
      Visited : Class_ID_Array (1 .. Max_Cycle_Length) := (others => Null_Class_ID);
      V_Count : Natural := 0;

      function Visit (ID : Class_ID; Depth : Natural) return Boolean is
      begin
         --  Check if already visited (cycle detection)
         for I in 1 .. V_Count loop
            if Visited (I) = ID then
               return True;  --  Cycle found!
            end if;
         end loop;

         --  Mark as visited
         if V_Count < Max_Cycle_Length then
            V_Count := V_Count + 1;
            Visited (V_Count) := ID;
         else
            return False;  --  Too deep
         end if;

         --  Visit all dependencies
         for I in Dependencies'Range loop
            if Dependencies (I).Source_Class = ID then
               if Visit (Dependencies (I).Target_Class, Depth + 1) then
                  return True;
               end if;
            end if;
         end loop;

         return False;
      end Visit;

      Result : Circular_Dep_Info := No_Cycle;
   begin
      if Visit (Start_Class, 0) then
         Result.Has_Cycle := True;
         Result.Cycle_Length := V_Count;
         Result.Start_Class := Start_Class;
      end if;
      return Result;
   end Detect_Circular_Dependencies;

   --  ============================================================
   --  Analyze_Class_Coupling Implementation
   --  ============================================================

   function Analyze_Class_Coupling (
      Class        : Class_Info;
      Classes      : Class_Array;
      Methods      : Method_Array;
      Links        : Inheritance_Array;
      Dependencies : Dependency_Array;
      CBO_Threshold : Natural;
      RFC_Threshold : Natural
   ) return Coupling_Result is
      Result    : Coupling_Result := Null_Coupling_Result;
      CBO_Val   : Natural;
      RFC_Val   : Natural;
      LCOM_Val  : Natural;
      Afferent  : Natural;
      Efferent  : Natural;
      Circ_Info : Circular_Dep_Info;
   begin
      Result.Class_ID := Class.ID;

      --  Calculate metrics using Coupling_Metrics package
      CBO_Val := Calculate_CBO (Class.ID, Dependencies);
      RFC_Val := Calculate_RFC (Class.ID, Methods);
      LCOM_Val := Calculate_LCOM (Class.ID, Methods);

      Result.CBO := CBO_Val;
      Result.RFC := RFC_Val;
      Result.LCOM := LCOM_Val;
      Result.DIT := Class.Inheritance_Depth;

      --  Count children
      Result.NOC := 0;
      for I in Links'Range loop
         if Links (I).Parent_ID = Class.ID then
            Result.NOC := Result.NOC + 1;
         end if;
      end loop;

      --  Count methods (WMC simplified as method count)
      Result.WMC := Class.Method_Count;

      --  Calculate afferent and efferent
      Afferent := Count_Afferent (Class.ID, Dependencies);
      Efferent := Count_Efferent (Class.ID, Dependencies);
      Result.Afferent := Afferent;
      Result.Efferent := Efferent;
      Result.Instability := Calculate_Instability (Afferent, Efferent);

      --  Check for circular dependencies
      Circ_Info := Detect_Circular_Dependencies (Class.ID, Dependencies);
      Result.Has_Circular_Dep := Circ_Info.Has_Cycle;

      --  Check thresholds
      Result.Exceeds_Thresholds := (CBO_Val > CBO_Threshold) or (RFC_Val > RFC_Threshold);

      --  Set status
      if Result.Has_Circular_Dep then
         Result.Status := Analysis_Error;
      elsif Result.Exceeds_Thresholds then
         Result.Status := Analysis_Warning;
      else
         Result.Status := Analysis_OK;
      end if;

      return Result;
   end Analyze_Class_Coupling;

   --  ============================================================
   --  Analyze_All_Coupling Implementation
   --  ============================================================

   procedure Analyze_All_Coupling (
      Classes       : in     Class_Array;
      Methods       : in     Method_Array;
      Links         : in     Inheritance_Array;
      CBO_Threshold : in     Natural;
      RFC_Threshold : in     Natural;
      Results       :    out Coupling_Result_Array;
      Summary       :    out Coupling_Summary;
      Success       :    out Boolean
   ) is
      Deps      : Dependency_Array (1 .. Max_Dependencies);
      Dep_Count : Natural;
      Build_OK  : Boolean;
      Total_CBO : Natural := 0;
      Total_RFC : Natural := 0;
      Max_CBO   : Natural := 0;
      Max_RFC   : Natural := 0;
      Over_CBO  : Natural := 0;
      Over_RFC  : Natural := 0;
      Circulars : Natural := 0;
   begin
      Summary := Null_Coupling_Summary;
      Summary.Total_Classes := Classes'Length;

      --  Build dependency graph
      Build_Dependency_Graph (Classes, Methods, Links, Deps, Dep_Count, Build_OK);
      Success := Build_OK;

      --  Analyze each class
      for I in Classes'Range loop
         if I in Results'Range then
            Results (I) := Analyze_Class_Coupling (
               Classes (I), Classes, Methods, Links,
               Deps (1 .. Dep_Count), CBO_Threshold, RFC_Threshold
            );

            --  Update summary statistics
            Total_CBO := Total_CBO + Results (I).CBO;
            Total_RFC := Total_RFC + Results (I).RFC;

            if Results (I).CBO > Max_CBO then
               Max_CBO := Results (I).CBO;
            end if;

            if Results (I).RFC > Max_RFC then
               Max_RFC := Results (I).RFC;
            end if;

            if Results (I).CBO > CBO_Threshold then
               Over_CBO := Over_CBO + 1;
            end if;

            if Results (I).RFC > RFC_Threshold then
               Over_RFC := Over_RFC + 1;
            end if;

            if Results (I).Has_Circular_Dep then
               Circulars := Circulars + 1;
               Success := False;
            end if;
         end if;
      end loop;

      --  Calculate averages (x100 for precision)
      if Classes'Length > 0 then
         Summary.Avg_CBO := (Total_CBO * 100) / Classes'Length;
         Summary.Avg_RFC := (Total_RFC * 100) / Classes'Length;
      end if;

      Summary.Max_CBO := Max_CBO;
      Summary.Max_RFC := Max_RFC;
      Summary.Classes_Over_CBO := Over_CBO;
      Summary.Classes_Over_RFC := Over_RFC;
      Summary.Circular_Deps := Circulars;

      --  Calculate coupling density
      if Classes'Length > 1 then
         Summary.Coupling_Density := (Dep_Count * 100) / (Classes'Length * (Classes'Length - 1));
      end if;
   end Analyze_All_Coupling;

   --  ============================================================
   --  Count_Afferent Implementation
   --  ============================================================

   function Count_Afferent (
      Class_ID     : OOP_Types.Class_ID;
      Dependencies : Dependency_Array
   ) return Natural is
      Count : Natural := 0;
   begin
      for I in Dependencies'Range loop
         if Dependencies (I).Target_Class = Class_ID then
            Count := Count + 1;
         end if;
      end loop;
      return Count;
   end Count_Afferent;

   --  ============================================================
   --  Count_Efferent Implementation
   --  ============================================================

   function Count_Efferent (
      Class_ID     : OOP_Types.Class_ID;
      Dependencies : Dependency_Array
   ) return Natural is
      Count : Natural := 0;
   begin
      for I in Dependencies'Range loop
         if Dependencies (I).Source_Class = Class_ID then
            Count := Count + 1;
         end if;
      end loop;
      return Count;
   end Count_Efferent;

   --  ============================================================
   --  Calculate_Instability Implementation
   --  ============================================================

   function Calculate_Instability (
      Afferent : Natural;
      Efferent : Natural
   ) return Natural is
   begin
      if Afferent + Efferent = 0 then
         return 0;
      end if;
      return (Efferent * 100) / (Afferent + Efferent);
   end Calculate_Instability;

end Coupling_Analyzer;
