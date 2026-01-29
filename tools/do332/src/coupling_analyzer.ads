--  STUNIR DO-332 Coupling Analyzer Specification
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0
--
--  This package implements DO-332 OO.4 object coupling analysis,
--  including dependency tracking and coupling metric calculation.

pragma SPARK_Mode (On);

with OOP_Types; use OOP_Types;
with OOP_Analysis; use OOP_Analysis;

package Coupling_Analyzer is

   --  ============================================================
   --  Constants
   --  ============================================================

   Max_Dependencies : constant := 10_000;
   Max_Cycle_Length : constant := 100;

   --  ============================================================
   --  Dependency Graph
   --  ============================================================

   type Dependency_Graph is record
      Class_Count      : Natural;
      Dependency_Count : Natural;
   end record;

   Null_Graph : constant Dependency_Graph := (
      Class_Count      => 0,
      Dependency_Count => 0
   );

   --  ============================================================
   --  Circular Dependency Information
   --  ============================================================

   type Circular_Dep_Info is record
      Has_Cycle    : Boolean;
      Cycle_Length : Natural;
      Start_Class  : Class_ID;
   end record;

   No_Cycle : constant Circular_Dep_Info := (
      Has_Cycle    => False,
      Cycle_Length => 0,
      Start_Class  => Null_Class_ID
   );

   --  ============================================================
   --  Coupling Summary
   --  ============================================================

   type Coupling_Summary is record
      Total_Classes        : Natural;
      Avg_CBO              : Natural;  --  x100 for precision
      Max_CBO              : Natural;
      Avg_RFC              : Natural;  --  x100 for precision
      Max_RFC              : Natural;
      Avg_LCOM             : Natural;  --  x100 for precision
      Max_LCOM             : Natural;
      Classes_Over_CBO     : Natural;
      Classes_Over_RFC     : Natural;
      Circular_Deps        : Natural;
      Coupling_Density     : Natural;  --  percentage
   end record;

   Null_Coupling_Summary : constant Coupling_Summary := (
      Total_Classes       => 0,
      Avg_CBO             => 0,
      Max_CBO             => 0,
      Avg_RFC             => 0,
      Max_RFC             => 0,
      Avg_LCOM            => 0,
      Max_LCOM            => 0,
      Classes_Over_CBO    => 0,
      Classes_Over_RFC    => 0,
      Circular_Deps       => 0,
      Coupling_Density    => 0
   );

   --  ============================================================
   --  Core Analysis Functions
   --  ============================================================

   --  Analyze dependencies for a single class
   function Analyze_Class_Dependencies (
      Class   : Class_Info;
      Classes : Class_Array;
      Methods : Method_Array;
      Links   : Inheritance_Array
   ) return Dependency_Array;

   --  Build complete dependency graph
   procedure Build_Dependency_Graph (
      Classes      : in     Class_Array;
      Methods      : in     Method_Array;
      Links        : in     Inheritance_Array;
      Dependencies : out    Dependency_Array;
      Dep_Count    : out    Natural;
      Success      : out    Boolean
   ) with Pre => Dependencies'Length >= Max_Dependencies;

   --  Detect circular dependencies
   function Detect_Circular_Dependencies (
      Start_Class  : Class_ID;
      Dependencies : Dependency_Array
   ) return Circular_Dep_Info;

   --  Full coupling analysis for a class
   function Analyze_Class_Coupling (
      Class        : Class_Info;
      Classes      : Class_Array;
      Methods      : Method_Array;
      Links        : Inheritance_Array;
      Dependencies : Dependency_Array;
      CBO_Threshold : Natural;
      RFC_Threshold : Natural
   ) return Coupling_Result
     with Pre  => Is_Valid_Class_ID (Class.ID),
          Post => Analyze_Class_Coupling'Result.Class_ID = Class.ID;

   --  Analyze all coupling in hierarchy
   procedure Analyze_All_Coupling (
      Classes       : in     Class_Array;
      Methods       : in     Method_Array;
      Links         : in     Inheritance_Array;
      CBO_Threshold : in     Natural;
      RFC_Threshold : in     Natural;
      Results       :    out Coupling_Result_Array;
      Summary       :    out Coupling_Summary;
      Success       :    out Boolean
   ) with Pre => Results'Length >= Classes'Length;

   --  ============================================================
   --  Dependency Counting Functions
   --  ============================================================

   --  Count afferent coupling (incoming dependencies)
   function Count_Afferent (
      Class_ID     : OOP_Types.Class_ID;
      Dependencies : Dependency_Array
   ) return Natural;

   --  Count efferent coupling (outgoing dependencies)
   function Count_Efferent (
      Class_ID     : OOP_Types.Class_ID;
      Dependencies : Dependency_Array
   ) return Natural;

   --  Calculate instability (Ce / (Ca + Ce)) * 100
   function Calculate_Instability (
      Afferent : Natural;
      Efferent : Natural
   ) return Natural;

end Coupling_Analyzer;
