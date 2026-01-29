--  STUNIR DO-332 Inheritance Metrics Specification
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0
--
--  This package calculates inheritance-related metrics for DO-332 analysis.

pragma SPARK_Mode (On);

with OOP_Types; use OOP_Types;
with OOP_Analysis; use OOP_Analysis;

package Inheritance_Metrics is

   --  ============================================================
   --  Metrics Record
   --  ============================================================

   type Metric_Value is new Natural;

   type Inheritance_Metrics_Record is record
      Total_Classes            : Metric_Value;
      Max_Depth                : Metric_Value;
      Avg_Depth_X100           : Metric_Value;  --  Average * 100 for precision
      Classes_With_Multiple    : Metric_Value;
      Diamond_Patterns         : Metric_Value;
      Circular_Errors          : Metric_Value;
      Total_Overrides          : Metric_Value;
      Invalid_Overrides        : Metric_Value;
      Abstract_Classes         : Metric_Value;
      Unimplemented_Abstract   : Metric_Value;
      Interface_Count          : Metric_Value;
      Total_Virtual_Methods    : Metric_Value;
   end record;

   Null_Metrics : constant Inheritance_Metrics_Record := (
      Total_Classes          => 0,
      Max_Depth              => 0,
      Avg_Depth_X100         => 0,
      Classes_With_Multiple  => 0,
      Diamond_Patterns       => 0,
      Circular_Errors        => 0,
      Total_Overrides        => 0,
      Invalid_Overrides      => 0,
      Abstract_Classes       => 0,
      Unimplemented_Abstract => 0,
      Interface_Count        => 0,
      Total_Virtual_Methods  => 0
   );

   --  ============================================================
   --  Analysis Functions
   --  ============================================================

   --  Calculate all inheritance metrics from analysis results
   function Calculate_Metrics (
      Classes : Class_Array;
      Methods : Method_Array;
      Links   : Inheritance_Array;
      Results : Inheritance_Result_Array
   ) return Inheritance_Metrics_Record
     with Pre => Classes'Length <= Max_Classes
                 and Results'Length >= Classes'Length;

   --  Calculate depth metrics only
   function Calculate_Depth_Metrics (
      Results : Inheritance_Result_Array
   ) return Inheritance_Metrics_Record;

   --  Calculate override metrics only
   function Calculate_Override_Metrics (
      Classes : Class_Array;
      Methods : Method_Array;
      Links   : Inheritance_Array
   ) return Inheritance_Metrics_Record;

   --  Check if metrics meet DO-332 thresholds
   function Metrics_Pass_Thresholds (
      Metrics      : Inheritance_Metrics_Record;
      Max_Depth    : Natural;
      Max_Multiple : Natural
   ) return Boolean;

end Inheritance_Metrics;
