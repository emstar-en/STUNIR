--  STUNIR DO-332 Coupling Metrics Specification
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0
--
--  This package implements coupling metrics calculation for DO-332 OO.4.

pragma SPARK_Mode (On);

with OOP_Types; use OOP_Types;
with OOP_Analysis; use OOP_Analysis;

package Coupling_Metrics is

   --  ============================================================
   --  Constants
   --  ============================================================

   --  Standard thresholds from literature
   Default_CBO_Threshold : constant := 14;
   Default_RFC_Threshold : constant := 50;
   Default_LCOM_Threshold : constant := 100;
   Default_DIT_Threshold : constant := 5;
   Default_NOC_Threshold : constant := 10;
   Default_WMC_Threshold : constant := 25;

   --  ============================================================
   --  Metric Calculation Functions
   --  ============================================================

   --  CBO: Coupling Between Objects
   --  Count of classes to which this class is coupled
   function Calculate_CBO (
      Class_ID     : OOP_Types.Class_ID;
      Dependencies : Dependency_Array
   ) return Natural;

   --  RFC: Response For Class
   --  Number of methods that can be invoked in response to a message
   function Calculate_RFC (
      Class_ID : OOP_Types.Class_ID;
      Methods  : Method_Array
   ) return Natural;

   --  LCOM: Lack of Cohesion in Methods
   --  Measures how closely methods are related via shared attributes
   function Calculate_LCOM (
      Class_ID : OOP_Types.Class_ID;
      Methods  : Method_Array
   ) return Natural;

   --  DIT: Depth of Inheritance Tree
   --  Maximum length from class to root of tree
   function Calculate_DIT (
      Class : Class_Info
   ) return Natural is (Class.Inheritance_Depth);

   --  NOC: Number of Children
   --  Number of immediate subclasses
   function Calculate_NOC (
      Class_ID : OOP_Types.Class_ID;
      Links    : Inheritance_Array
   ) return Natural;

   --  WMC: Weighted Methods per Class
   --  Sum of complexities of methods (simplified as method count)
   function Calculate_WMC (
      Class_ID : OOP_Types.Class_ID;
      Methods  : Method_Array
   ) return Natural;

   --  ============================================================
   --  Threshold Checking
   --  ============================================================

   type Threshold_Set is record
      CBO_Threshold  : Natural;
      RFC_Threshold  : Natural;
      LCOM_Threshold : Natural;
      DIT_Threshold  : Natural;
      NOC_Threshold  : Natural;
      WMC_Threshold  : Natural;
   end record;

   Default_Thresholds : constant Threshold_Set := (
      CBO_Threshold  => Default_CBO_Threshold,
      RFC_Threshold  => Default_RFC_Threshold,
      LCOM_Threshold => Default_LCOM_Threshold,
      DIT_Threshold  => Default_DIT_Threshold,
      NOC_Threshold  => Default_NOC_Threshold,
      WMC_Threshold  => Default_WMC_Threshold
   );

   type Threshold_Violations is record
      CBO_Exceeded  : Boolean;
      RFC_Exceeded  : Boolean;
      LCOM_Exceeded : Boolean;
      DIT_Exceeded  : Boolean;
      NOC_Exceeded  : Boolean;
      WMC_Exceeded  : Boolean;
   end record;

   No_Violations : constant Threshold_Violations := (
      CBO_Exceeded  => False,
      RFC_Exceeded  => False,
      LCOM_Exceeded => False,
      DIT_Exceeded  => False,
      NOC_Exceeded  => False,
      WMC_Exceeded  => False
   );

   --  Check all thresholds for a class
   function Check_Thresholds (
      CBO        : Natural;
      RFC        : Natural;
      LCOM       : Natural;
      DIT        : Natural;
      NOC        : Natural;
      WMC        : Natural;
      Thresholds : Threshold_Set
   ) return Threshold_Violations;

   --  Check if any threshold is exceeded
   function Any_Exceeded (
      Violations : Threshold_Violations
   ) return Boolean is (
      Violations.CBO_Exceeded or
      Violations.RFC_Exceeded or
      Violations.LCOM_Exceeded or
      Violations.DIT_Exceeded or
      Violations.NOC_Exceeded or
      Violations.WMC_Exceeded
   );

end Coupling_Metrics;
