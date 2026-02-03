--  STUNIR DO-332 Test Generator Specification
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0
--
--  This package generates test cases for DO-332 OOP verification.

pragma SPARK_Mode (On);

with OOP_Types; use OOP_Types;
with OOP_Analysis; use OOP_Analysis;
with Dispatch_Analyzer; use Dispatch_Analyzer;

package Test_Generator is

   --  ============================================================
   --  Constants
   --  ============================================================

   Max_Test_Cases    : constant := 10_000;
   Max_Setup_Steps   : constant := 20;
   Max_Assertions    : constant := 50;

   --  ============================================================
   --  Test Case Record
   --  ============================================================

   type Test_Case is record
      Test_ID       : Natural;
      Category      : Test_Category;
      Objective     : DO332_Objective;
      Target_Class  : Class_ID;
      Target_Method : Method_ID;
      Site_ID       : Natural;  --  For dispatch tests
      Setup_Steps   : Natural;
      Assertions    : Natural;
      Coverage_Points : Natural;
   end record;

   Null_Test_Case : constant Test_Case := (
      Test_ID       => 0,
      Category      => Dispatch_Test,
      Objective     => OO_3,
      Target_Class  => Null_Class_ID,
      Target_Method => Null_Method_ID,
      Site_ID       => 0,
      Setup_Steps   => 0,
      Assertions    => 0,
      Coverage_Points => 0
   );

   type Test_Case_Array is array (Positive range <>) of Test_Case;

   --  ============================================================
   --  Test Generation Summary
   --  ============================================================

   type Generation_Summary is record
      Total_Tests        : Natural;
      Inheritance_Tests  : Natural;
      Override_Tests     : Natural;
      Polymorphism_Tests : Natural;
      Dispatch_Tests     : Natural;
      Coupling_Tests     : Natural;
      Lifecycle_Tests    : Natural;
      Coverage_Points    : Natural;
   end record;

   Null_Generation_Summary : constant Generation_Summary := (
      Total_Tests        => 0,
      Inheritance_Tests  => 0,
      Override_Tests     => 0,
      Polymorphism_Tests => 0,
      Dispatch_Tests     => 0,
      Coupling_Tests     => 0,
      Lifecycle_Tests    => 0,
      Coverage_Points    => 0
   );

   --  ============================================================
   --  Core Generation Functions
   --  ============================================================

   --  Generate inheritance test cases
   function Generate_Inheritance_Tests (
      Classes : Class_Array;
      Links   : Inheritance_Array;
      Results : Inheritance_Result_Array
   ) return Test_Case_Array;

   --  Generate polymorphism test cases
   function Generate_Polymorphism_Tests (
      Classes : Class_Array;
      Methods : Method_Array;
      Results : Polymorphism_Result_Array
   ) return Test_Case_Array;

   --  Generate dispatch test cases (one per target)
   function Generate_Dispatch_Tests (
      Sites    : Dispatch_Site_Array;
      Classes  : Class_Array;
      Methods  : Method_Array;
      Links    : Inheritance_Array
   ) return Test_Case_Array;

   --  Generate coupling test cases
   function Generate_Coupling_Tests (
      Classes : Class_Array;
      Results : Coupling_Result_Array
   ) return Test_Case_Array;

   --  Generate all test cases
   procedure Generate_All_Tests (
      Classes       : in     Class_Array;
      Methods       : in     Method_Array;
      Links         : in     Inheritance_Array;
      Dispatch_Sites: in     Dispatch_Site_Array;
      Inh_Results   : in     Inheritance_Result_Array;
      Poly_Results  : in     Polymorphism_Result_Array;
      Coup_Results  : in     Coupling_Result_Array;
      Tests         :    out Test_Case_Array;
      Test_Count    :    out Natural;
      Summary       :    out Generation_Summary
   ) with Pre => Tests'Length >= Max_Test_Cases;

   --  ============================================================
   --  Test Coverage Functions
   --  ============================================================

   --  Check if all dispatch targets are covered
   function All_Targets_Covered (
      Tests  : Test_Case_Array;
      Sites  : Dispatch_Site_Array;
      Results : Site_Analysis_Array
   ) return Boolean;

   --  Count coverage points in tests
   function Count_Coverage_Points (
      Tests : Test_Case_Array
   ) return Natural;

end Test_Generator;
