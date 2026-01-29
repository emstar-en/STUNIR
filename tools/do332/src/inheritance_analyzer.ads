--  STUNIR DO-332 Inheritance Analyzer Specification
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0
--
--  This package implements DO-332 OO.1 inheritance analysis,
--  including depth calculation, diamond detection, and override verification.

pragma SPARK_Mode (On);

with OOP_Types; use OOP_Types;
with OOP_Analysis; use OOP_Analysis;

package Inheritance_Analyzer is

   --  ============================================================
   --  Constants
   --  ============================================================

   Max_Visited : constant := 5_000;

   --  ============================================================
   --  Diamond Pattern Information
   --  ============================================================

   type Diamond_Info is record
      Has_Diamond       : Boolean;
      Common_Ancestor   : Class_ID;
      Left_Path_Length  : Natural;
      Right_Path_Length : Natural;
      Join_Class        : Class_ID;
   end record;

   Null_Diamond_Info : constant Diamond_Info := (
      Has_Diamond       => False,
      Common_Ancestor   => Null_Class_ID,
      Left_Path_Length  => 0,
      Right_Path_Length => 0,
      Join_Class        => Null_Class_ID
   );

   --  ============================================================
   --  Override Verification Result
   --  ============================================================

   type Override_Status is (
      Override_OK,
      Override_Signature_Mismatch,
      Override_Return_Type_Mismatch,
      Override_Visibility_Reduced,
      Override_Of_Final_Method,
      Override_Not_Virtual
   );

   type Override_Result is record
      Child_Method    : Method_ID;
      Parent_Method   : Method_ID;
      Status          : Override_Status;
   end record;

   --  ============================================================
   --  Core Analysis Functions
   --  ============================================================

   --  Calculate inheritance depth for a class
   function Calculate_Depth (
      Class_ID : OOP_Types.Class_ID;
      Links    : Inheritance_Array
   ) return Natural
     with Post => Calculate_Depth'Result <= Max_Inheritance;

   --  Detect diamond inheritance pattern
   function Detect_Diamond_Pattern (
      Class_ID : OOP_Types.Class_ID;
      Links    : Inheritance_Array
   ) return Diamond_Info;

   --  Detect circular inheritance (error condition)
   function Has_Circular_Inheritance (
      Class_ID : OOP_Types.Class_ID;
      Links    : Inheritance_Array
   ) return Boolean;

   --  Get all ancestors of a class (transitive closure)
   function Get_All_Ancestors (
      Class_ID : OOP_Types.Class_ID;
      Links    : Inheritance_Array
   ) return Class_ID_Array;

   --  Verify a method override is correct
   function Verify_Override (
      Child_Method  : Method_Info;
      Parent_Method : Method_Info
   ) return Override_Status
     with Pre => Child_Method.Has_Override;

   --  Analyze all overrides in a class
   function Analyze_Class_Overrides (
      Class_ID : OOP_Types.Class_ID;
      Methods  : Method_Array;
      Links    : Inheritance_Array
   ) return Natural;  --  Returns count of invalid overrides

   --  Full inheritance analysis for a class
   function Analyze_Inheritance (
      Class   : Class_Info;
      Classes : Class_Array;
      Methods : Method_Array;
      Links   : Inheritance_Array
   ) return Inheritance_Result
     with Pre  => Is_Valid_Class_ID (Class.ID),
          Post => Analyze_Inheritance'Result.Class_ID = Class.ID;

   --  Analyze complete hierarchy
   procedure Analyze_All_Inheritance (
      Classes  : in     Class_Array;
      Methods  : in     Method_Array;
      Links    : in     Inheritance_Array;
      Results  :    out Inheritance_Result_Array;
      Success  :    out Boolean
   ) with Pre => Results'Length >= Classes'Length;

   --  ============================================================
   --  Utility Functions
   --  ============================================================

   --  Check if Class A is an ancestor of Class B
   function Is_Ancestor (
      Ancestor_ID   : Class_ID;
      Descendant_ID : Class_ID;
      Links         : Inheritance_Array
   ) return Boolean;

   --  Count total overrides in a class
   function Count_Overrides (
      Class_ID : OOP_Types.Class_ID;
      Methods  : Method_Array
   ) return Natural;

   --  Count virtual methods in a class
   function Count_Virtual_Methods (
      Class_ID : OOP_Types.Class_ID;
      Methods  : Method_Array
   ) return Natural;

   --  Count abstract methods in a class
   function Count_Abstract_Methods (
      Class_ID : OOP_Types.Class_ID;
      Methods  : Method_Array
   ) return Natural;

end Inheritance_Analyzer;
