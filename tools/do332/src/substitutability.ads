--  STUNIR DO-332 Substitutability Checker Specification
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0
--
--  This package implements Liskov Substitution Principle (LSP) checking
--  for DO-332 OO.2 compliance.

pragma SPARK_Mode (On);

with OOP_Types; use OOP_Types;
with OOP_Analysis; use OOP_Analysis;

package Substitutability is

   --  ============================================================
   --  LSP Check Result
   --  ============================================================

   type LSP_Check_Result is (
      LSP_Compliant,
      LSP_Precondition_Strengthened,
      LSP_Postcondition_Weakened,
      LSP_Invariant_Violated,
      LSP_Covariance_Violation,
      LSP_Contravariance_Violation,
      LSP_Exception_Mismatch,
      LSP_Signature_Mismatch
   );

   --  ============================================================
   --  LSP Violation Record
   --  ============================================================

   type LSP_Violation is record
      Parent_Class  : Class_ID;
      Child_Class   : Class_ID;
      Method_ID     : OOP_Types.Method_ID;
      Violation     : LSP_Check_Result;
      Line_Number   : Natural;
   end record;

   type LSP_Violation_Array is array (Positive range <>) of LSP_Violation;

   --  ============================================================
   --  Substitutability Summary
   --  ============================================================

   type Substitutability_Summary is record
      Total_Checked       : Natural;
      Total_Compliant     : Natural;
      Precond_Violations  : Natural;
      Postcond_Violations : Natural;
      Invariant_Violations: Natural;
      Covariance_Violations: Natural;
      Contravariance_Violations: Natural;
      Exception_Violations: Natural;
      Signature_Violations: Natural;
   end record;

   Null_Summary : constant Substitutability_Summary := (
      Total_Checked       => 0,
      Total_Compliant     => 0,
      Precond_Violations  => 0,
      Postcond_Violations => 0,
      Invariant_Violations => 0,
      Covariance_Violations => 0,
      Contravariance_Violations => 0,
      Exception_Violations => 0,
      Signature_Violations => 0
   );

   --  ============================================================
   --  Core LSP Checking Functions
   --  ============================================================

   --  Check LSP for a method override
   function Check_LSP (
      Parent_Method : Method_Info;
      Child_Method  : Method_Info
   ) return LSP_Check_Result
     with Pre => Child_Method.Has_Override 
                 and Child_Method.Override_Of = Parent_Method.ID;

   --  Check if child class is substitutable for parent
   function Is_Substitutable (
      Parent_Class : Class_ID;
      Child_Class  : Class_ID;
      Methods      : Method_Array;
      Links        : Inheritance_Array
   ) return Boolean
     with Pre => Is_Valid_Class_ID (Parent_Class) 
                 and Is_Valid_Class_ID (Child_Class);

   --  Find all LSP violations in hierarchy
   procedure Find_LSP_Violations (
      Classes    : in     Class_Array;
      Methods    : in     Method_Array;
      Links      : in     Inheritance_Array;
      Summary    :    out Substitutability_Summary;
      Has_Errors :    out Boolean
   );

   --  ============================================================
   --  Specific LSP Checks
   --  ============================================================

   --  Check signature compatibility
   function Check_Signature_Compatible (
      Parent_Method : Method_Info;
      Child_Method  : Method_Info
   ) return Boolean;

   --  Check return type covariance
   function Check_Return_Type_Covariance (
      Child_Method  : Method_Info;
      Parent_Method : Method_Info;
      Links         : Inheritance_Array
   ) return Boolean;

   --  Check parameter contravariance
   function Check_Parameter_Contravariance (
      Child_Method  : Method_Info;
      Parent_Method : Method_Info;
      Links         : Inheritance_Array
   ) return Boolean;

end Substitutability;
