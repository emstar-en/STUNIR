--  STUNIR DO-332 Polymorphism Verifier Specification
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0
--
--  This package implements DO-332 OO.2 polymorphism verification,
--  including virtual function analysis and type substitution checking.

pragma SPARK_Mode (On);

with OOP_Types; use OOP_Types;
with OOP_Analysis; use OOP_Analysis;

package Polymorphism_Verifier is

   --  ============================================================
   --  Constants
   --  ============================================================

   Max_Virtual_Methods    : constant := 10_000;
   Max_Polymorphic_Calls  : constant := 50_000;

   --  ============================================================
   --  Polymorphic Call Site
   --  ============================================================

   type Call_Site is record
      Site_ID       : Natural;
      Location_Line : Natural;
      Receiver_Type : Class_ID;
      Method_Name   : Name_String;
      Name_Length   : Natural;
      Static_Type   : Class_ID;
      Is_Interface  : Boolean;
   end record;

   Null_Call_Site : constant Call_Site := (
      Site_ID       => 0,
      Location_Line => 0,
      Receiver_Type => Null_Class_ID,
      Method_Name   => (others => ' '),
      Name_Length   => 0,
      Static_Type   => Null_Class_ID,
      Is_Interface  => False
   );

   type Call_Site_Array is array (Positive range <>) of Call_Site;

   --  ============================================================
   --  Virtual Method Information
   --  ============================================================

   type Virtual_Method_Info is record
      Method_ID       : Method_ID;
      Declaring_Class : Class_ID;
      Override_Count  : Natural;
      Is_Abstract     : Boolean;
      Is_Final        : Boolean;
      All_Impl_Found  : Boolean;
   end record;

   type Virtual_Method_Array is array (Positive range <>) of Virtual_Method_Info;

   --  ============================================================
   --  LSP Check Result (from substitutability)
   --  ============================================================

   type LSP_Status is (
      LSP_OK,
      LSP_Precondition_Strengthened,
      LSP_Postcondition_Weakened,
      LSP_Invariant_Violated,
      LSP_Covariance_Violation,
      LSP_Contravariance_Violation,
      LSP_Exception_Added
   );

   type LSP_Result is record
      Parent_Class  : Class_ID;
      Child_Class   : Class_ID;
      Method_ID     : OOP_Types.Method_ID;
      Status        : LSP_Status;
   end record;

   type LSP_Result_Array is array (Positive range <>) of LSP_Result;

   --  ============================================================
   --  Core Verification Functions
   --  ============================================================

   --  Identify all virtual methods in hierarchy
   function Scan_Virtual_Methods (
      Methods : Method_Array
   ) return Virtual_Method_Array;

   --  Verify all polymorphic call sites are bounded
   function Verify_Polymorphic_Calls (
      Call_Sites : Call_Site_Array;
      Classes    : Class_Array;
      Links      : Inheritance_Array
   ) return Boolean;

   --  Count possible types for a polymorphic call
   function Count_Possible_Types (
      Static_Type : Class_ID;
      Links       : Inheritance_Array
   ) return Natural;

   --  Full polymorphism verification for a class
   function Verify_Class_Polymorphism (
      Class   : Class_Info;
      Classes : Class_Array;
      Methods : Method_Array;
      Links   : Inheritance_Array
   ) return Polymorphism_Result
     with Pre  => Is_Valid_Class_ID (Class.ID),
          Post => Verify_Class_Polymorphism'Result.Class_ID = Class.ID;

   --  Verify all polymorphism in hierarchy
   procedure Verify_All_Polymorphism (
      Classes  : in     Class_Array;
      Methods  : in     Method_Array;
      Links    : in     Inheritance_Array;
      Results  :    out Polymorphism_Result_Array;
      Success  :    out Boolean
   ) with Pre => Results'Length >= Classes'Length;

   --  ============================================================
   --  Type Checking Functions
   --  ============================================================

   --  Check if a type substitution is safe
   function Is_Safe_Substitution (
      Source_Type : Class_ID;
      Target_Type : Class_ID;
      Links       : Inheritance_Array
   ) return Boolean;

   --  Check covariant return type
   function Check_Covariance (
      Child_Return  : Class_ID;
      Parent_Return : Class_ID;
      Links         : Inheritance_Array
   ) return Boolean;

   --  Check contravariant parameter type
   function Check_Contravariance (
      Child_Param  : Class_ID;
      Parent_Param : Class_ID;
      Links        : Inheritance_Array
   ) return Boolean;

end Polymorphism_Verifier;
