--  STUNIR DO-332 Test Templates Specification
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0
--
--  This package provides test case templates for DO-332 verification.

pragma SPARK_Mode (On);

with OOP_Types; use OOP_Types;

package Test_Templates is

   --  ============================================================
   --  Constants
   --  ============================================================

   Max_Template_Length : constant := 4096;
   Max_Step_Length     : constant := 256;

   --  ============================================================
   --  Template Types
   --  ============================================================

   subtype Template_String is String (1 .. Max_Template_Length);
   subtype Step_String is String (1 .. Max_Step_Length);

   type Template_Kind is (
      Inheritance_Template,
      Override_Template,
      Polymorphism_Template,
      Dispatch_Template,
      Coupling_Template,
      Lifecycle_Template
   );

   --  ============================================================
   --  Test Template Record
   --  ============================================================

   type Test_Template is record
      Kind            : Template_Kind;
      Name            : Name_String;
      Name_Length     : Natural;
      Description     : Template_String;
      Desc_Length     : Natural;
      Setup_Template  : Template_String;
      Setup_Length    : Natural;
      Action_Template : Template_String;
      Action_Length   : Natural;
      Assert_Template : Template_String;
      Assert_Length   : Natural;
   end record;

   --  ============================================================
   --  Standard Templates
   --  ============================================================

   --  Get template for inheritance testing
   function Get_Inheritance_Template return Test_Template;

   --  Get template for override testing
   function Get_Override_Template return Test_Template;

   --  Get template for polymorphism testing
   function Get_Polymorphism_Template return Test_Template;

   --  Get template for dispatch testing
   function Get_Dispatch_Template return Test_Template;

   --  Get template for coupling testing
   function Get_Coupling_Template return Test_Template;

   --  Get template for lifecycle testing
   function Get_Lifecycle_Template return Test_Template;

   --  Get template by kind
   function Get_Template (Kind : Template_Kind) return Test_Template;

   --  ============================================================
   --  Template Application
   --  ============================================================

   type Substitution is record
      Placeholder : Name_String;
      Placeholder_Len : Natural;
      Value       : Name_String;
      Value_Len   : Natural;
   end record;

   type Substitution_Array is array (Positive range <>) of Substitution;

   --  Apply substitutions to a template string
   function Apply_Substitutions (
      Template      : String;
      Substitutions : Substitution_Array
   ) return String;

end Test_Templates;
