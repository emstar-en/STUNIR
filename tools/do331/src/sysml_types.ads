--  STUNIR DO-331 SysML 2.0 Type Definitions
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0
--
--  This package provides SysML 2.0 specific type definitions and
--  mappings from STUNIR IR types to SysML 2.0 types.

pragma SPARK_Mode (On);

with Model_IR; use Model_IR;

package SysML_Types is

   --  ============================================================
   --  SysML 2.0 Primitive Types
   --  ============================================================

   type SysML_Primitive_Type is (
      Boolean_Type,
      Integer_Type,
      Natural_Type,
      Positive_Type,
      Real_Type,
      String_Type,
      Any_Type
   );

   --  ============================================================
   --  SysML 2.0 Visibility
   --  ============================================================

   type Visibility_Kind is (
      Public_Visibility,
      Protected_Visibility,
      Private_Visibility,
      Package_Visibility
   );

   --  ============================================================
   --  SysML 2.0 Keywords
   --  ============================================================

   Max_Keyword_Length : constant := 32;
   subtype Keyword_String is String (1 .. Max_Keyword_Length);

   --  Get the SysML 2.0 keyword for an element kind
   function Get_Keyword (Kind : Element_Kind) return String
     with Post => Get_Keyword'Result'Length > 0 and
                  Get_Keyword'Result'Length <= Max_Keyword_Length;

   --  ============================================================
   --  Type Mapping
   --  ============================================================

   --  Map IR type name to SysML primitive type
   function IR_Type_To_SysML (IR_Type : String) return SysML_Primitive_Type
     with Pre => IR_Type'Length > 0;

   --  Get SysML type name string
   function Get_Type_Name (T : SysML_Primitive_Type) return String
     with Post => Get_Type_Name'Result'Length > 0;

   --  ============================================================
   --  Operator Mapping
   --  ============================================================

   type SysML_Operator is (
      Op_Add,
      Op_Subtract,
      Op_Multiply,
      Op_Divide,
      Op_Modulo,
      Op_Equal,
      Op_Not_Equal,
      Op_Less,
      Op_Less_Equal,
      Op_Greater,
      Op_Greater_Equal,
      Op_And,
      Op_Or,
      Op_Not,
      Op_Assign,
      Op_Unknown
   );

   --  Map IR operator to SysML operator
   function IR_Op_To_SysML (IR_Op : String) return SysML_Operator
     with Pre => IR_Op'Length > 0;

   --  Get SysML operator symbol string
   function Get_Operator_Symbol (Op : SysML_Operator) return String;

   --  ============================================================
   --  State Machine Types
   --  ============================================================

   type SysML_State_Kind is (
      State_Simple,
      State_Composite,
      State_Initial,
      State_Final,
      State_Choice,
      State_Fork,
      State_Join,
      State_History,
      State_Deep_History
   );

   --  Get SysML state keyword
   function Get_State_Keyword (Kind : SysML_State_Kind) return String;

   --  ============================================================
   --  Relationship Types
   --  ============================================================

   type SysML_Relationship is (
      Rel_Satisfy,
      Rel_Verify,
      Rel_Derive,
      Rel_Refine,
      Rel_Trace,
      Rel_Allocate,
      Rel_Copy,
      Rel_Expose
   );

   --  Get relationship keyword
   function Get_Relationship_Keyword (Rel : SysML_Relationship) return String;

   --  ============================================================
   --  Standard Library References
   --  ============================================================

   --  SysML 2.0 standard library import strings
   function Get_Standard_Import (T : SysML_Primitive_Type) return String;

end SysML_Types;
