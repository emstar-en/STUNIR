--  STUNIR DO-331 IR-to-Model Transformer Specification
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0
--
--  This package provides the main transformation engine that converts
--  STUNIR IR to Model IR suitable for SysML 2.0 generation.

pragma SPARK_Mode (On);

with Model_IR; use Model_IR;

package IR_To_Model is

   --  ============================================================
   --  Transformation Status
   --  ============================================================

   type Transform_Status is (
      Status_Success,
      Status_Partial_Success,
      Status_IR_Parse_Error,
      Status_Validation_Error,
      Status_Resource_Limit,
      Status_Unknown_Error
   );

   --  ============================================================
   --  Transformation Options
   --  ============================================================

   type Transform_Options is record
      DAL_Level                : Model_IR.DAL_Level := DAL_C;
      Include_Coverage_Points  : Boolean := True;
      Include_Traceability     : Boolean := True;
      Emit_Documentation       : Boolean := True;
      Strict_Type_Checking     : Boolean := True;
      Generate_State_Machines  : Boolean := True;
      Max_Elements             : Natural := Max_Elements;
   end record;

   Default_Options : constant Transform_Options := (
      DAL_Level                => DAL_C,
      Include_Coverage_Points  => True,
      Include_Traceability     => True,
      Emit_Documentation       => True,
      Strict_Type_Checking     => True,
      Generate_State_Machines  => True,
      Max_Elements             => Max_Elements
   );

   --  ============================================================
   --  Transformation Result
   --  ============================================================

   type Transform_Result is record
      Status          : Transform_Status;
      Element_Count   : Natural;
      Warning_Count   : Natural;
      Error_Count     : Natural;
      Package_Count   : Natural;
      Action_Count    : Natural;
      State_Count     : Natural;
      Transition_Count: Natural;
   end record;

   Null_Result : constant Transform_Result := (
      Status           => Status_Unknown_Error,
      Element_Count    => 0,
      Warning_Count    => 0,
      Error_Count      => 0,
      Package_Count    => 0,
      Action_Count     => 0,
      State_Count      => 0,
      Transition_Count => 0
   );

   --  ============================================================
   --  Transformation Rules (for traceability)
   --  ============================================================

   type Transformation_Rule is (
      Rule_Module_To_Package,
      Rule_Function_To_Action,
      Rule_Type_To_Attribute_Def,
      Rule_Variable_To_Attribute,
      Rule_If_To_Decision,
      Rule_Loop_To_Action,
      Rule_Return_To_Output,
      Rule_Call_To_Perform,
      Rule_Import_To_Import,
      Rule_Export_To_Expose,
      Rule_State_Machine,
      Rule_Transition,
      Rule_Requirement,
      Rule_Constraint,
      Rule_Satisfy_Relationship
   );

   --  ============================================================
   --  Element Storage (bounded arrays for SPARK)
   --  ============================================================

   type Element_Storage is record
      Elements      : Element_Array (1 .. Max_Elements);
      Count         : Element_Count := 0;
      Action_Data   : array (1 .. Max_Actions) of Model_IR.Action_Data;
      Action_Count  : Natural := 0;
      State_Data    : array (1 .. Max_States) of Model_IR.State_Data;
      State_Count   : Natural := 0;
      Trans_Data    : array (1 .. Max_Transitions) of Model_IR.Transition_Data;
      Trans_Count   : Natural := 0;
   end record;

   --  ============================================================
   --  Main Transformation Interface
   --  ============================================================

   --  Initialize element storage
   procedure Initialize_Storage (Storage : out Element_Storage);

   --  Transform module (package)
   procedure Transform_Module (
      Module_Name : in     String;
      IR_Hash     : in     String;
      Options     : in     Transform_Options;
      Container   : in out Model_Container;
      Storage     : in Out Element_Storage;
      Root_ID     : out    Element_ID
   ) with Pre => Module_Name'Length > 0 and Module_Name'Length <= Max_Name_Length;

   --  Transform function to action
   procedure Transform_Function (
      Func_Name   : in     String;
      Parent_ID   : in     Element_ID;
      Has_Inputs  : in     Boolean;
      Has_Outputs : in     Boolean;
      Input_Count : in     Natural;
      Output_Count: in     Natural;
      Storage     : in Out Element_Storage;
      Action_ID   : out    Element_ID
   ) with Pre => Func_Name'Length > 0 and Func_Name'Length <= Max_Name_Length;

   --  Transform type to attribute definition
   procedure Transform_Type (
      Type_Name   : in     String;
      Parent_ID   : in     Element_ID;
      Storage     : in Out Element_Storage;
      Attr_ID     : out    Element_ID
   ) with Pre => Type_Name'Length > 0 and Type_Name'Length <= Max_Name_Length;

   --  Transform variable to attribute
   procedure Transform_Variable (
      Var_Name    : in     String;
      Var_Type    : in     String;
      Parent_ID   : in     Element_ID;
      Storage     : in Out Element_Storage;
      Var_ID      : out    Element_ID
   ) with Pre => Var_Name'Length > 0 and Var_Name'Length <= Max_Name_Length;

   --  Add a state to the model
   procedure Add_State (
      State_Name  : in     String;
      Parent_ID   : in     Element_ID;
      Is_Initial  : in     Boolean;
      Is_Final    : in     Boolean;
      Storage     : in Out Element_Storage;
      State_ID    : out    Element_ID
   ) with Pre => State_Name'Length > 0 and State_Name'Length <= Max_Name_Length;

   --  Add a transition
   procedure Add_Transition (
      Trans_Name  : in     String;
      Source_ID   : in     Element_ID;
      Target_ID   : in     Element_ID;
      Has_Guard   : in     Boolean;
      Guard_Expr  : in     String;
      Parent_ID   : in     Element_ID;
      Storage     : in Out Element_Storage;
      Trans_ID    : out    Element_ID
   ) with Pre => Trans_Name'Length <= Max_Name_Length;

   --  Add a requirement
   procedure Add_Requirement (
      Req_Name    : in     String;
      Req_Text    : in     String;
      Parent_ID   : in     Element_ID;
      Storage     : in Out Element_Storage;
      Req_ID      : out    Element_ID
   ) with Pre => Req_Name'Length > 0 and Req_Name'Length <= Max_Name_Length;

   --  Add satisfy relationship
   procedure Add_Satisfy (
      Req_ID      : in     Element_ID;
      Element_ID  : in     Model_IR.Element_ID;
      Parent_ID   : in     Model_IR.Element_ID;
      Storage     : in Out Element_Storage;
      Sat_ID      : out    Model_IR.Element_ID
   );

   --  ============================================================
   --  Lookup Operations
   --  ============================================================

   --  Get element by ID
   function Get_Element (
      Storage : Element_Storage;
      ID      : Element_ID
   ) return Model_Element;

   --  Get action data for an action element
   function Get_Action_Data (
      Storage   : Element_Storage;
      Action_ID : Element_ID
   ) return Action_Data;

   --  Get state data for a state element
   function Get_State_Data (
      Storage  : Element_Storage;
      State_ID : Element_ID
   ) return State_Data;

   --  Get transition data
   function Get_Transition_Data (
      Storage  : Element_Storage;
      Trans_ID : Element_ID
   ) return Transition_Data;

   --  Get children of an element
   function Get_Children (
      Storage   : Element_Storage;
      Parent_ID : Element_ID
   ) return Element_ID_Array;

   --  ============================================================
   --  Transformation Rule Information
   --  ============================================================

   --  Get human-readable name for a rule
   function Get_Rule_Name (Rule : Transformation_Rule) return String;

   --  Get DO-331 objective mapping for a rule
   function Get_DO331_Objective (Rule : Transformation_Rule) return String;

end IR_To_Model;
