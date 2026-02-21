--  STUNIR DO-331 Model IR Specification
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0
--
--  This package provides the core data structures for representing
--  models in an intermediate representation suitable for transformation
--  to SysML 2.0 and other model formats.

pragma SPARK_Mode (On);

package Model_IR is

   --  ============================================================
   --  Constants and Limits
   --  ============================================================

   Max_Elements       : constant := 10_000;
   Max_Name_Length    : constant := 256;
   Max_Path_Length    : constant := 1024;
   Max_Attributes     : constant := 100;
   Max_Transitions    : constant := 500;
   Max_Actions        : constant := 200;
   Max_States         : constant := 100;
   Max_Children       : constant := 500;
   Max_Hash_Length    : constant := 64;

   --  ============================================================
   --  Basic Types
   --  ============================================================

   --  Bounded string types
   subtype Name_String is String (1 .. Max_Name_Length);
   subtype Path_String is String (1 .. Max_Path_Length);
   subtype Hash_String is String (1 .. Max_Hash_Length);

   --  Unique element identifier
   type Element_ID is mod 2**64;

   Null_Element_ID : constant Element_ID := 0;

   --  Element count type
   subtype Element_Count is Natural range 0 .. Max_Elements;

   --  ============================================================
   --  Element Kind Enumeration
   --  ============================================================

   type Element_Kind is (
      --  Structural elements
      Package_Element,
      Part_Element,
      Attribute_Element,
      Connection_Element,
      Port_Element,
      Interface_Element,

      --  Behavioral elements
      Action_Element,
      State_Element,
      Transition_Element,
      Activity_Element,

      --  Requirements elements
      Requirement_Element,
      Constraint_Element,

      --  Traceability elements
      Satisfy_Element,
      Verify_Element,
      Derive_Element,
      Allocate_Element,

      --  Auxiliary elements
      Comment_Element,
      Import_Element
   );

   --  ============================================================
   --  DAL Level Types
   --  ============================================================

   type DAL_Level is (DAL_A, DAL_B, DAL_C, DAL_D, DAL_E);

   type DAL_Level_Set is array (DAL_Level) of Boolean;

   All_DAL_Levels : constant DAL_Level_Set := (others => True);
   No_DAL_Levels  : constant DAL_Level_Set := (others => False);

   --  DAL coverage requirements per DO-331 Table MB-A.5
   type DAL_Coverage_Requirement is record
      Requires_MCDC              : Boolean;
      Requires_Decision_Coverage : Boolean;
      Requires_Statement_Coverage: Boolean;
      Requires_State_Coverage    : Boolean;
      Requires_Transition_Coverage: Boolean;
   end record;

   function Get_DAL_Requirements (Level : DAL_Level) return DAL_Coverage_Requirement;

   --  ============================================================
   --  Model Element Record
   --  ============================================================

   type Model_Element is record
      ID           : Element_ID;
      Kind         : Element_Kind;
      Name         : Name_String;
      Name_Length  : Natural;
      Parent_ID    : Element_ID;
      Child_Count  : Natural;
      Line_Number  : Natural;
      Is_Abstract  : Boolean;
      Is_Root      : Boolean;
   end record;

   --  Default/null element
   Null_Element : constant Model_Element := (
      ID          => Null_Element_ID,
      Kind        => Package_Element,
      Name        => (others => ' '),
      Name_Length => 0,
      Parent_ID   => Null_Element_ID,
      Child_Count => 0,
      Line_Number => 0,
      Is_Abstract => False,
      Is_Root     => False
   );

   --  ============================================================
   --  Action-specific Data
   --  ============================================================

   type Parameter_Direction is (Dir_In, Dir_Out, Dir_InOut);

   type Parameter_Info is record
      Name        : Name_String;
      Name_Length : Natural;
      Type_Name   : Name_String;
      Type_Length : Natural;
      Direction   : Parameter_Direction;
   end record;

   type Parameter_Array is array (Positive range <>) of Parameter_Info;

   type Action_Data is record
      Has_Inputs    : Boolean;
      Has_Outputs   : Boolean;
      Input_Count   : Natural;
      Output_Count  : Natural;
      Is_Entry      : Boolean;
      Is_Exit       : Boolean;
   end record;

   Null_Action_Data : constant Action_Data := (
      Has_Inputs   => False,
      Has_Outputs  => False,
      Input_Count  => 0,
      Output_Count => 0,
      Is_Entry     => False,
      Is_Exit      => False
   );

   --  ============================================================
   --  State-specific Data
   --  ============================================================

   type State_Kind is (
      Simple_State,
      Composite_State,
      Initial_State,
      Final_State,
      Choice_State,
      Fork_State,
      Join_State,
      History_State
   );

   type State_Data is record
      State_Type       : State_Kind;
      Is_Initial       : Boolean;
      Is_Final         : Boolean;
      Has_Entry_Action : Boolean;
      Has_Exit_Action  : Boolean;
      Has_Do_Activity  : Boolean;
      Entry_Action_ID  : Element_ID;
      Exit_Action_ID   : Element_ID;
   end record;

   Null_State_Data : constant State_Data := (
      State_Type       => Simple_State,
      Is_Initial       => False,
      Is_Final         => False,
      Has_Entry_Action => False,
      Has_Exit_Action  => False,
      Has_Do_Activity  => False,
      Entry_Action_ID  => Null_Element_ID,
      Exit_Action_ID   => Null_Element_ID
   );

   --  ============================================================
   --  Transition-specific Data
   --  ============================================================

   type Transition_Data is record
      Source_State_ID : Element_ID;
      Target_State_ID : Element_ID;
      Has_Trigger     : Boolean;
      Has_Guard       : Boolean;
      Has_Effect      : Boolean;
      Trigger_Name    : Name_String;
      Trigger_Length  : Natural;
      Guard_Expr      : Name_String;
      Guard_Length    : Natural;
   end record;

   Null_Transition_Data : constant Transition_Data := (
      Source_State_ID => Null_Element_ID,
      Target_State_ID => Null_Element_ID,
      Has_Trigger     => False,
      Has_Guard       => False,
      Has_Effect      => False,
      Trigger_Name    => (others => ' '),
      Trigger_Length  => 0,
      Guard_Expr      => (others => ' '),
      Guard_Length    => 0
   );

   --  ============================================================
   --  Requirement-specific Data
   --  ============================================================

   type Requirement_Data is record
      Req_Text        : Path_String;
      Text_Length     : Natural;
      Priority        : Natural;
      Is_Derived      : Boolean;
   end record;

   --  ============================================================
   --  Model Container
   --  ============================================================

   type Element_Array is array (Positive range <>) of Model_Element;
   type Element_ID_Array is array (Positive range <>) of Element_ID;

   type Model_Container is record
      Schema_Version   : Natural;
      Element_Count    : Element_Count;
      IR_Source_Hash   : Hash_String;
      Hash_Length      : Natural;
      Generation_Epoch : Natural;
      DAL_Level        : Model_IR.DAL_Level;
      Module_Name      : Name_String;
      Module_Name_Len  : Natural;
   end record;

   Null_Container : constant Model_Container := (
      Schema_Version   => 1,
      Element_Count    => 0,
      IR_Source_Hash   => (others => '0'),
      Hash_Length      => 0,
      Generation_Epoch => 0,
      DAL_Level        => DAL_C,
      Module_Name      => (others => ' '),
      Module_Name_Len  => 0
   );

   --  ============================================================
   --  Element Operations
   --  ============================================================

   --  Create a new element with the given kind and name
   function Create_Element (
      Kind   : Element_Kind;
      Name   : String;
      Parent : Element_ID := Null_Element_ID
   ) return Model_Element
     with Pre  => Name'Length > 0 and Name'Length <= Max_Name_Length,
          Post => Create_Element'Result.Kind = Kind
                  and Create_Element'Result.ID /= Null_Element_ID;

   --  Get the name of an element as a trimmed string
   function Get_Name (Element : Model_Element) return String
     with Pre  => Element.Name_Length > 0 and Element.Name_Length <= Max_Name_Length,
          Post => Get_Name'Result'Length = Element.Name_Length;

   --  Set the name of an element
   procedure Set_Name (
      Element : in out Model_Element;
      Name    : in     String
   ) with Pre => Name'Length > 0 and Name'Length <= Max_Name_Length,
          Post => Element.Name_Length = Name'Length;

   --  Check if an element is valid
   function Is_Valid (Element : Model_Element) return Boolean;

   --  Check if an element is a structural element
   function Is_Structural (Kind : Element_Kind) return Boolean;

   --  Check if an element is a behavioral element
   function Is_Behavioral (Kind : Element_Kind) return Boolean;

   --  ============================================================
   --  Container Operations
   --  ============================================================

   --  Initialize a model container
   function Create_Container return Model_Container
     with Post => Create_Container'Result.Element_Count = 0;

   --  Set the IR source hash
   procedure Set_IR_Hash (
      Container : in out Model_Container;
      Hash      : in     String
   ) with Pre => Hash'Length > 0 and Hash'Length <= Max_Hash_Length,
          Post => Container.Hash_Length = Hash'Length;

   --  Set the module name
   procedure Set_Module_Name (
      Container : in Out Model_Container;
      Name      : in     String
   ) with Pre => Name'Length > 0 and Name'Length <= Max_Name_Length,
          Post => Container.Module_Name_Len = Name'Length;

   --  ============================================================
   --  ID Generation
   --  ============================================================

   --  Generate a unique element ID
   function Generate_ID return Element_ID
     with Post => Generate_ID'Result /= Null_Element_ID;

   --  Reset the ID generator (for testing)
   procedure Reset_ID_Generator;

private

   --  Internal ID counter
   Next_ID : Element_ID := 1;

end Model_IR;
