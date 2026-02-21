--  STUNIR DO-332 OOP Types Specification
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0
--
--  This package provides core type definitions for representing
--  object-oriented constructs for DO-332 verification analysis.

pragma SPARK_Mode (On);

package OOP_Types is

   --  ============================================================
   --  Constants and Limits
   --  ============================================================

   Max_Classes        : constant := 5_000;
   Max_Methods        : constant := 50_000;
   Max_Fields         : constant := 10_000;
   Max_Inheritance    : constant := 20;
   Max_Name_Length    : constant := 256;
   Max_Path_Length    : constant := 1024;
   Max_Parameters     : constant := 50;
   Max_Vtable_Size    : constant := 1_000;
   Max_Parents        : constant := 10;
   Max_Children       : constant := 500;
   Max_Dependencies   : constant := 1_000;
   Max_Hash_Length    : constant := 64;

   --  ============================================================
   --  Basic Identifier Types
   --  ============================================================

   --  Class identifier
   type Class_ID is mod 2**32;
   Null_Class_ID : constant Class_ID := 0;

   --  Method identifier
   type Method_ID is mod 2**32;
   Null_Method_ID : constant Method_ID := 0;

   --  Field identifier
   type Field_ID is mod 2**32;
   Null_Field_ID : constant Field_ID := 0;

   --  Generic element count
   subtype Element_Count is Natural range 0 .. Max_Classes;
   subtype Method_Count is Natural range 0 .. Max_Methods;

   --  ============================================================
   --  Bounded String Types
   --  ============================================================

   subtype Name_String is String (1 .. Max_Name_Length);
   subtype Path_String is String (1 .. Max_Path_Length);
   subtype Hash_String is String (1 .. Max_Hash_Length);

   --  ============================================================
   --  Enumeration Types
   --  ============================================================

   --  Visibility levels
   type Visibility is (V_Public, V_Protected, V_Private);

   --  Method kinds
   type Method_Kind is (
      Regular_Method,
      Virtual_Method,
      Abstract_Method,
      Final_Method,
      Static_Method,
      Constructor,
      Destructor,
      Pure_Virtual
   );

   --  Class kinds
   type Class_Kind is (
      Regular_Class,
      Abstract_Class,
      Final_Class,
      Interface_Type,
      Mixin_Class,
      Trait_Type
   );

   --  Parameter direction
   type Parameter_Direction is (Dir_In, Dir_Out, Dir_InOut);

   --  Dependency kinds for coupling analysis
   type Dependency_Kind is (
      Inheritance_Dep,     --  Extends/implements
      Composition_Dep,     --  Has member of type
      Association_Dep,     --  Uses reference to type
      Call_Dep,            --  Calls method on type
      Parameter_Dep,       --  Parameter of type
      Return_Dep,          --  Returns type
      Throws_Dep,          --  Throws exception of type
      Template_Dep         --  Template/generic parameter
   );

   --  DO-332 Objectives
   type DO332_Objective is (
      OO_1,  --  Inheritance analysis
      OO_2,  --  Polymorphism verification
      OO_3,  --  Dynamic dispatch analysis
      OO_4,  --  Object coupling analysis
      OO_5,  --  Exception handling in OOP
      OO_6   --  Constructor/destructor verification
   );

   --  DAL levels
   type DAL_Level is (DAL_A, DAL_B, DAL_C, DAL_D, DAL_E);

   --  Test categories
   type Test_Category is (
      Inheritance_Test,
      Override_Test,
      Polymorphism_Test,
      Dispatch_Test,
      Coupling_Test,
      Lifecycle_Test
   );

   --  ============================================================
   --  Class Information Record
   --  ============================================================

   type Class_Info is record
      ID                : Class_ID;
      Name              : Name_String;
      Name_Length       : Natural;
      Kind              : Class_Kind;
      Parent_Count      : Natural;
      Method_Count      : Natural;
      Field_Count       : Natural;
      Is_Root           : Boolean;
      Is_Abstract       : Boolean;
      Is_Final          : Boolean;
      Inheritance_Depth : Natural;
      Line_Number       : Natural;
      File_Path         : Path_String;
      File_Path_Length  : Natural;
   end record;

   --  Null/default class
   Null_Class : constant Class_Info := (
      ID                => Null_Class_ID,
      Name              => (others => ' '),
      Name_Length       => 0,
      Kind              => Regular_Class,
      Parent_Count      => 0,
      Method_Count      => 0,
      Field_Count       => 0,
      Is_Root           => True,
      Is_Abstract       => False,
      Is_Final          => False,
      Inheritance_Depth => 0,
      Line_Number       => 0,
      File_Path         => (others => ' '),
      File_Path_Length  => 0
   );

   --  ============================================================
   --  Method Information Record
   --  ============================================================

   type Method_Info is record
      ID              : Method_ID;
      Name            : Name_String;
      Name_Length     : Natural;
      Owning_Class    : Class_ID;
      Kind            : Method_Kind;
      Visibility      : OOP_Types.Visibility;
      Parameter_Count : Natural;
      Has_Override    : Boolean;
      Override_Of     : Method_ID;
      Is_Covariant    : Boolean;
      Is_Contravariant: Boolean;
      Line_Number     : Natural;
   end record;

   --  Null/default method
   Null_Method : constant Method_Info := (
      ID               => Null_Method_ID,
      Name             => (others => ' '),
      Name_Length      => 0,
      Owning_Class     => Null_Class_ID,
      Kind             => Regular_Method,
      Visibility       => V_Public,
      Parameter_Count  => 0,
      Has_Override     => False,
      Override_Of      => Null_Method_ID,
      Is_Covariant     => False,
      Is_Contravariant => False,
      Line_Number      => 0
   );

   --  ============================================================
   --  Field Information Record
   --  ============================================================

   type Field_Info is record
      ID              : Field_ID;
      Name            : Name_String;
      Name_Length     : Natural;
      Owning_Class    : Class_ID;
      Type_Name       : Name_String;
      Type_Length     : Natural;
      Visibility      : OOP_Types.Visibility;
      Is_Static       : Boolean;
      Is_Const        : Boolean;
      Is_Reference    : Boolean;
   end record;

   --  ============================================================
   --  Inheritance Link
   --  ============================================================

   type Inheritance_Link is record
      Child_ID      : Class_ID;
      Parent_ID     : Class_ID;
      Is_Virtual    : Boolean;  --  Virtual inheritance (C++)
      Is_Interface  : Boolean;  --  Interface implementation
      Link_Index    : Positive; --  Order in parent list
   end record;

   Null_Inheritance_Link : constant Inheritance_Link := (
      Child_ID     => Null_Class_ID,
      Parent_ID    => Null_Class_ID,
      Is_Virtual   => False,
      Is_Interface => False,
      Link_Index   => 1
   );

   --  ============================================================
   --  Dependency Record
   --  ============================================================

   type Dependency is record
      Source_Class  : Class_ID;
      Target_Class  : Class_ID;
      Kind          : Dependency_Kind;
      Count         : Natural;  --  Number of occurrences
   end record;

   Null_Dependency : constant Dependency := (
      Source_Class => Null_Class_ID,
      Target_Class => Null_Class_ID,
      Kind         => Association_Dep,
      Count        => 0
   );

   --  ============================================================
   --  VTable Entry
   --  ============================================================

   type VTable_Entry is record
      Slot_Index      : Natural;
      Method_ID       : OOP_Types.Method_ID;
      Declaring_Class : Class_ID;
      Impl_Class      : Class_ID;
      Is_Abstract     : Boolean;
      Is_Final        : Boolean;
   end record;

   Null_VTable_Entry : constant VTable_Entry := (
      Slot_Index      => 0,
      Method_ID       => Null_Method_ID,
      Declaring_Class => Null_Class_ID,
      Impl_Class      => Null_Class_ID,
      Is_Abstract     => False,
      Is_Final        => False
   );

   --  ============================================================
   --  Array Types
   --  ============================================================

   type Class_Array is array (Positive range <>) of Class_Info;
   type Method_Array is array (Positive range <>) of Method_Info;
   type Field_Array is array (Positive range <>) of Field_Info;
   type Inheritance_Array is array (Positive range <>) of Inheritance_Link;
   type Dependency_Array is array (Positive range <>) of Dependency;
   type VTable_Array is array (Positive range <>) of VTable_Entry;
   type Class_ID_Array is array (Positive range <>) of Class_ID;
   type Method_ID_Array is array (Positive range <>) of Method_ID;

   --  ============================================================
   --  DAL Requirements
   --  ============================================================

   type DAL_Requirements is record
      Requires_Inheritance_Analysis    : Boolean;
      Requires_Diamond_Detection       : Boolean;
      Requires_Polymorphism_Analysis   : Boolean;
      Requires_LSP_Checking            : Boolean;
      Requires_Dispatch_Analysis       : Boolean;
      Requires_Dispatch_Timing         : Boolean;
      Requires_Coupling_Analysis       : Boolean;
      Requires_Lifecycle_Analysis      : Boolean;
   end record;

   function Get_DAL_Requirements (Level : DAL_Level) return DAL_Requirements;

   --  ============================================================
   --  Utility Functions
   --  ============================================================

   --  Check if a class ID is valid
   function Is_Valid_Class_ID (ID : Class_ID) return Boolean is
      (ID /= Null_Class_ID);

   --  Check if a method ID is valid
   function Is_Valid_Method_ID (ID : Method_ID) return Boolean is
      (ID /= Null_Method_ID);

   --  Check if a method is virtual (can be overridden)
   function Is_Virtual (Kind : Method_Kind) return Boolean is
      (Kind = Virtual_Method or Kind = Abstract_Method or Kind = Pure_Virtual);

   --  Check if a class can be instantiated
   function Is_Instantiable (Kind : Class_Kind) return Boolean is
      (Kind = Regular_Class or Kind = Final_Class);

   --  Check if a class can be inherited from
   function Is_Inheritable (Kind : Class_Kind) return Boolean is
      (Kind /= Final_Class);

private

   --  ID generation counter
   Next_Class_ID  : Class_ID := 1;
   Next_Method_ID : Method_ID := 1;
   Next_Field_ID  : Field_ID := 1;

end OOP_Types;
