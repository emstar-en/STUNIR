--  STUNIR DO-332 Integration Types Specification
--  Object-Oriented Technology Data Types
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0
--
--  This package defines types for DO-332 OOP verification:
--  - Class hierarchy analysis
--  - Inheritance verification
--  - Polymorphism safety checking
--  - Coupling metrics

pragma SPARK_Mode (On);

package DO332_Types is

   --  ============================================================
   --  Constants
   --  ============================================================

   Max_Class_Name_Length  : constant := 128;
   Max_Class_Count        : constant := 512;
   Max_Methods_Per_Class  : constant := 64;
   Max_Inheritance_Depth  : constant := 16;
   Max_Polymorphic_Calls  : constant := 256;

   --  ============================================================
   --  String Types
   --  ============================================================

   subtype Class_Name_Index is Positive range 1 .. Max_Class_Name_Length;
   subtype Class_Name_Length is Natural range 0 .. Max_Class_Name_Length;
   subtype Class_Name_String is String (Class_Name_Index);

   --  ============================================================
   --  OOP Analysis Types
   --  ============================================================

   type Inheritance_Kind is (
      Single_Inheritance,
      Multiple_Inheritance,
      Interface_Implementation,
      No_Inheritance
   );

   type Visibility_Kind is (
      Public_Visibility,
      Protected_Visibility,
      Private_Visibility
   );

   type Method_Kind is (
      Virtual_Method,
      Non_Virtual_Method,
      Abstract_Method,
      Override_Method,
      Static_Method
   );

   --  ============================================================
   --  Class Method
   --  ============================================================

   type Class_Method is record
      Name       : Class_Name_String;
      Name_Len   : Class_Name_Length;
      Kind       : Method_Kind;
      Visibility : Visibility_Kind;
      Is_Valid   : Boolean;
   end record;

   Null_Class_Method : constant Class_Method := (
      Name       => (others => ' '),
      Name_Len   => 0,
      Kind       => Non_Virtual_Method,
      Visibility => Public_Visibility,
      Is_Valid   => False
   );

   subtype Method_Index is Positive range 1 .. Max_Methods_Per_Class;
   subtype Method_Count is Natural range 0 .. Max_Methods_Per_Class;
   type Method_Array is array (Method_Index) of Class_Method;

   --  ============================================================
   --  Class Entry
   --  ============================================================

   type Class_Entry is record
      Name            : Class_Name_String;
      Name_Len        : Class_Name_Length;
      Parent_Name     : Class_Name_String;
      Parent_Len      : Class_Name_Length;
      Inheritance     : Inheritance_Kind;
      Depth           : Natural;
      Methods         : Method_Array;
      Method_Cnt      : Method_Count;
      Is_Abstract     : Boolean;
      Is_Final        : Boolean;
      Is_Valid        : Boolean;
   end record;

   Null_Class_Entry : constant Class_Entry := (
      Name         => (others => ' '),
      Name_Len     => 0,
      Parent_Name  => (others => ' '),
      Parent_Len   => 0,
      Inheritance  => No_Inheritance,
      Depth        => 0,
      Methods      => (others => Null_Class_Method),
      Method_Cnt   => 0,
      Is_Abstract  => False,
      Is_Final     => False,
      Is_Valid     => False
   );

   subtype Class_Index is Positive range 1 .. Max_Class_Count;
   subtype Class_Count is Natural range 0 .. Max_Class_Count;
   type Class_Array is array (Class_Index) of Class_Entry;

   --  ============================================================
   --  Polymorphic Call
   --  ============================================================

   type Polymorphic_Call is record
      Caller_Class : Class_Name_String;
      Caller_Len   : Class_Name_Length;
      Target_Method: Class_Name_String;
      Target_Len   : Class_Name_Length;
      Is_Safe      : Boolean;
      Is_Verified  : Boolean;
   end record;

   Null_Polymorphic_Call : constant Polymorphic_Call := (
      Caller_Class => (others => ' '),
      Caller_Len   => 0,
      Target_Method=> (others => ' '),
      Target_Len   => 0,
      Is_Safe      => False,
      Is_Verified  => False
   );

   subtype Poly_Call_Index is Positive range 1 .. Max_Polymorphic_Calls;
   subtype Poly_Call_Count is Natural range 0 .. Max_Polymorphic_Calls;
   type Poly_Call_Array is array (Poly_Call_Index) of Polymorphic_Call;

   --  ============================================================
   --  Coupling Metrics
   --  ============================================================

   type Coupling_Metrics is record
      CBO   : Natural;  --  Coupling Between Objects
      RFC   : Natural;  --  Response For Class
      WMC   : Natural;  --  Weighted Methods per Class
      DIT   : Natural;  --  Depth of Inheritance Tree
      NOC   : Natural;  --  Number of Children
      LCOM  : Natural;  --  Lack of Cohesion in Methods
   end record;

   Null_Coupling_Metrics : constant Coupling_Metrics := (
      CBO  => 0,
      RFC  => 0,
      WMC  => 0,
      DIT  => 0,
      NOC  => 0,
      LCOM => 0
   );

   --  ============================================================
   --  DO-332 Result
   --  ============================================================

   type DO332_Result is record
      --  Class analysis
      Classes          : Class_Array;
      Class_Total      : Class_Count;

      --  Polymorphic calls
      Poly_Calls       : Poly_Call_Array;
      Poly_Total       : Poly_Call_Count;

      --  Metrics
      Metrics          : Coupling_Metrics;
      Max_Depth        : Natural;

      --  Verification status
      Inheritance_OK   : Boolean;
      Polymorphism_OK  : Boolean;
      Coupling_OK      : Boolean;
      Success          : Boolean;
   end record;

   Null_DO332_Result : constant DO332_Result := (
      Classes        => (others => Null_Class_Entry),
      Class_Total    => 0,
      Poly_Calls     => (others => Null_Polymorphic_Call),
      Poly_Total     => 0,
      Metrics        => Null_Coupling_Metrics,
      Max_Depth      => 0,
      Inheritance_OK => False,
      Polymorphism_OK=> False,
      Coupling_OK    => False,
      Success        => False
   );

   --  ============================================================
   --  Status
   --  ============================================================

   type DO332_Status is (
      Success,
      Class_Not_Found,
      Invalid_Hierarchy,
      Depth_Exceeded,
      Unsafe_Polymorphism,
      High_Coupling,
      Analysis_Failed,
      IO_Error
   );

   --  ============================================================
   --  Utility Functions
   --  ============================================================

   function Status_Message (Status : DO332_Status) return String;
   function Inheritance_Name (Kind : Inheritance_Kind) return String;
   function Method_Kind_Name (Kind : Method_Kind) return String;

end DO332_Types;
