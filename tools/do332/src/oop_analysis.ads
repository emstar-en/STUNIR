--  STUNIR DO-332 OOP Analysis Framework Specification
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0
--
--  This package provides the main OOP analysis framework for DO-332
--  verification, orchestrating inheritance, polymorphism, dispatch,
--  and coupling analyses.

pragma SPARK_Mode (On);

with OOP_Types; use OOP_Types;

package OOP_Analysis is

   --  ============================================================
   --  Constants
   --  ============================================================

   Max_Analysis_Classes  : constant := 5_000;
   Max_Analysis_Methods  : constant := 50_000;
   Max_Analysis_Links    : constant := 10_000;
   Max_Dispatch_Sites    : constant := 10_000;
   Max_Report_Entries    : constant := 1_000;

   --  ============================================================
   --  Analysis Configuration
   --  ============================================================

   type Analysis_Config is record
      DAL_Level              : OOP_Types.DAL_Level;
      Enable_Inheritance     : Boolean;
      Enable_Polymorphism    : Boolean;
      Enable_Dispatch        : Boolean;
      Enable_Coupling        : Boolean;
      Enable_Test_Gen        : Boolean;
      Max_Inheritance_Depth  : Positive;
      CBO_Threshold          : Natural;
      RFC_Threshold          : Natural;
      Generate_Evidence      : Boolean;
   end record;

   Default_Config : constant Analysis_Config := (
      DAL_Level              => DAL_C,
      Enable_Inheritance     => True,
      Enable_Polymorphism    => True,
      Enable_Dispatch        => True,
      Enable_Coupling        => True,
      Enable_Test_Gen        => True,
      Max_Inheritance_Depth  => 10,
      CBO_Threshold          => 10,
      RFC_Threshold          => 50,
      Generate_Evidence      => True
   );

   --  ============================================================
   --  Class Hierarchy Container
   --  ============================================================

   type Class_Hierarchy is record
      Class_Count      : Element_Count;
      Method_Count     : Natural;
      Link_Count       : Natural;
      Schema_Version   : Natural;
      IR_Source_Hash   : Hash_String;
      Hash_Length      : Natural;
      Analysis_Epoch   : Natural;
      Module_Name      : Name_String;
      Module_Name_Len  : Natural;
   end record;

   Null_Hierarchy : constant Class_Hierarchy := (
      Class_Count     => 0,
      Method_Count    => 0,
      Link_Count      => 0,
      Schema_Version  => 1,
      IR_Source_Hash  => (others => '0'),
      Hash_Length     => 0,
      Analysis_Epoch  => 0,
      Module_Name     => (others => ' '),
      Module_Name_Len => 0
   );

   --  ============================================================
   --  Analysis Result Types
   --  ============================================================

   --  Overall analysis status
   type Analysis_Status is (
      Analysis_OK,
      Analysis_Warning,
      Analysis_Error,
      Analysis_Not_Run
   );

   --  Inheritance analysis result
   type Inheritance_Result is record
      Class_ID              : Class_ID;
      Depth                 : Natural;
      Has_Diamond           : Boolean;
      Has_Circular          : Boolean;
      Multiple_Count        : Natural;
      Override_Count        : Natural;
      Virtual_Count         : Natural;
      Abstract_Count        : Natural;
      All_Overrides_Valid   : Boolean;
      Status                : Analysis_Status;
   end record;

   Null_Inheritance_Result : constant Inheritance_Result := (
      Class_ID            => Null_Class_ID,
      Depth               => 0,
      Has_Diamond         => False,
      Has_Circular        => False,
      Multiple_Count      => 0,
      Override_Count      => 0,
      Virtual_Count       => 0,
      Abstract_Count      => 0,
      All_Overrides_Valid => True,
      Status              => Analysis_Not_Run
   );

   --  Polymorphism verification result
   type Polymorphism_Result is record
      Class_ID              : Class_ID;
      Virtual_Methods       : Natural;
      Polymorphic_Calls     : Natural;
      LSP_Violations        : Natural;
      Covariance_Issues     : Natural;
      Contravariance_Issues : Natural;
      Type_Safe             : Boolean;
      All_Bounded           : Boolean;
      Status                : Analysis_Status;
   end record;

   Null_Polymorphism_Result : constant Polymorphism_Result := (
      Class_ID              => Null_Class_ID,
      Virtual_Methods       => 0,
      Polymorphic_Calls     => 0,
      LSP_Violations        => 0,
      Covariance_Issues     => 0,
      Contravariance_Issues => 0,
      Type_Safe             => True,
      All_Bounded           => True,
      Status                => Analysis_Not_Run
   );

   --  Dispatch analysis result
   type Dispatch_Result is record
      Site_ID               : Natural;
      Location_Line         : Natural;
      Receiver_Type         : Class_ID;
      Method_Name           : Name_String;
      Method_Name_Len       : Natural;
      Target_Count          : Natural;
      Is_Bounded            : Boolean;
      Is_Devirtualizable    : Boolean;
      Worst_Case_Target     : Class_ID;
      Status                : Analysis_Status;
   end record;

   Null_Dispatch_Result : constant Dispatch_Result := (
      Site_ID            => 0,
      Location_Line      => 0,
      Receiver_Type      => Null_Class_ID,
      Method_Name        => (others => ' '),
      Method_Name_Len    => 0,
      Target_Count       => 0,
      Is_Bounded         => True,
      Is_Devirtualizable => False,
      Worst_Case_Target  => Null_Class_ID,
      Status             => Analysis_Not_Run
   );

   --  Coupling metrics result
   type Coupling_Result is record
      Class_ID              : Class_ID;
      CBO                   : Natural;  --  Coupling Between Objects
      RFC                   : Natural;  --  Response For Class
      LCOM                  : Natural;  --  Lack of Cohesion in Methods
      DIT                   : Natural;  --  Depth of Inheritance Tree
      NOC                   : Natural;  --  Number of Children
      WMC                   : Natural;  --  Weighted Methods per Class
      Afferent              : Natural;  --  Incoming dependencies
      Efferent              : Natural;  --  Outgoing dependencies
      Instability           : Natural;  --  Ce / (Ca + Ce) * 100
      Has_Circular_Dep      : Boolean;
      Exceeds_Thresholds    : Boolean;
      Status                : Analysis_Status;
   end record;

   Null_Coupling_Result : constant Coupling_Result := (
      Class_ID           => Null_Class_ID,
      CBO                => 0,
      RFC                => 0,
      LCOM               => 0,
      DIT                => 0,
      NOC                => 0,
      WMC                => 0,
      Afferent           => 0,
      Efferent           => 0,
      Instability        => 0,
      Has_Circular_Dep   => False,
      Exceeds_Thresholds => False,
      Status             => Analysis_Not_Run
   );

   --  ============================================================
   --  Result Arrays
   --  ============================================================

   type Inheritance_Result_Array is array (Positive range <>) of Inheritance_Result;
   type Polymorphism_Result_Array is array (Positive range <>) of Polymorphism_Result;
   type Dispatch_Result_Array is array (Positive range <>) of Dispatch_Result;
   type Coupling_Result_Array is array (Positive range <>) of Coupling_Result;

   --  ============================================================
   --  Global Analysis Summary
   --  ============================================================

   type Analysis_Summary is record
      Total_Classes         : Natural;
      Total_Methods         : Natural;
      Total_Inheritance     : Natural;
      Max_Depth             : Natural;
      Diamond_Patterns      : Natural;
      Circular_Inheritance  : Natural;
      Virtual_Methods       : Natural;
      Polymorphic_Calls     : Natural;
      LSP_Violations        : Natural;
      Dispatch_Sites        : Natural;
      Bounded_Sites         : Natural;
      Devirtualized_Calls   : Natural;
      Avg_CBO               : Natural;
      Max_CBO               : Natural;
      Circular_Dependencies : Natural;
      Overall_Status        : Analysis_Status;
   end record;

   Null_Summary : constant Analysis_Summary := (
      Total_Classes        => 0,
      Total_Methods        => 0,
      Total_Inheritance    => 0,
      Max_Depth            => 0,
      Diamond_Patterns     => 0,
      Circular_Inheritance => 0,
      Virtual_Methods      => 0,
      Polymorphic_Calls    => 0,
      LSP_Violations       => 0,
      Dispatch_Sites       => 0,
      Bounded_Sites        => 0,
      Devirtualized_Calls  => 0,
      Avg_CBO              => 0,
      Max_CBO              => 0,
      Circular_Dependencies => 0,
      Overall_Status       => Analysis_Not_Run
   );

   --  ============================================================
   --  Core Analysis Functions
   --  ============================================================

   --  Initialize a class hierarchy from arrays
   function Create_Hierarchy (
      Classes : Class_Array;
      Methods : Method_Array;
      Links   : Inheritance_Array
   ) return Class_Hierarchy
     with Pre  => Classes'Length <= Max_Analysis_Classes
                  and Methods'Length <= Max_Analysis_Methods
                  and Links'Length <= Max_Analysis_Links,
          Post => Create_Hierarchy'Result.Class_Count = Classes'Length
                  and Create_Hierarchy'Result.Method_Count = Methods'Length;

   --  Run complete DO-332 analysis on a hierarchy
   procedure Run_Analysis (
      Hierarchy     : in     Class_Hierarchy;
      Classes       : in     Class_Array;
      Methods       : in     Method_Array;
      Links         : in     Inheritance_Array;
      Config        : in     Analysis_Config;
      Summary       : out    Analysis_Summary;
      Success       : out    Boolean
   ) with Pre => Classes'Length <= Max_Analysis_Classes
                 and Hierarchy.Class_Count = Classes'Length;

   --  Check if analysis should run based on DAL
   function Should_Run_Inheritance (Config : Analysis_Config) return Boolean is
      (Config.Enable_Inheritance and 
       Get_DAL_Requirements (Config.DAL_Level).Requires_Inheritance_Analysis);

   function Should_Run_Polymorphism (Config : Analysis_Config) return Boolean is
      (Config.Enable_Polymorphism and 
       Get_DAL_Requirements (Config.DAL_Level).Requires_Polymorphism_Analysis);

   function Should_Run_Dispatch (Config : Analysis_Config) return Boolean is
      (Config.Enable_Dispatch and 
       Get_DAL_Requirements (Config.DAL_Level).Requires_Dispatch_Analysis);

   function Should_Run_Coupling (Config : Analysis_Config) return Boolean is
      (Config.Enable_Coupling and 
       Get_DAL_Requirements (Config.DAL_Level).Requires_Coupling_Analysis);

   --  ============================================================
   --  Helper Functions
   --  ============================================================

   --  Find a class by ID in an array
   function Find_Class (
      Classes : Class_Array;
      ID      : Class_ID
   ) return Natural
     with Post => Find_Class'Result = 0 
                  or (Find_Class'Result in Classes'Range
                      and Classes (Find_Class'Result).ID = ID);

   --  Find a method by ID in an array
   function Find_Method (
      Methods : Method_Array;
      ID      : Method_ID
   ) return Natural
     with Post => Find_Method'Result = 0 
                  or (Find_Method'Result in Methods'Range
                      and Methods (Find_Method'Result).ID = ID);

   --  Get all parents of a class
   function Get_Parents (
      Class_ID : OOP_Types.Class_ID;
      Links    : Inheritance_Array
   ) return Class_ID_Array;

   --  Get all children of a class
   function Get_Children (
      Class_ID : OOP_Types.Class_ID;
      Links    : Inheritance_Array
   ) return Class_ID_Array;

   --  Get all methods of a class
   function Get_Class_Methods (
      Class_ID : OOP_Types.Class_ID;
      Methods  : Method_Array
   ) return Method_ID_Array;

end OOP_Analysis;
