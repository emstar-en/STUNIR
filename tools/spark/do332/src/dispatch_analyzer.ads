--  STUNIR DO-332 Dispatch Analyzer Specification
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0
--
--  This package implements DO-332 OO.3 dynamic dispatch analysis,
--  including dispatch site resolution and target enumeration.

pragma SPARK_Mode (On);

with OOP_Types; use OOP_Types;
with OOP_Analysis; use OOP_Analysis;

package Dispatch_Analyzer is

   --  ============================================================
   --  Constants
   --  ============================================================

   Max_Dispatch_Sites   : constant := 10_000;
   Max_Targets_Per_Site : constant := 100;

   --  ============================================================
   --  Dispatch Site Record
   --  ============================================================

   type Dispatch_Site is record
      Site_ID         : Natural;
      Location_File   : Path_String;
      File_Length     : Natural;
      Location_Line   : Natural;
      Receiver_Type   : Class_ID;
      Method_Name     : Name_String;
      Method_Name_Len : Natural;
      Is_Super_Call   : Boolean;
      Is_Interface    : Boolean;
   end record;

   Null_Dispatch_Site : constant Dispatch_Site := (
      Site_ID         => 0,
      Location_File   => (others => ' '),
      File_Length     => 0,
      Location_Line   => 0,
      Receiver_Type   => Null_Class_ID,
      Method_Name     => (others => ' '),
      Method_Name_Len => 0,
      Is_Super_Call   => False,
      Is_Interface    => False
   );

   type Dispatch_Site_Array is array (Positive range <>) of Dispatch_Site;

   --  ============================================================
   --  Target Information
   --  ============================================================

   type Target_Info is record
      Target_Class  : Class_ID;
      Target_Method : Method_ID;
      Is_Final      : Boolean;
      Is_Abstract   : Boolean;
   end record;

   Null_Target : constant Target_Info := (
      Target_Class  => Null_Class_ID,
      Target_Method => Null_Method_ID,
      Is_Final      => False,
      Is_Abstract   => False
   );

   type Target_Array is array (Positive range <>) of Target_Info;

   --  ============================================================
   --  Dispatch Analysis Result
   --  ============================================================

   type Site_Analysis is record
      Site_ID            : Natural;
      Target_Count       : Natural;
      Is_Bounded         : Boolean;
      Is_Devirtualizable : Boolean;
      Proven_Single      : Boolean;
      Has_Abstract       : Boolean;
   end record;

   Null_Site_Analysis : constant Site_Analysis := (
      Site_ID            => 0,
      Target_Count       => 0,
      Is_Bounded         => True,
      Is_Devirtualizable => False,
      Proven_Single      => False,
      Has_Abstract       => False
   );

   type Site_Analysis_Array is array (Positive range <>) of Site_Analysis;

   --  ============================================================
   --  Dispatch Summary
   --  ============================================================

   type Dispatch_Summary is record
      Total_Sites        : Natural;
      Bounded_Sites      : Natural;
      Unbounded_Sites    : Natural;
      Devirtualized      : Natural;
      Max_Targets        : Natural;
      Sites_With_Abstract: Natural;
   end record;

   Null_Dispatch_Summary : constant Dispatch_Summary := (
      Total_Sites        => 0,
      Bounded_Sites      => 0,
      Unbounded_Sites    => 0,
      Devirtualized      => 0,
      Max_Targets        => 0,
      Sites_With_Abstract => 0
   );

   --  ============================================================
   --  Core Analysis Functions
   --  ============================================================

   --  Resolve all possible targets for a dispatch site
   function Resolve_Targets (
      Site    : Dispatch_Site;
      Classes : Class_Array;
      Methods : Method_Array;
      Links   : Inheritance_Array
   ) return Target_Array
     with Post => Resolve_Targets'Result'Length >= 0;

   --  Analyze a single dispatch site
   function Analyze_Site (
      Site    : Dispatch_Site;
      Classes : Class_Array;
      Methods : Method_Array;
      Links   : Inheritance_Array
   ) return Site_Analysis;

   --  Check if a dispatch can be devirtualized
   function Can_Devirtualize (
      Site    : Dispatch_Site;
      Classes : Class_Array;
      Methods : Method_Array;
      Links   : Inheritance_Array
   ) return Boolean;

   --  Analyze all dispatch sites
   procedure Analyze_All_Dispatch (
      Sites    : in     Dispatch_Site_Array;
      Classes  : in     Class_Array;
      Methods  : in     Method_Array;
      Links    : in     Inheritance_Array;
      Results  :    out Site_Analysis_Array;
      Summary  :    out Dispatch_Summary;
      Success  :    out Boolean
   ) with Pre => Results'Length >= Sites'Length;

   --  ============================================================
   --  Utility Functions
   --  ============================================================

   --  Find method in a class by name
   function Find_Method_By_Name (
      Class_ID    : OOP_Types.Class_ID;
      Method_Name : String;
      Methods     : Method_Array
   ) return Method_ID;

   --  Get all concrete implementations of a virtual method
   function Get_Implementations (
      Virtual_Method : Method_ID;
      Methods        : Method_Array
   ) return Method_ID_Array;

   --  Count targets for a dispatch site
   function Count_Targets (
      Site    : Dispatch_Site;
      Classes : Class_Array;
      Methods : Method_Array;
      Links   : Inheritance_Array
   ) return Natural;

end Dispatch_Analyzer;
