--  STUNIR DO-331 Coverage Framework Specification
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0
--
--  This package provides coverage point management for DO-331 model coverage.

pragma SPARK_Mode (On);

with Model_IR; use Model_IR;

package Coverage is

   --  ============================================================
   --  Constants
   --  ============================================================

   Max_Coverage_Points : constant := 100_000;
   Max_Point_ID_Length : constant := 64;

   --  ============================================================
   --  Coverage Point Types per DO-331
   --  ============================================================

   type Coverage_Type is (
      State_Coverage,         --  Each state visited
      Transition_Coverage,    --  Each transition executed
      Decision_Coverage,      --  Each decision true/false
      Condition_Coverage,     --  Each condition true/false
      MCDC_Coverage,          --  MC/DC (DAL A only)
      Action_Coverage,        --  Each action executed
      Entry_Coverage,         --  Function entry
      Exit_Coverage,          --  Function exit
      Path_Coverage,          --  Execution paths
      Loop_Coverage           --  Loop iterations
   );

   --  ============================================================
   --  DAL Level Requirements
   --  ============================================================

   --  Coverage requirements per DAL level (DO-331 Table MB-A.5)
   function Is_Required (Level : DAL_Level; Cov_Type : Coverage_Type) return Boolean;

   --  Get description of DAL requirements
   function Get_DAL_Description (Level : DAL_Level) return String;

   --  ============================================================
   --  Coverage Point Record
   --  ============================================================

   type Coverage_Point is record
      Point_ID        : String (1 .. Max_Point_ID_Length);
      Point_ID_Len    : Natural;
      Point_Type      : Coverage_Type;
      Model_Element   : Element_ID;
      Model_Path      : String (1 .. 512);
      Path_Length     : Natural;
      DAL_Required    : DAL_Level_Set;
      Instrumented    : Boolean;
      Covered         : Boolean;  --  For analysis results
      Coverage_Count  : Natural;  --  Number of times covered
      Line_Number     : Natural;  --  Line in generated model
   end record;

   Null_Coverage_Point : constant Coverage_Point := (
      Point_ID       => (others => ' '),
      Point_ID_Len   => 0,
      Point_Type     => Action_Coverage,
      Model_Element  => Null_Element_ID,
      Model_Path     => (others => ' '),
      Path_Length    => 0,
      DAL_Required   => All_DAL_Levels,
      Instrumented   => False,
      Covered        => False,
      Coverage_Count => 0,
      Line_Number    => 0
   );

   --  ============================================================
   --  Coverage Point Array
   --  ============================================================

   type Coverage_Point_Array is array (Positive range <>) of Coverage_Point;

   --  ============================================================
   --  Coverage Points Container
   --  ============================================================

   type Coverage_Points is record
      Points      : Coverage_Point_Array (1 .. Max_Coverage_Points);
      Point_Count : Natural := 0;
   end record;

   --  ============================================================
   --  Container Operations
   --  ============================================================

   --  Initialize coverage container
   function Create_Coverage return Coverage_Points
     with Post => Create_Coverage'Result.Point_Count = 0;

   --  Add a coverage point
   procedure Add_Point (
      Container : in Out Coverage_Points;
      Kind      : in     Coverage_Type;
      Element   : in     Element_ID;
      Path      : in     String;
      Point_ID  : in     String
   ) with
      Pre  => Container.Point_Count < Max_Coverage_Points and
              Point_ID'Length > 0 and Point_ID'Length <= Max_Point_ID_Length,
      Post => Container.Point_Count = Container.Point_Count'Old + 1;

   --  Add point with specific DAL requirements
   procedure Add_Point_With_DAL (
      Container    : in Out Coverage_Points;
      Kind         : in     Coverage_Type;
      Element      : in     Element_ID;
      Path         : in     String;
      Point_ID     : in     String;
      DAL_Required : in     DAL_Level_Set
   ) with
      Pre  => Container.Point_Count < Max_Coverage_Points and
              Point_ID'Length > 0 and Point_ID'Length <= Max_Point_ID_Length,
      Post => Container.Point_Count = Container.Point_Count'Old + 1;

   --  ============================================================
   --  Query Operations
   --  ============================================================

   --  Get all points
   function Get_All_Points (
      Container : Coverage_Points
   ) return Coverage_Point_Array;

   --  Get points for specific DAL level
   function Get_Points_For_DAL (
      Container : Coverage_Points;
      Level     : DAL_Level
   ) return Coverage_Point_Array;

   --  Get points by type
   function Get_Points_By_Type (
      Container : Coverage_Points;
      Kind      : Coverage_Type
   ) return Coverage_Point_Array;

   --  Get point by ID
   function Get_Point_By_ID (
      Container : Coverage_Points;
      Point_ID  : String
   ) return Coverage_Point
     with Pre => Point_ID'Length > 0;

   --  Check if point exists
   function Point_Exists (
      Container : Coverage_Points;
      Point_ID  : String
   ) return Boolean
     with Pre => Point_ID'Length > 0;

   --  ============================================================
   --  Coverage Status
   --  ============================================================

   --  Mark point as instrumented
   procedure Mark_Instrumented (
      Container : in Out Coverage_Points;
      Point_ID  : in     String
   ) with Pre => Point_ID'Length > 0;

   --  Mark point as covered
   procedure Mark_Covered (
      Container : in Out Coverage_Points;
      Point_ID  : in     String
   ) with Pre => Point_ID'Length > 0;

   --  Increment coverage count
   procedure Increment_Coverage (
      Container : in Out Coverage_Points;
      Point_ID  : in     String
   ) with Pre => Point_ID'Length > 0;

   --  ============================================================
   --  Point ID Generation
   --  ============================================================

   --  Generate unique point ID
   function Generate_Point_ID (
      Kind    : Coverage_Type;
      Element : String;
      Index   : Natural
   ) return String
     with Pre  => Element'Length > 0,
          Post => Generate_Point_ID'Result'Length > 0 and
                  Generate_Point_ID'Result'Length <= Max_Point_ID_Length;

   --  Get point type prefix
   function Get_Type_Prefix (Kind : Coverage_Type) return String;

end Coverage;
