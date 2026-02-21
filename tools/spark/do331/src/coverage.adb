--  STUNIR DO-331 Coverage Framework Implementation
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

with Transformer_Utils; use Transformer_Utils;

package body Coverage is

   --  Point ID counter for uniqueness
   Next_Point_Index : Natural := 1;

   --  ============================================================
   --  DAL Requirements (per DO-331 Table MB-A.5)
   --  ============================================================

   function Is_Required (Level : DAL_Level; Cov_Type : Coverage_Type) return Boolean is
   begin
      case Level is
         when DAL_A =>
            --  DAL A requires all coverage types
            return True;
            
         when DAL_B =>
            --  DAL B: Decision + all basic coverage, no MC/DC
            return Cov_Type /= MCDC_Coverage;
            
         when DAL_C =>
            --  DAL C: Statement-level coverage only
            return Cov_Type in State_Coverage | Transition_Coverage |
                               Action_Coverage | Entry_Coverage | Exit_Coverage;
            
         when DAL_D | DAL_E =>
            --  DAL D and E have minimal coverage requirements
            return Cov_Type in Entry_Coverage | Exit_Coverage;
      end case;
   end Is_Required;

   function Get_DAL_Description (Level : DAL_Level) return String is
   begin
      case Level is
         when DAL_A =>
            return "Design Assurance Level A: Catastrophic failure condition. " &
                   "Requires MC/DC, Decision, Statement, State, and Transition coverage.";
         when DAL_B =>
            return "Design Assurance Level B: Hazardous failure condition. " &
                   "Requires Decision, Statement, State, and Transition coverage.";
         when DAL_C =>
            return "Design Assurance Level C: Major failure condition. " &
                   "Requires Statement, State, and Transition coverage.";
         when DAL_D =>
            return "Design Assurance Level D: Minor failure condition. " &
                   "Minimal coverage requirements.";
         when DAL_E =>
            return "Design Assurance Level E: No safety effect. " &
                   "No specific coverage requirements.";
      end case;
   end Get_DAL_Description;

   --  ============================================================
   --  Container Operations
   --  ============================================================

   function Create_Coverage return Coverage_Points is
      Result : Coverage_Points;
   begin
      Result.Point_Count := 0;
      for I in Result.Points'Range loop
         Result.Points (I) := Null_Coverage_Point;
      end loop;
      return Result;
   end Create_Coverage;

   procedure Add_Point (
      Container : in Out Coverage_Points;
      Kind      : in     Coverage_Type;
      Element   : in     Element_ID;
      Path      : in     String;
      Point_ID  : in     String
   ) is
      --  Determine default DAL requirements based on coverage type
      DAL_Req : DAL_Level_Set := (others => False);
   begin
      --  Set DAL requirements based on coverage type
      for Level in DAL_Level loop
         DAL_Req (Level) := Is_Required (Level, Kind);
      end loop;
      
      Add_Point_With_DAL (Container, Kind, Element, Path, Point_ID, DAL_Req);
   end Add_Point;

   procedure Add_Point_With_DAL (
      Container    : in Out Coverage_Points;
      Kind         : in     Coverage_Type;
      Element      : in     Element_ID;
      Path         : in     String;
      Point_ID     : in     String;
      DAL_Required : in     DAL_Level_Set
   ) is
      Point    : Coverage_Point := Null_Coverage_Point;
      Path_Len : Natural;
   begin
      --  Copy point ID
      Point.Point_ID := (others => ' ');
      Point.Point_ID (1 .. Point_ID'Length) := Point_ID;
      Point.Point_ID_Len := Point_ID'Length;
      
      --  Set other fields
      Point.Point_Type := Kind;
      Point.Model_Element := Element;
      Point.DAL_Required := DAL_Required;
      Point.Instrumented := True;  --  Automatically instrumented when added
      Point.Covered := False;
      Point.Coverage_Count := 0;
      
      --  Copy path
      Point.Model_Path := (others => ' ');
      Path_Len := Natural'Min (Path'Length, 512);
      if Path_Len > 0 then
         Point.Model_Path (1 .. Path_Len) := Path (Path'First .. Path'First + Path_Len - 1);
      end if;
      Point.Path_Length := Path_Len;
      
      --  Add to container
      Container.Point_Count := Container.Point_Count + 1;
      Container.Points (Container.Point_Count) := Point;
   end Add_Point_With_DAL;

   --  ============================================================
   --  Query Operations
   --  ============================================================

   function Get_All_Points (
      Container : Coverage_Points
   ) return Coverage_Point_Array is
   begin
      if Container.Point_Count = 0 then
         return (1 .. 0 => Null_Coverage_Point);
      else
         return Container.Points (1 .. Container.Point_Count);
      end if;
   end Get_All_Points;

   function Get_Points_For_DAL (
      Container : Coverage_Points;
      Level     : DAL_Level
   ) return Coverage_Point_Array is
      Count : Natural := 0;
   begin
      --  Count matching points
      for I in 1 .. Container.Point_Count loop
         if Container.Points (I).DAL_Required (Level) then
            Count := Count + 1;
         end if;
      end loop;
      
      --  Build result
      declare
         Result : Coverage_Point_Array (1 .. Count);
         J      : Natural := 0;
      begin
         for I in 1 .. Container.Point_Count loop
            if Container.Points (I).DAL_Required (Level) then
               J := J + 1;
               Result (J) := Container.Points (I);
            end if;
         end loop;
         return Result;
      end;
   end Get_Points_For_DAL;

   function Get_Points_By_Type (
      Container : Coverage_Points;
      Kind      : Coverage_Type
   ) return Coverage_Point_Array is
      Count : Natural := 0;
   begin
      for I in 1 .. Container.Point_Count loop
         if Container.Points (I).Point_Type = Kind then
            Count := Count + 1;
         end if;
      end loop;
      
      declare
         Result : Coverage_Point_Array (1 .. Count);
         J      : Natural := 0;
      begin
         for I in 1 .. Container.Point_Count loop
            if Container.Points (I).Point_Type = Kind then
               J := J + 1;
               Result (J) := Container.Points (I);
            end if;
         end loop;
         return Result;
      end;
   end Get_Points_By_Type;

   function Get_Point_By_ID (
      Container : Coverage_Points;
      Point_ID  : String
   ) return Coverage_Point is
   begin
      for I in 1 .. Container.Point_Count loop
         if Container.Points (I).Point_ID_Len = Point_ID'Length and then
            Container.Points (I).Point_ID (1 .. Point_ID'Length) = Point_ID
         then
            return Container.Points (I);
         end if;
      end loop;
      return Null_Coverage_Point;
   end Get_Point_By_ID;

   function Point_Exists (
      Container : Coverage_Points;
      Point_ID  : String
   ) return Boolean is
   begin
      for I in 1 .. Container.Point_Count loop
         if Container.Points (I).Point_ID_Len = Point_ID'Length and then
            Container.Points (I).Point_ID (1 .. Point_ID'Length) = Point_ID
         then
            return True;
         end if;
      end loop;
      return False;
   end Point_Exists;

   --  ============================================================
   --  Coverage Status
   --  ============================================================

   procedure Mark_Instrumented (
      Container : in Out Coverage_Points;
      Point_ID  : in     String
   ) is
   begin
      for I in 1 .. Container.Point_Count loop
         if Container.Points (I).Point_ID_Len = Point_ID'Length and then
            Container.Points (I).Point_ID (1 .. Point_ID'Length) = Point_ID
         then
            Container.Points (I).Instrumented := True;
            exit;
         end if;
      end loop;
   end Mark_Instrumented;

   procedure Mark_Covered (
      Container : in Out Coverage_Points;
      Point_ID  : in     String
   ) is
   begin
      for I in 1 .. Container.Point_Count loop
         if Container.Points (I).Point_ID_Len = Point_ID'Length and then
            Container.Points (I).Point_ID (1 .. Point_ID'Length) = Point_ID
         then
            Container.Points (I).Covered := True;
            Container.Points (I).Coverage_Count := Container.Points (I).Coverage_Count + 1;
            exit;
         end if;
      end loop;
   end Mark_Covered;

   procedure Increment_Coverage (
      Container : in Out Coverage_Points;
      Point_ID  : in     String
   ) is
   begin
      for I in 1 .. Container.Point_Count loop
         if Container.Points (I).Point_ID_Len = Point_ID'Length and then
            Container.Points (I).Point_ID (1 .. Point_ID'Length) = Point_ID
         then
            Container.Points (I).Coverage_Count := Container.Points (I).Coverage_Count + 1;
            if not Container.Points (I).Covered then
               Container.Points (I).Covered := True;
            end if;
            exit;
         end if;
      end loop;
   end Increment_Coverage;

   --  ============================================================
   --  Point ID Generation
   --  ============================================================

   function Generate_Point_ID (
      Kind    : Coverage_Type;
      Element : String;
      Index   : Natural
   ) return String is
      Prefix : constant String := Get_Type_Prefix (Kind);
      Idx_Str : constant String := Natural_To_String (Index);
   begin
      Next_Point_Index := Next_Point_Index + 1;
      return Prefix & "_" & Element & "_" & Idx_Str;
   end Generate_Point_ID;

   function Get_Type_Prefix (Kind : Coverage_Type) return String is
   begin
      case Kind is
         when State_Coverage      => return "CP_STATE";
         when Transition_Coverage => return "CP_TRANS";
         when Decision_Coverage   => return "CP_DEC";
         when Condition_Coverage  => return "CP_COND";
         when MCDC_Coverage       => return "CP_MCDC";
         when Action_Coverage     => return "CP_ACT";
         when Entry_Coverage      => return "CP_ENTRY";
         when Exit_Coverage       => return "CP_EXIT";
         when Path_Coverage       => return "CP_PATH";
         when Loop_Coverage       => return "CP_LOOP";
      end case;
   end Get_Type_Prefix;

end Coverage;
