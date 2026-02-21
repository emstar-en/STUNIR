--  STUNIR DO-332 Dispatch Analyzer Implementation
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

with Inheritance_Analyzer; use Inheritance_Analyzer;

package body Dispatch_Analyzer is

   --  ============================================================
   --  Resolve_Targets Implementation
   --  ============================================================

   function Resolve_Targets (
      Site    : Dispatch_Site;
      Classes : Class_Array;
      Methods : Method_Array;
      Links   : Inheritance_Array
   ) return Target_Array is
      Temp_Targets : Target_Array (1 .. Max_Targets_Per_Site);
      Count        : Natural := 0;
      Method_Name_Str : constant String := 
         Site.Method_Name (1 .. Site.Method_Name_Len);
   begin
      --  Find method in receiver class and all subclasses
      for I in Classes'Range loop
         --  Check if this class is the receiver or a subclass
         if Classes (I).ID = Site.Receiver_Type or else
            Is_Ancestor (Site.Receiver_Type, Classes (I).ID, Links) then
            
            --  Find the method in this class
            declare
               Mid : constant Method_ID := 
                  Find_Method_By_Name (Classes (I).ID, Method_Name_Str, Methods);
            begin
               if Mid /= Null_Method_ID and Count < Max_Targets_Per_Site then
                  --  Find method info
                  for J in Methods'Range loop
                     if Methods (J).ID = Mid then
                        Count := Count + 1;
                        Temp_Targets (Count) := (
                           Target_Class  => Classes (I).ID,
                           Target_Method => Mid,
                           Is_Final      => Methods (J).Kind = Final_Method,
                           Is_Abstract   => Methods (J).Kind = Abstract_Method or
                                           Methods (J).Kind = Pure_Virtual
                        );
                        exit;
                     end if;
                  end loop;
               end if;
            end;
         end if;
      end loop;

      --  Return the targets found
      if Count = 0 then
         declare
            Empty : Target_Array (1 .. 0);
         begin
            return Empty;
         end;
      else
         return Temp_Targets (1 .. Count);
      end if;
   end Resolve_Targets;

   --  ============================================================
   --  Analyze_Site Implementation
   --  ============================================================

   function Analyze_Site (
      Site    : Dispatch_Site;
      Classes : Class_Array;
      Methods : Method_Array;
      Links   : Inheritance_Array
   ) return Site_Analysis is
      Result  : Site_Analysis := Null_Site_Analysis;
      Targets : constant Target_Array := Resolve_Targets (Site, Classes, Methods, Links);
   begin
      Result.Site_ID := Site.Site_ID;
      Result.Target_Count := Targets'Length;
      Result.Is_Bounded := True;  --  Always bounded since we enumerate all

      --  Check if devirtualizable (single target)
      Result.Is_Devirtualizable := (Targets'Length = 1);
      Result.Proven_Single := Result.Is_Devirtualizable;

      --  Check for abstract targets
      for I in Targets'Range loop
         if Targets (I).Is_Abstract then
            Result.Has_Abstract := True;
            exit;
         end if;
      end loop;

      return Result;
   end Analyze_Site;

   --  ============================================================
   --  Can_Devirtualize Implementation
   --  ============================================================

   function Can_Devirtualize (
      Site    : Dispatch_Site;
      Classes : Class_Array;
      Methods : Method_Array;
      Links   : Inheritance_Array
   ) return Boolean is
      Analysis : constant Site_Analysis := Analyze_Site (Site, Classes, Methods, Links);
   begin
      return Analysis.Is_Devirtualizable;
   end Can_Devirtualize;

   --  ============================================================
   --  Analyze_All_Dispatch Implementation
   --  ============================================================

   procedure Analyze_All_Dispatch (
      Sites    : in     Dispatch_Site_Array;
      Classes  : in     Class_Array;
      Methods  : in     Method_Array;
      Links    : in     Inheritance_Array;
      Results  :    out Site_Analysis_Array;
      Summary  :    out Dispatch_Summary;
      Success  :    out Boolean
   ) is
      Max_Targets : Natural := 0;
   begin
      Summary := Null_Dispatch_Summary;
      Summary.Total_Sites := Sites'Length;
      Success := True;

      for I in Sites'Range loop
         if I in Results'Range then
            Results (I) := Analyze_Site (Sites (I), Classes, Methods, Links);

            --  Update summary
            if Results (I).Is_Bounded then
               Summary.Bounded_Sites := Summary.Bounded_Sites + 1;
            else
               Summary.Unbounded_Sites := Summary.Unbounded_Sites + 1;
               Success := False;  --  Unbounded dispatch is an error
            end if;

            if Results (I).Is_Devirtualizable then
               Summary.Devirtualized := Summary.Devirtualized + 1;
            end if;

            if Results (I).Target_Count > Max_Targets then
               Max_Targets := Results (I).Target_Count;
            end if;

            if Results (I).Has_Abstract then
               Summary.Sites_With_Abstract := Summary.Sites_With_Abstract + 1;
            end if;
         end if;
      end loop;

      Summary.Max_Targets := Max_Targets;
   end Analyze_All_Dispatch;

   --  ============================================================
   --  Find_Method_By_Name Implementation
   --  ============================================================

   function Find_Method_By_Name (
      Class_ID    : OOP_Types.Class_ID;
      Method_Name : String;
      Methods     : Method_Array
   ) return Method_ID is
   begin
      for I in Methods'Range loop
         if Methods (I).Owning_Class = Class_ID then
            declare
               M_Name : constant String := Methods (I).Name (1 .. Methods (I).Name_Length);
            begin
               if M_Name'Length = Method_Name'Length then
                  --  Simple string comparison
                  declare
                     Match : Boolean := True;
                  begin
                     for J in Method_Name'Range loop
                        if Method_Name (J) /= M_Name (J - Method_Name'First + M_Name'First) then
                           Match := False;
                           exit;
                        end if;
                     end loop;
                     if Match then
                        return Methods (I).ID;
                     end if;
                  end;
               end if;
            end;
         end if;
      end loop;
      return Null_Method_ID;
   end Find_Method_By_Name;

   --  ============================================================
   --  Get_Implementations Implementation
   --  ============================================================

   function Get_Implementations (
      Virtual_Method : Method_ID;
      Methods        : Method_Array
   ) return Method_ID_Array is
      Count : Natural := 0;
   begin
      --  Count implementations (overrides)
      for I in Methods'Range loop
         if Methods (I).Override_Of = Virtual_Method then
            Count := Count + 1;
         end if;
      end loop;

      if Count = 0 then
         declare
            Empty : Method_ID_Array (1 .. 0);
         begin
            return Empty;
         end;
      end if;

      declare
         Result : Method_ID_Array (1 .. Count);
         Idx    : Positive := 1;
      begin
         for I in Methods'Range loop
            if Methods (I).Override_Of = Virtual_Method then
               Result (Idx) := Methods (I).ID;
               if Idx < Count then
                  Idx := Idx + 1;
               end if;
            end if;
         end loop;
         return Result;
      end;
   end Get_Implementations;

   --  ============================================================
   --  Count_Targets Implementation
   --  ============================================================

   function Count_Targets (
      Site    : Dispatch_Site;
      Classes : Class_Array;
      Methods : Method_Array;
      Links   : Inheritance_Array
   ) return Natural is
      Targets : constant Target_Array := Resolve_Targets (Site, Classes, Methods, Links);
   begin
      return Targets'Length;
   end Count_Targets;

end Dispatch_Analyzer;
