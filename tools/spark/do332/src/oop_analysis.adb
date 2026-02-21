--  STUNIR DO-332 OOP Analysis Framework Implementation
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

package body OOP_Analysis is

   --  ============================================================
   --  Create_Hierarchy Implementation
   --  ============================================================

   function Create_Hierarchy (
      Classes : Class_Array;
      Methods : Method_Array;
      Links   : Inheritance_Array
   ) return Class_Hierarchy is
      Result : Class_Hierarchy := Null_Hierarchy;
   begin
      Result.Class_Count := Classes'Length;
      Result.Method_Count := Methods'Length;
      Result.Link_Count := Links'Length;
      Result.Schema_Version := 1;
      return Result;
   end Create_Hierarchy;

   --  ============================================================
   --  Run_Analysis Implementation
   --  ============================================================

   procedure Run_Analysis (
      Hierarchy     : in     Class_Hierarchy;
      Classes       : in     Class_Array;
      Methods       : in     Method_Array;
      Links         : in     Inheritance_Array;
      Config        : in     Analysis_Config;
      Summary       : out    Analysis_Summary;
      Success       : out    Boolean
   ) is
      Max_Depth        : Natural := 0;
      Diamond_Count    : Natural := 0;
      Virtual_Count    : Natural := 0;
      Dispatch_Count   : Natural := 0;
      Total_CBO        : Natural := 0;
      Max_CBO          : Natural := 0;
   begin
      --  Initialize summary
      Summary := Null_Summary;
      Summary.Total_Classes := Hierarchy.Class_Count;
      Summary.Total_Methods := Hierarchy.Method_Count;
      Summary.Total_Inheritance := Hierarchy.Link_Count;
      Success := True;

      --  Count virtual methods
      for I in Methods'Range loop
         if Is_Virtual (Methods (I).Kind) then
            Virtual_Count := Virtual_Count + 1;
         end if;
      end loop;
      Summary.Virtual_Methods := Virtual_Count;

      --  Calculate max inheritance depth
      for I in Classes'Range loop
         if Classes (I).Inheritance_Depth > Max_Depth then
            Max_Depth := Classes (I).Inheritance_Depth;
         end if;
      end loop;
      Summary.Max_Depth := Max_Depth;

      --  Check for diamond patterns (simplified check)
      for I in Classes'Range loop
         if Classes (I).Parent_Count > 1 then
            --  Could be diamond, need deeper analysis
            Diamond_Count := Diamond_Count + 1;
         end if;
      end loop;
      Summary.Diamond_Patterns := Diamond_Count;

      --  Count polymorphic calls (virtual method calls)
      Summary.Polymorphic_Calls := Virtual_Count;

      --  Set overall status
      if Diamond_Count > 0 or Max_Depth > Natural (Config.Max_Inheritance_Depth) then
         Summary.Overall_Status := Analysis_Warning;
      else
         Summary.Overall_Status := Analysis_OK;
      end if;

   end Run_Analysis;

   --  ============================================================
   --  Find_Class Implementation
   --  ============================================================

   function Find_Class (
      Classes : Class_Array;
      ID      : Class_ID
   ) return Natural is
   begin
      for I in Classes'Range loop
         if Classes (I).ID = ID then
            return I;
         end if;
      end loop;
      return 0;
   end Find_Class;

   --  ============================================================
   --  Find_Method Implementation
   --  ============================================================

   function Find_Method (
      Methods : Method_Array;
      ID      : Method_ID
   ) return Natural is
   begin
      for I in Methods'Range loop
         if Methods (I).ID = ID then
            return I;
         end if;
      end loop;
      return 0;
   end Find_Method;

   --  ============================================================
   --  Get_Parents Implementation
   --  ============================================================

   function Get_Parents (
      Class_ID : OOP_Types.Class_ID;
      Links    : Inheritance_Array
   ) return Class_ID_Array is
      Count : Natural := 0;
   begin
      --  First count parents
      for I in Links'Range loop
         if Links (I).Child_ID = Class_ID then
            Count := Count + 1;
         end if;
      end loop;

      --  If no parents, return empty array
      if Count = 0 then
         declare
            Empty : Class_ID_Array (1 .. 0);
         begin
            return Empty;
         end;
      end if;

      --  Build result array
      declare
         Result : Class_ID_Array (1 .. Count);
         Idx    : Positive := 1;
      begin
         for I in Links'Range loop
            if Links (I).Child_ID = Class_ID then
               Result (Idx) := Links (I).Parent_ID;
               if Idx < Count then
                  Idx := Idx + 1;
               end if;
            end if;
         end loop;
         return Result;
      end;
   end Get_Parents;

   --  ============================================================
   --  Get_Children Implementation
   --  ============================================================

   function Get_Children (
      Class_ID : OOP_Types.Class_ID;
      Links    : Inheritance_Array
   ) return Class_ID_Array is
      Count : Natural := 0;
   begin
      --  First count children
      for I in Links'Range loop
         if Links (I).Parent_ID = Class_ID then
            Count := Count + 1;
         end if;
      end loop;

      --  If no children, return empty array
      if Count = 0 then
         declare
            Empty : Class_ID_Array (1 .. 0);
         begin
            return Empty;
         end;
      end if;

      --  Build result array
      declare
         Result : Class_ID_Array (1 .. Count);
         Idx    : Positive := 1;
      begin
         for I in Links'Range loop
            if Links (I).Parent_ID = Class_ID then
               Result (Idx) := Links (I).Child_ID;
               if Idx < Count then
                  Idx := Idx + 1;
               end if;
            end if;
         end loop;
         return Result;
      end;
   end Get_Children;

   --  ============================================================
   --  Get_Class_Methods Implementation
   --  ============================================================

   function Get_Class_Methods (
      Class_ID : OOP_Types.Class_ID;
      Methods  : Method_Array
   ) return Method_ID_Array is
      Count : Natural := 0;
   begin
      --  First count methods
      for I in Methods'Range loop
         if Methods (I).Owning_Class = Class_ID then
            Count := Count + 1;
         end if;
      end loop;

      --  If no methods, return empty array
      if Count = 0 then
         declare
            Empty : Method_ID_Array (1 .. 0);
         begin
            return Empty;
         end;
      end if;

      --  Build result array
      declare
         Result : Method_ID_Array (1 .. Count);
         Idx    : Positive := 1;
      begin
         for I in Methods'Range loop
            if Methods (I).Owning_Class = Class_ID then
               Result (Idx) := Methods (I).ID;
               if Idx < Count then
                  Idx := Idx + 1;
               end if;
            end if;
         end loop;
         return Result;
      end;
   end Get_Class_Methods;

end OOP_Analysis;
