--  STUNIR DO-332 Inheritance Analyzer Implementation
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

package body Inheritance_Analyzer is

   --  ============================================================
   --  Calculate_Depth Implementation
   --  ============================================================

   function Calculate_Depth (
      Class_ID : OOP_Types.Class_ID;
      Links    : Inheritance_Array
   ) return Natural is
      Max_Parent_Depth : Natural := 0;
      Current_Depth    : Natural;
   begin
      --  Get all direct parents
      for I in Links'Range loop
         if Links (I).Child_ID = Class_ID then
            --  Recursively get parent depth (with limit)
            Current_Depth := Calculate_Depth (Links (I).Parent_ID, Links);
            if Current_Depth < Max_Inheritance then
               Current_Depth := Current_Depth + 1;
            end if;
            if Current_Depth > Max_Parent_Depth then
               Max_Parent_Depth := Current_Depth;
            end if;
         end if;
      end loop;

      return Max_Parent_Depth;
   end Calculate_Depth;

   --  ============================================================
   --  Detect_Diamond_Pattern Implementation
   --  ============================================================

   function Detect_Diamond_Pattern (
      Class_ID : OOP_Types.Class_ID;
      Links    : Inheritance_Array
   ) return Diamond_Info is
      Result     : Diamond_Info := Null_Diamond_Info;
      Parents    : constant Class_ID_Array := Get_Parents (Class_ID, Links);
      Ancestors1 : Class_ID_Array (1 .. Max_Visited);
      Ancestors2 : Class_ID_Array (1 .. Max_Visited);
      Count1     : Natural := 0;
      Count2     : Natural := 0;
   begin
      --  Need at least 2 parents for diamond
      if Parents'Length < 2 then
         return Result;
      end if;

      --  Get ancestors of first two parents
      declare
         A1 : constant Class_ID_Array := Get_All_Ancestors (Parents (1), Links);
         A2 : constant Class_ID_Array := Get_All_Ancestors (Parents (2), Links);
      begin
         --  Copy to fixed arrays with bounds checking
         Count1 := Natural'Min (A1'Length, Max_Visited);
         Count2 := Natural'Min (A2'Length, Max_Visited);
         
         for I in 1 .. Count1 loop
            Ancestors1 (I) := A1 (A1'First + I - 1);
         end loop;
         
         for I in 1 .. Count2 loop
            Ancestors2 (I) := A2 (A2'First + I - 1);
         end loop;
      end;

      --  Check for common ancestor
      for I in 1 .. Count1 loop
         for J in 1 .. Count2 loop
            if Ancestors1 (I) = Ancestors2 (J) then
               Result.Has_Diamond := True;
               Result.Common_Ancestor := Ancestors1 (I);
               Result.Join_Class := Class_ID;
               Result.Left_Path_Length := I;
               Result.Right_Path_Length := J;
               return Result;
            end if;
         end loop;
      end loop;

      return Result;
   end Detect_Diamond_Pattern;

   --  ============================================================
   --  Has_Circular_Inheritance Implementation
   --  ============================================================

   function Has_Circular_Inheritance (
      Class_ID : OOP_Types.Class_ID;
      Links    : Inheritance_Array
   ) return Boolean is
      Visited  : Class_ID_Array (1 .. Max_Visited) := (others => Null_Class_ID);
      Count    : Natural := 0;
      
      function Visit (ID : Class_ID) return Boolean is
         Parents : constant Class_ID_Array := Get_Parents (ID, Links);
      begin
         --  Check if already visited
         for I in 1 .. Count loop
            if Visited (I) = ID then
               return True;  --  Circular!
            end if;
         end loop;

         --  Mark as visited
         if Count < Max_Visited then
            Count := Count + 1;
            Visited (Count) := ID;
         else
            return False;  --  Too deep, assume no cycle
         end if;

         --  Check all parents
         for I in Parents'Range loop
            if Visit (Parents (I)) then
               return True;
            end if;
         end loop;

         return False;
      end Visit;
   begin
      return Visit (Class_ID);
   end Has_Circular_Inheritance;

   --  ============================================================
   --  Get_All_Ancestors Implementation
   --  ============================================================

   function Get_All_Ancestors (
      Class_ID : OOP_Types.Class_ID;
      Links    : Inheritance_Array
   ) return Class_ID_Array is
      Result  : Class_ID_Array (1 .. Max_Visited);
      Count   : Natural := 0;
      
      procedure Collect (ID : Class_ID) is
         Parents : constant Class_ID_Array := Get_Parents (ID, Links);
         Found   : Boolean;
      begin
         for I in Parents'Range loop
            --  Check if already collected
            Found := False;
            for J in 1 .. Count loop
               if Result (J) = Parents (I) then
                  Found := True;
                  exit;
               end if;
            end loop;

            if not Found and Count < Max_Visited then
               Count := Count + 1;
               Result (Count) := Parents (I);
               Collect (Parents (I));
            end if;
         end loop;
      end Collect;
   begin
      Collect (Class_ID);
      return Result (1 .. Count);
   end Get_All_Ancestors;

   --  ============================================================
   --  Verify_Override Implementation
   --  ============================================================

   function Verify_Override (
      Child_Method  : Method_Info;
      Parent_Method : Method_Info
   ) return Override_Status is
   begin
      --  Check if parent method is final
      if Parent_Method.Kind = Final_Method then
         return Override_Of_Final_Method;
      end if;

      --  Check if parent method is virtual
      if not Is_Virtual (Parent_Method.Kind) then
         return Override_Not_Virtual;
      end if;

      --  Check visibility (child can't reduce visibility)
      if Child_Method.Visibility = V_Private and 
         Parent_Method.Visibility /= V_Private then
         return Override_Visibility_Reduced;
      end if;

      --  Check parameter count (simplified signature check)
      if Child_Method.Parameter_Count /= Parent_Method.Parameter_Count then
         return Override_Signature_Mismatch;
      end if;

      return Override_OK;
   end Verify_Override;

   --  ============================================================
   --  Analyze_Class_Overrides Implementation
   --  ============================================================

   function Analyze_Class_Overrides (
      Class_ID : OOP_Types.Class_ID;
      Methods  : Method_Array;
      Links    : Inheritance_Array
   ) return Natural is
      Invalid_Count : Natural := 0;
      Status        : Override_Status;
   begin
      for I in Methods'Range loop
         if Methods (I).Owning_Class = Class_ID and Methods (I).Has_Override then
            --  Find parent method
            for J in Methods'Range loop
               if Methods (J).ID = Methods (I).Override_Of then
                  Status := Verify_Override (Methods (I), Methods (J));
                  if Status /= Override_OK then
                     Invalid_Count := Invalid_Count + 1;
                  end if;
                  exit;
               end if;
            end loop;
         end if;
      end loop;
      return Invalid_Count;
   end Analyze_Class_Overrides;

   --  ============================================================
   --  Analyze_Inheritance Implementation
   --  ============================================================

   function Analyze_Inheritance (
      Class   : Class_Info;
      Classes : Class_Array;
      Methods : Method_Array;
      Links   : Inheritance_Array
   ) return Inheritance_Result is
      Result        : Inheritance_Result := Null_Inheritance_Result;
      Diamond       : Diamond_Info;
      Invalid_Count : Natural;
   begin
      Result.Class_ID := Class.ID;
      
      --  Calculate depth
      Result.Depth := Calculate_Depth (Class.ID, Links);
      
      --  Check for diamond
      Diamond := Detect_Diamond_Pattern (Class.ID, Links);
      Result.Has_Diamond := Diamond.Has_Diamond;
      
      --  Check for circular inheritance
      Result.Has_Circular := Has_Circular_Inheritance (Class.ID, Links);
      
      --  Count multiple inheritance
      Result.Multiple_Count := Class.Parent_Count;
      
      --  Count overrides and check validity
      Result.Override_Count := Count_Overrides (Class.ID, Methods);
      Invalid_Count := Analyze_Class_Overrides (Class.ID, Methods, Links);
      Result.All_Overrides_Valid := (Invalid_Count = 0);
      
      --  Count virtual and abstract methods
      Result.Virtual_Count := Count_Virtual_Methods (Class.ID, Methods);
      Result.Abstract_Count := Count_Abstract_Methods (Class.ID, Methods);
      
      --  Set status
      if Result.Has_Circular then
         Result.Status := Analysis_Error;
      elsif Result.Has_Diamond or not Result.All_Overrides_Valid then
         Result.Status := Analysis_Warning;
      else
         Result.Status := Analysis_OK;
      end if;
      
      return Result;
   end Analyze_Inheritance;

   --  ============================================================
   --  Analyze_All_Inheritance Implementation
   --  ============================================================

   procedure Analyze_All_Inheritance (
      Classes  : in     Class_Array;
      Methods  : in     Method_Array;
      Links    : in     Inheritance_Array;
      Results  :    out Inheritance_Result_Array;
      Success  :    out Boolean
   ) is
   begin
      Success := True;
      for I in Classes'Range loop
         if I in Results'Range then
            Results (I) := Analyze_Inheritance (Classes (I), Classes, Methods, Links);
            if Results (I).Status = Analysis_Error then
               Success := False;
            end if;
         end if;
      end loop;
   end Analyze_All_Inheritance;

   --  ============================================================
   --  Is_Ancestor Implementation
   --  ============================================================

   function Is_Ancestor (
      Ancestor_ID   : Class_ID;
      Descendant_ID : Class_ID;
      Links         : Inheritance_Array
   ) return Boolean is
      Ancestors : constant Class_ID_Array := Get_All_Ancestors (Descendant_ID, Links);
   begin
      for I in Ancestors'Range loop
         if Ancestors (I) = Ancestor_ID then
            return True;
         end if;
      end loop;
      return False;
   end Is_Ancestor;

   --  ============================================================
   --  Count Functions Implementation
   --  ============================================================

   function Count_Overrides (
      Class_ID : OOP_Types.Class_ID;
      Methods  : Method_Array
   ) return Natural is
      Count : Natural := 0;
   begin
      for I in Methods'Range loop
         if Methods (I).Owning_Class = Class_ID and Methods (I).Has_Override then
            Count := Count + 1;
         end if;
      end loop;
      return Count;
   end Count_Overrides;

   function Count_Virtual_Methods (
      Class_ID : OOP_Types.Class_ID;
      Methods  : Method_Array
   ) return Natural is
      Count : Natural := 0;
   begin
      for I in Methods'Range loop
         if Methods (I).Owning_Class = Class_ID and Is_Virtual (Methods (I).Kind) then
            Count := Count + 1;
         end if;
      end loop;
      return Count;
   end Count_Virtual_Methods;

   function Count_Abstract_Methods (
      Class_ID : OOP_Types.Class_ID;
      Methods  : Method_Array
   ) return Natural is
      Count : Natural := 0;
   begin
      for I in Methods'Range loop
         if Methods (I).Owning_Class = Class_ID and 
            (Methods (I).Kind = Abstract_Method or Methods (I).Kind = Pure_Virtual) then
            Count := Count + 1;
         end if;
      end loop;
      return Count;
   end Count_Abstract_Methods;

end Inheritance_Analyzer;
