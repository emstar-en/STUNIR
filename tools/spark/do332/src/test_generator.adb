--  STUNIR DO-332 Test Generator Implementation
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

package body Test_Generator is

   --  Local test ID counter
   Next_Test_ID : Natural := 1;

   --  ============================================================
   --  Generate_Inheritance_Tests Implementation
   --  ============================================================

   function Generate_Inheritance_Tests (
      Classes : Class_Array;
      Links   : Inheritance_Array;
      Results : Inheritance_Result_Array
   ) return Test_Case_Array is
      Temp  : Test_Case_Array (1 .. Max_Test_Cases);
      Count : Natural := 0;
   begin
      for I in Classes'Range loop
         if Count < Max_Test_Cases then
            --  Generate inheritance chain test
            Count := Count + 1;
            Temp (Count) := (
               Test_ID       => Next_Test_ID,
               Category      => Inheritance_Test,
               Objective     => OO_1,
               Target_Class  => Classes (I).ID,
               Target_Method => Null_Method_ID,
               Site_ID       => 0,
               Setup_Steps   => 1,
               Assertions    => 2,  --  Check depth and parent chain
               Coverage_Points => 1
            );
            Next_Test_ID := Next_Test_ID + 1;

            --  Generate override tests if class has overrides
            if I in Results'Range and Results (I).Override_Count > 0 then
               if Count < Max_Test_Cases then
                  Count := Count + 1;
                  Temp (Count) := (
                     Test_ID       => Next_Test_ID,
                     Category      => Override_Test,
                     Objective     => OO_1,
                     Target_Class  => Classes (I).ID,
                     Target_Method => Null_Method_ID,
                     Site_ID       => 0,
                     Setup_Steps   => 2,
                     Assertions    => Natural (Results (I).Override_Count),
                     Coverage_Points => Natural (Results (I).Override_Count)
                  );
                  Next_Test_ID := Next_Test_ID + 1;
               end if;
            end if;
         end if;
      end loop;

      if Count = 0 then
         declare
            Empty : Test_Case_Array (1 .. 0);
         begin
            return Empty;
         end;
      end if;

      return Temp (1 .. Count);
   end Generate_Inheritance_Tests;

   --  ============================================================
   --  Generate_Polymorphism_Tests Implementation
   --  ============================================================

   function Generate_Polymorphism_Tests (
      Classes : Class_Array;
      Methods : Method_Array;
      Results : Polymorphism_Result_Array
   ) return Test_Case_Array is
      Temp  : Test_Case_Array (1 .. Max_Test_Cases);
      Count : Natural := 0;
   begin
      for I in Classes'Range loop
         if I in Results'Range and Results (I).Virtual_Methods > 0 then
            if Count < Max_Test_Cases then
               Count := Count + 1;
               Temp (Count) := (
                  Test_ID       => Next_Test_ID,
                  Category      => Polymorphism_Test,
                  Objective     => OO_2,
                  Target_Class  => Classes (I).ID,
                  Target_Method => Null_Method_ID,
                  Site_ID       => 0,
                  Setup_Steps   => 2,
                  Assertions    => Natural (Results (I).Virtual_Methods),
                  Coverage_Points => Natural (Results (I).Virtual_Methods)
               );
               Next_Test_ID := Next_Test_ID + 1;
            end if;
         end if;
      end loop;

      if Count = 0 then
         declare
            Empty : Test_Case_Array (1 .. 0);
         begin
            return Empty;
         end;
      end if;

      return Temp (1 .. Count);
   end Generate_Polymorphism_Tests;

   --  ============================================================
   --  Generate_Dispatch_Tests Implementation
   --  ============================================================

   function Generate_Dispatch_Tests (
      Sites    : Dispatch_Site_Array;
      Classes  : Class_Array;
      Methods  : Method_Array;
      Links    : Inheritance_Array
   ) return Test_Case_Array is
      Temp  : Test_Case_Array (1 .. Max_Test_Cases);
      Count : Natural := 0;
   begin
      for I in Sites'Range loop
         --  Get targets for this site
         declare
            Targets : constant Target_Array := 
               Resolve_Targets (Sites (I), Classes, Methods, Links);
         begin
            --  Generate one test per target
            for J in Targets'Range loop
               if Count < Max_Test_Cases then
                  Count := Count + 1;
                  Temp (Count) := (
                     Test_ID       => Next_Test_ID,
                     Category      => Dispatch_Test,
                     Objective     => OO_3,
                     Target_Class  => Targets (J).Target_Class,
                     Target_Method => Targets (J).Target_Method,
                     Site_ID       => Sites (I).Site_ID,
                     Setup_Steps   => 3,
                     Assertions    => 2,
                     Coverage_Points => 1
                  );
                  Next_Test_ID := Next_Test_ID + 1;
               end if;
            end loop;
         end;
      end loop;

      if Count = 0 then
         declare
            Empty : Test_Case_Array (1 .. 0);
         begin
            return Empty;
         end;
      end if;

      return Temp (1 .. Count);
   end Generate_Dispatch_Tests;

   --  ============================================================
   --  Generate_Coupling_Tests Implementation
   --  ============================================================

   function Generate_Coupling_Tests (
      Classes : Class_Array;
      Results : Coupling_Result_Array
   ) return Test_Case_Array is
      Temp  : Test_Case_Array (1 .. Max_Test_Cases);
      Count : Natural := 0;
   begin
      for I in Classes'Range loop
         if I in Results'Range then
            if Count < Max_Test_Cases then
               Count := Count + 1;
               Temp (Count) := (
                  Test_ID       => Next_Test_ID,
                  Category      => Coupling_Test,
                  Objective     => OO_4,
                  Target_Class  => Classes (I).ID,
                  Target_Method => Null_Method_ID,
                  Site_ID       => 0,
                  Setup_Steps   => 1,
                  Assertions    => 6,  --  One per metric
                  Coverage_Points => 1
               );
               Next_Test_ID := Next_Test_ID + 1;
            end if;
         end if;
      end loop;

      if Count = 0 then
         declare
            Empty : Test_Case_Array (1 .. 0);
         begin
            return Empty;
         end;
      end if;

      return Temp (1 .. Count);
   end Generate_Coupling_Tests;

   --  ============================================================
   --  Generate_All_Tests Implementation
   --  ============================================================

   procedure Generate_All_Tests (
      Classes       : in     Class_Array;
      Methods       : in     Method_Array;
      Links         : in     Inheritance_Array;
      Dispatch_Sites: in     Dispatch_Site_Array;
      Inh_Results   : in     Inheritance_Result_Array;
      Poly_Results  : in     Polymorphism_Result_Array;
      Coup_Results  : in     Coupling_Result_Array;
      Tests         :    out Test_Case_Array;
      Test_Count    :    out Natural;
      Summary       :    out Generation_Summary
   ) is
      Idx : Natural := 0;
   begin
      Summary := Null_Generation_Summary;
      Next_Test_ID := 1;  --  Reset counter

      --  Generate inheritance tests
      declare
         Inh_Tests : constant Test_Case_Array := 
            Generate_Inheritance_Tests (Classes, Links, Inh_Results);
      begin
         for I in Inh_Tests'Range loop
            if Idx < Tests'Length then
               Idx := Idx + 1;
               Tests (Tests'First + Idx - 1) := Inh_Tests (I);
               case Inh_Tests (I).Category is
                  when Inheritance_Test =>
                     Summary.Inheritance_Tests := Summary.Inheritance_Tests + 1;
                  when Override_Test =>
                     Summary.Override_Tests := Summary.Override_Tests + 1;
                  when others => null;
               end case;
            end if;
         end loop;
      end;

      --  Generate polymorphism tests
      declare
         Poly_Tests : constant Test_Case_Array := 
            Generate_Polymorphism_Tests (Classes, Methods, Poly_Results);
      begin
         for I in Poly_Tests'Range loop
            if Idx < Tests'Length then
               Idx := Idx + 1;
               Tests (Tests'First + Idx - 1) := Poly_Tests (I);
               Summary.Polymorphism_Tests := Summary.Polymorphism_Tests + 1;
            end if;
         end loop;
      end;

      --  Generate dispatch tests
      declare
         Disp_Tests : constant Test_Case_Array := 
            Generate_Dispatch_Tests (Dispatch_Sites, Classes, Methods, Links);
      begin
         for I in Disp_Tests'Range loop
            if Idx < Tests'Length then
               Idx := Idx + 1;
               Tests (Tests'First + Idx - 1) := Disp_Tests (I);
               Summary.Dispatch_Tests := Summary.Dispatch_Tests + 1;
            end if;
         end loop;
      end;

      --  Generate coupling tests
      declare
         Coup_Tests : constant Test_Case_Array := 
            Generate_Coupling_Tests (Classes, Coup_Results);
      begin
         for I in Coup_Tests'Range loop
            if Idx < Tests'Length then
               Idx := Idx + 1;
               Tests (Tests'First + Idx - 1) := Coup_Tests (I);
               Summary.Coupling_Tests := Summary.Coupling_Tests + 1;
            end if;
         end loop;
      end;

      Test_Count := Idx;
      Summary.Total_Tests := Idx;
      Summary.Coverage_Points := Count_Coverage_Points (Tests (Tests'First .. Tests'First + Idx - 1));
   end Generate_All_Tests;

   --  ============================================================
   --  All_Targets_Covered Implementation
   --  ============================================================

   function All_Targets_Covered (
      Tests  : Test_Case_Array;
      Sites  : Dispatch_Site_Array;
      Results : Site_Analysis_Array
   ) return Boolean is
      Targets_Tested : Natural := 0;
      Total_Targets  : Natural := 0;
   begin
      --  Count total targets
      for I in Results'Range loop
         Total_Targets := Total_Targets + Results (I).Target_Count;
      end loop;

      --  Count tested targets
      for I in Tests'Range loop
         if Tests (I).Category = Dispatch_Test then
            Targets_Tested := Targets_Tested + 1;
         end if;
      end loop;

      return Targets_Tested >= Total_Targets;
   end All_Targets_Covered;

   --  ============================================================
   --  Count_Coverage_Points Implementation
   --  ============================================================

   function Count_Coverage_Points (
      Tests : Test_Case_Array
   ) return Natural is
      Total : Natural := 0;
   begin
      for I in Tests'Range loop
         Total := Total + Tests (I).Coverage_Points;
      end loop;
      return Total;
   end Count_Coverage_Points;

end Test_Generator;
