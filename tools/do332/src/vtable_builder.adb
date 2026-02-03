--  STUNIR DO-332 VTable Builder Implementation
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

with Inheritance_Analyzer; use Inheritance_Analyzer;

package body VTable_Builder is

   --  ============================================================
   --  Build_VTable Implementation
   --  ============================================================

   function Build_VTable (
      Class   : Class_Info;
      Classes : Class_Array;
      Methods : Method_Array;
      Links   : Inheritance_Array
   ) return VTable is
      Result       : VTable := Null_VTable;
      Virtual_Count : Natural := 0;
      Has_Abstract  : Boolean := False;
   begin
      Result.Class_ID := Class.ID;

      --  Count virtual methods (own and inherited)
      for I in Methods'Range loop
         if Is_Virtual (Methods (I).Kind) then
            --  Check if method belongs to this class or is inherited
            if Methods (I).Owning_Class = Class.ID then
               Virtual_Count := Virtual_Count + 1;
               if Methods (I).Kind = Abstract_Method or Methods (I).Kind = Pure_Virtual then
                  Has_Abstract := True;
               end if;
            elsif Is_Ancestor (Methods (I).Owning_Class, Class.ID, Links) then
               --  Inherited method - only count if not overridden
               declare
                  Is_Overridden : Boolean := False;
               begin
                  for J in Methods'Range loop
                     if Methods (J).Owning_Class = Class.ID and
                        Methods (J).Override_Of = Methods (I).ID then
                        Is_Overridden := True;
                        exit;
                     end if;
                  end loop;
                  if not Is_Overridden then
                     Virtual_Count := Virtual_Count + 1;
                     if Methods (I).Kind = Abstract_Method or Methods (I).Kind = Pure_Virtual then
                        Has_Abstract := True;
                     end if;
                  end if;
               end;
            end if;
         end if;
      end loop;

      Result.Entry_Count := Virtual_Count;
      Result.Has_Abstract := Has_Abstract;
      return Result;
   end Build_VTable;

   --  ============================================================
   --  Build_All_VTables Implementation
   --  ============================================================

   procedure Build_All_VTables (
      Classes  : in     Class_Array;
      Methods  : in     Method_Array;
      Links    : in     Inheritance_Array;
      VTables  :    out VTable_Set;
      Success  :    out Boolean
   ) is
   begin
      Success := True;
      for I in Classes'Range loop
         if I in VTables'Range then
            VTables (I) := Build_VTable (Classes (I), Classes, Methods, Links);
            --  Warn if class is not abstract but has unimplemented methods
            if VTables (I).Has_Abstract and Classes (I).Kind /= Abstract_Class then
               Success := False;  --  This is an error condition
            end if;
         end if;
      end loop;
   end Build_All_VTables;

   --  ============================================================
   --  Get_VTable_Entries Implementation
   --  ============================================================

   function Get_VTable_Entries (
      Class   : Class_Info;
      Classes : Class_Array;
      Methods : Method_Array;
      Links   : Inheritance_Array
   ) return Extended_VTable_Array is
      Temp    : Extended_VTable_Array (1 .. Max_VTable_Entries);
      Count   : Natural := 0;
      Slot    : Natural := 0;
   begin
      --  Collect all virtual methods for this class
      for I in Methods'Range loop
         if Is_Virtual (Methods (I).Kind) and Count < Max_VTable_Entries then
            --  Method in this class
            if Methods (I).Owning_Class = Class.ID then
               Slot := Slot + 1;
               Count := Count + 1;
               Temp (Count) := (
                  Slot_Index      => Slot,
                  Method_ID       => Methods (I).ID,
                  Method_Name     => Methods (I).Name,
                  Name_Length     => Methods (I).Name_Length,
                  Declaring_Class => Methods (I).Owning_Class,
                  Impl_Class      => Methods (I).Owning_Class,
                  Is_Abstract     => Methods (I).Kind = Abstract_Method or
                                    Methods (I).Kind = Pure_Virtual,
                  Is_Final        => Methods (I).Kind = Final_Method,
                  Is_Override     => Methods (I).Has_Override
               );
            end if;
         end if;
      end loop;

      if Count = 0 then
         declare
            Empty : Extended_VTable_Array (1 .. 0);
         begin
            return Empty;
         end;
      end if;

      return Temp (1 .. Count);
   end Get_VTable_Entries;

   --  ============================================================
   --  Find_Slot Implementation
   --  ============================================================

   function Find_Slot (
      Method_Name : String;
      Entries     : Extended_VTable_Array
   ) return Natural is
   begin
      for I in Entries'Range loop
         declare
            E_Name : constant String := Entries (I).Method_Name (1 .. Entries (I).Name_Length);
         begin
            if E_Name'Length = Method_Name'Length then
               declare
                  Match : Boolean := True;
               begin
                  for J in Method_Name'Range loop
                     if Method_Name (J) /= E_Name (J - Method_Name'First + E_Name'First) then
                        Match := False;
                        exit;
                     end if;
                  end loop;
                  if Match then
                     return Entries (I).Slot_Index;
                  end if;
               end;
            end if;
         end;
      end loop;
      return 0;
   end Find_Slot;

   --  ============================================================
   --  Merge_VTable Implementation
   --  ============================================================

   function Merge_VTable (
      Parent_Entries : Extended_VTable_Array;
      Child_Methods  : Method_Array;
      Child_Class    : Class_ID
   ) return Extended_VTable_Array is
      Result : Extended_VTable_Array := Parent_Entries;
   begin
      --  Update slots with child overrides
      for I in Result'Range loop
         for J in Child_Methods'Range loop
            if Child_Methods (J).Owning_Class = Child_Class and
               Child_Methods (J).Override_Of = Result (I).Method_ID then
               Result (I).Impl_Class := Child_Class;
               Result (I).Method_ID := Child_Methods (J).ID;
               Result (I).Is_Abstract := Child_Methods (J).Kind = Abstract_Method or
                                        Child_Methods (J).Kind = Pure_Virtual;
               Result (I).Is_Final := Child_Methods (J).Kind = Final_Method;
               Result (I).Is_Override := True;
               exit;
            end if;
         end loop;
      end loop;
      return Result;
   end Merge_VTable;

   --  ============================================================
   --  Get_Inherited_Virtuals Implementation
   --  ============================================================

   function Get_Inherited_Virtuals (
      Class   : Class_ID;
      Classes : Class_Array;
      Methods : Method_Array;
      Links   : Inheritance_Array
   ) return Method_ID_Array is
      Ancestors : constant Class_ID_Array := Get_All_Ancestors (Class, Links);
      Temp      : Method_ID_Array (1 .. Max_VTable_Entries);
      Count     : Natural := 0;
   begin
      --  Collect virtual methods from all ancestors
      for I in Ancestors'Range loop
         for J in Methods'Range loop
            if Methods (J).Owning_Class = Ancestors (I) and
               Is_Virtual (Methods (J).Kind) and
               Count < Max_VTable_Entries then
               Count := Count + 1;
               Temp (Count) := Methods (J).ID;
            end if;
         end loop;
      end loop;

      if Count = 0 then
         declare
            Empty : Method_ID_Array (1 .. 0);
         begin
            return Empty;
         end;
      end if;

      return Temp (1 .. Count);
   end Get_Inherited_Virtuals;

end VTable_Builder;
