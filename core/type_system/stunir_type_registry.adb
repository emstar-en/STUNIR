-------------------------------------------------------------------------------
--  STUNIR Type Registry - Ada SPARK Body
--  Part of Phase 1 SPARK Migration
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

package body Stunir_Type_Registry is

   -------------------------------------------------------------------------
   --  Initialize: Set up registry with built-in types
   -------------------------------------------------------------------------
   procedure Initialize (Reg : out Type_Registry) is
      Id : Type_Id;
      pragma Unreferenced (Id);
   begin
      --  Clear all entries
      Reg.Count := 0;
      for I in Reg.Entries'Range loop
         Reg.Entries (I) := (Name => Empty_Type_Name, T => Void_Type, Is_Used => False);
      end loop;

      --  Register built-in types
      Register (Reg, Make_Type_Name ("void"), Void_Type, Id);
      Register (Reg, Make_Type_Name ("()"), Unit_Type, Id);
      Register (Reg, Make_Type_Name ("bool"), Bool_Type, Id);
      Register (Reg, Make_Type_Name ("i8"), I8, Id);
      Register (Reg, Make_Type_Name ("i16"), I16, Id);
      Register (Reg, Make_Type_Name ("i32"), I32, Id);
      Register (Reg, Make_Type_Name ("i64"), I64, Id);
      Register (Reg, Make_Type_Name ("u8"), U8, Id);
      Register (Reg, Make_Type_Name ("u16"), U16, Id);
      Register (Reg, Make_Type_Name ("u32"), U32, Id);
      Register (Reg, Make_Type_Name ("u64"), U64, Id);
      Register (Reg, Make_Type_Name ("f32"), F32, Id);
      Register (Reg, Make_Type_Name ("f64"), F64, Id);
      Register (Reg, Make_Type_Name ("char"), (Kind => Char_Kind, Name => Empty_Type_Name, Is_Unicode => True), Id);
      Register (Reg, Make_Type_Name ("String"), (Kind => String_Kind, Name => Empty_Type_Name, Is_Owned => True), Id);

      --  Add C type aliases
      Register (Reg, Make_Type_Name ("int"), I32, Id);
      Register (Reg, Make_Type_Name ("long"), I64, Id);
      Register (Reg, Make_Type_Name ("short"), I16, Id);
      Register (Reg, Make_Type_Name ("float"), F32, Id);
      Register (Reg, Make_Type_Name ("double"), F64, Id);
   end Initialize;

   -------------------------------------------------------------------------
   --  Register: Add a new type to the registry
   -------------------------------------------------------------------------
   procedure Register (
      Reg  : in out Type_Registry;
      Name : Type_Name;
      T    : STUNIR_Type;
      Id   : out Type_Id)
   is
   begin
      --  Check if name already exists
      if Has_Name (Reg, Name) then
         Id := Lookup (Reg, Name);
         return;
      end if;

      --  Add new entry
      Reg.Count := Reg.Count + 1;
      Id := Type_Id (Reg.Count);
      Reg.Entries (Id) := (Name => Name, T => T, Is_Used => True);
   end Register;

   -------------------------------------------------------------------------
   --  Lookup: Find a type by name
   -------------------------------------------------------------------------
   function Lookup (
      Reg  : Type_Registry;
      Name : Type_Name) return Type_Id
   is
   begin
      for I in 1 .. Type_Id (Reg.Count) loop
         if Reg.Entries (I).Is_Used and then Equal (Reg.Entries (I).Name, Name) then
            return I;
         end if;
      end loop;
      return No_Type_Id;
   end Lookup;

   -------------------------------------------------------------------------
   --  Get_Type: Retrieve a type by ID
   -------------------------------------------------------------------------
   function Get_Type (
      Reg : Type_Registry;
      Id  : Type_Id) return STUNIR_Type
   is
   begin
      return Reg.Entries (Id).T;
   end Get_Type;

   -------------------------------------------------------------------------
   --  Has_Type: Check if a type ID exists
   -------------------------------------------------------------------------
   function Has_Type (
      Reg : Type_Registry;
      Id  : Type_Id) return Boolean
   is
   begin
      return Id > 0 and then Id <= Type_Id (Reg.Count) and then Reg.Entries (Id).Is_Used;
   end Has_Type;

   -------------------------------------------------------------------------
   --  Has_Name: Check if a name is registered
   -------------------------------------------------------------------------
   function Has_Name (
      Reg  : Type_Registry;
      Name : Type_Name) return Boolean
   is
   begin
      return Lookup (Reg, Name) /= No_Type_Id;
   end Has_Name;

   -------------------------------------------------------------------------
   --  Get_Count: Get the number of registered types
   -------------------------------------------------------------------------
   function Get_Count (Reg : Type_Registry) return Natural is
   begin
      return Reg.Count;
   end Get_Count;

end Stunir_Type_Registry;
