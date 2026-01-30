-------------------------------------------------------------------------------
--  STUNIR IR Parser - Ada SPARK Body
--  Part of Phase 1 SPARK Migration
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

package body IR_Parser is

   -------------------------------------------------------------------------
   --  Make_Name: Create a bounded name from a string
   -------------------------------------------------------------------------
   function Make_Name (S : String) return Bounded_Name is
      Result : Bounded_Name;
   begin
      Result.Length := S'Length;
      for I in S'Range loop
         Result.Data (I - S'First + 1) := S (I);
      end loop;
      return Result;
   end Make_Name;

   -------------------------------------------------------------------------
   --  Make_Schema: Create a schema name from a string
   -------------------------------------------------------------------------
   function Make_Schema (S : String) return Schema_Name is
      Result : Schema_Name;
   begin
      Result.Length := S'Length;
      for I in S'Range loop
         Result.Data (I - S'First + 1) := S (I);
      end loop;
      return Result;
   end Make_Schema;

   -------------------------------------------------------------------------
   --  Names_Equal: Compare two bounded names
   -------------------------------------------------------------------------
   function Names_Equal (Left, Right : Bounded_Name) return Boolean is
   begin
      if Left.Length /= Right.Length then
         return False;
      end if;
      for I in 1 .. Left.Length loop
         if Left.Data (I) /= Right.Data (I) then
            return False;
         end if;
      end loop;
      return True;
   end Names_Equal;

   -------------------------------------------------------------------------
   --  Schemas_Equal: Compare two schema names
   -------------------------------------------------------------------------
   function Schemas_Equal (Left, Right : Schema_Name) return Boolean is
   begin
      if Left.Length /= Right.Length then
         return False;
      end if;
      for I in 1 .. Left.Length loop
         if Left.Data (I) /= Right.Data (I) then
            return False;
         end if;
      end loop;
      return True;
   end Schemas_Equal;

   -------------------------------------------------------------------------
   --  Identify_Schema: Identify the schema kind
   -------------------------------------------------------------------------
   function Identify_Schema (S : Schema_Name) return IR_Schema_Kind is
   begin
      --  Check for stunir.ir.v1
      if S.Length >= 12 and then S.Data (1 .. 10) = "stunir.ir." then
         return IR_V1_Schema;
      end if;
      return Unknown_Schema;
   end Identify_Schema;

   -------------------------------------------------------------------------
   --  Is_Valid_Schema: Check if a schema name is valid
   -------------------------------------------------------------------------
   function Is_Valid_Schema (S : Schema_Name) return Boolean is
   begin
      return S.Length > 0 and then Identify_Schema (S) /= Unknown_Schema;
   end Is_Valid_Schema;

   -------------------------------------------------------------------------
   --  Is_Valid_Module_Name: Check if a module name is valid
   -------------------------------------------------------------------------
   function Is_Valid_Module_Name (N : Bounded_Name) return Boolean is
   begin
      if N.Length = 0 then
         return False;
      end if;

      --  Check for valid identifier characters
      for I in 1 .. N.Length loop
         declare
            C : constant Character := N.Data (I);
         begin
            if not (C in 'a' .. 'z' | 'A' .. 'Z' | '0' .. '9' | '_' | '.') then
               return False;
            end if;
         end;
      end loop;

      return True;
   end Is_Valid_Module_Name;

   -------------------------------------------------------------------------
   --  Add_Error: Add an error message
   -------------------------------------------------------------------------
   procedure Add_Error (
      Result  : in Out Parse_Result;
      Message : Bounded_Name)
   is
   begin
      Result.Error_Count := Result.Error_Count + 1;
      Result.Errors (Result.Error_Count) := (
         Is_Error => True,
         Message  => Message
      );
      Result.Is_Valid := False;
   end Add_Error;

   -------------------------------------------------------------------------
   --  Add_Warning: Add a warning message
   -------------------------------------------------------------------------
   procedure Add_Warning (
      Result  : in Out Parse_Result;
      Message : Bounded_Name)
   is
   begin
      Result.Warning_Count := Result.Warning_Count + 1;
      Result.Warnings (Result.Warning_Count) := (
         Is_Error => False,
         Message  => Message
      );
   end Add_Warning;

   -------------------------------------------------------------------------
   --  Initialize: Initialize a parse result
   -------------------------------------------------------------------------
   procedure Initialize (Result : out Parse_Result) is
   begin
      Result.Is_Valid := True;  --  Assume valid until proven otherwise
      Result.Schema := Empty_Schema;
      Result.Schema_Kind := Unknown_Schema;
      Result.Module_Name := Empty_Name;
      Result.Has_Epoch := False;
      Result.Epoch := 0;
      Result.Has_Spec_Hash := False;
      Result.Function_Count := 0;
      Result.Type_Count := 0;
      Result.Error_Count := 0;
      Result.Warning_Count := 0;
      Result.Functions := (others => (others => <>));
      Result.Types := (others => (others => <>));
      Result.Errors := (others => (others => <>));
      Result.Warnings := (others => (others => <>));
   end Initialize;

   -------------------------------------------------------------------------
   --  Validate_Structure: Validate IR structure
   -------------------------------------------------------------------------
   procedure Validate_Structure (
      Result      : in Out Parse_Result;
      Has_Schema  : Boolean;
      Has_Module  : Boolean;
      Has_Functions : Boolean;
      Strict      : Boolean := False)
   is
      pragma Unreferenced (Has_Functions);
   begin
      --  Check schema field
      if not Has_Schema then
         if Result.Error_Count < Max_Errors then
            Add_Error (Result, Make_Name ("Missing 'schema' field"));
         end if;
      elsif not Is_Valid_Schema (Result.Schema) then
         if Result.Error_Count < Max_Errors then
            Add_Error (Result, Make_Name ("Invalid schema format"));
         end if;
      end if;

      --  Check module name
      if not Has_Module then
         if Result.Error_Count < Max_Errors then
            Add_Error (Result, Make_Name ("Missing 'ir_module' field"));
         end if;
      elsif Result.Module_Name.Length = 0 then
         if Result.Error_Count < Max_Errors then
            Add_Error (Result, Make_Name ("Empty ir_module"));
         end if;
      elsif not Is_Valid_Module_Name (Result.Module_Name) then
         if Result.Error_Count < Max_Errors then
            Add_Error (Result, Make_Name ("Invalid module name"));
         end if;
      end if;

      --  Strict mode checks
      if Strict then
         if not Result.Has_Epoch then
            if Result.Warning_Count < Max_Errors then
               Add_Warning (Result, Make_Name ("Missing optional field: ir_epoch"));
            end if;
         end if;

         if not Result.Has_Spec_Hash then
            if Result.Warning_Count < Max_Errors then
               Add_Warning (Result, Make_Name ("Missing optional field: ir_spec_hash"));
            end if;
         end if;
      end if;
   end Validate_Structure;

   -------------------------------------------------------------------------
   --  Add_Function: Add a function to the result
   -------------------------------------------------------------------------
   procedure Add_Function (
      Result    : in Out Parse_Result;
      Name      : Bounded_Name;
      Has_Body  : Boolean := False;
      Params    : Natural := 0)
   is
   begin
      Result.Function_Count := Result.Function_Count + 1;
      Result.Functions (Result.Function_Count) := (
         Name        => Name,
         Has_Body    => Has_Body,
         Param_Count => Params
      );

      --  Validate function has a name
      if Name.Length = 0 then
         if Result.Warning_Count < Max_Errors then
            Add_Warning (Result, Make_Name ("Function missing 'name'"));
         end if;
      end if;
   end Add_Function;

end IR_Parser;
