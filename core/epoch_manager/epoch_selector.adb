-------------------------------------------------------------------------------
--  STUNIR Epoch Selector - Ada SPARK Implementation
--  Part of Phase 2 SPARK Migration
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

package body Epoch_Selector is

   --  Check if environment variable exists
   --  Note: Actual implementation would use Ada.Environment_Variables
   function Has_Env_Variable (Name : String) return Boolean is
      pragma Unreferenced (Name);
   begin
      --  SPARK-safe stub: actual implementation outside SPARK
      return False;
   end Has_Env_Variable;

   --  Get environment variable value
   function Get_Env_Variable (Name : String) return Env_Value is
      Result : Env_Value := Empty_Env_Value;
      pragma Unreferenced (Name);
   begin
      --  SPARK-safe stub: actual implementation outside SPARK
      return Result;
   end Get_Env_Variable;

   --  Parse epoch value from string (numeric parsing)
   function Parse_Epoch_Value (S : String) return Epoch_Value is
      Result : Epoch_Value := 0;
      Digit  : Natural;
   begin
      for I in S'Range loop
         if S (I) in '0' .. '9' then
            Digit := Character'Pos (S (I)) - Character'Pos ('0');
            --  Overflow protection
            if Result <= (Epoch_Value'Last - Epoch_Value (Digit)) / 10 then
               Result := Result * 10 + Epoch_Value (Digit);
            else
               --  Overflow: return max safe value
               return Epoch_Value'Last;
            end if;
         else
            --  Non-digit: stop parsing
            exit;
         end if;
      end loop;
      return Result;
   end Parse_Epoch_Value;

   --  Compute spec directory digest (simplified)
   function Compute_Spec_Digest (Spec_Root : Path_String) return Hash_Hex is
      Result : Hash_Hex := Zero_Hash;
      pragma Unreferenced (Spec_Root);
   begin
      --  Simplified: would normally scan directory and compute SHA256
      --  Return a deterministic placeholder based on path
      Result.Data (1 .. 8) := "00000000";
      return Result;
   end Compute_Spec_Digest;

   --  Derive epoch from spec digest (first 8 hex chars)
   function Derive_Epoch_From_Digest (Digest : Hash_Hex) return Epoch_Value is
      Result : Epoch_Value := 0;
      Char   : Character;
      Value  : Natural;
   begin
      --  Use first 8 hex characters (32 bits)
      for I in 1 .. 8 loop
         Char := Digest.Data (I);
         if Char in '0' .. '9' then
            Value := Character'Pos (Char) - Character'Pos ('0');
         elsif Char in 'a' .. 'f' then
            Value := Character'Pos (Char) - Character'Pos ('a') + 10;
         elsif Char in 'A' .. 'F' then
            Value := Character'Pos (Char) - Character'Pos ('A') + 10;
         else
            Value := 0;
         end if;
         Result := Result * 16 + Epoch_Value (Value);
      end loop;
      return Result;
   end Derive_Epoch_From_Digest;

   --  Main epoch selection procedure
   procedure Select_Epoch (
      Spec_Root     : Path_String;
      Allow_Current : Boolean := False;
      Selection     : out Epoch_Selection)
   is
      Env_Val : Env_Value;
   begin
      --  Initialize with zero epoch
      Selection := Default_Epoch_Selection;

      --  Priority 1: STUNIR_BUILD_EPOCH
      if Has_Env_Variable ("STUNIR_BUILD_EPOCH") then
         Env_Val := Get_Env_Variable ("STUNIR_BUILD_EPOCH");
         if Env_Val.Exists and Env_Val.Length > 0 then
            Selection.Value := Parse_Epoch_Value (
               Env_Val.Data (1 .. Env_Val.Length));
            Selection.Source := Source_Env_Build_Epoch;
            Selection.Is_Deterministic := True;
            return;
         end if;
      end if;

      --  Priority 2: SOURCE_DATE_EPOCH
      if Has_Env_Variable ("SOURCE_DATE_EPOCH") then
         Env_Val := Get_Env_Variable ("SOURCE_DATE_EPOCH");
         if Env_Val.Exists and Env_Val.Length > 0 then
            Selection.Value := Parse_Epoch_Value (
               Env_Val.Data (1 .. Env_Val.Length));
            Selection.Source := Source_Env_Source_Date;
            Selection.Is_Deterministic := True;
            return;
         end if;
      end if;

      --  Priority 3: Derived from spec digest
      if Spec_Root.Length > 0 then
         Selection.Spec_Digest := Compute_Spec_Digest (Spec_Root);
         Selection.Value := Derive_Epoch_From_Digest (Selection.Spec_Digest);
         Selection.Source := Source_Derived_Spec_Digest;
         Selection.Is_Deterministic := True;
         return;
      end if;

      --  Priority 4: Zero (fallback)
      Selection.Value := Zero_Epoch_Value;
      Selection.Source := Source_Zero;
      Selection.Is_Deterministic := True;
      
      --  Note: Current time would only be used if Allow_Current = True
      --  and no other source available (not implemented for SPARK safety)
      pragma Unreferenced (Allow_Current);
   end Select_Epoch;

   --  Convert selection to JSON record
   function To_JSON (Selection : Epoch_Selection) return Epoch_JSON is
      Result : Epoch_JSON;
      Source_Str : constant String := Source_To_String (Selection.Source);
   begin
      Result.Selected_Epoch := Selection.Value;
      Result.Deterministic := Selection.Is_Deterministic;
      Result.Spec_Digest_Hex := Selection.Spec_Digest;
      
      --  Copy source string
      if Source_Str'Length <= Max_Medium_String then
         Result.Epoch_Source_Str.Length := Source_Str'Length;
         Result.Epoch_Source_Str.Data (1 .. Source_Str'Length) := Source_Str;
      end if;
      
      return Result;
   end To_JSON;

end Epoch_Selector;
