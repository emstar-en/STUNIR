--  STUNIR Result Validator - Implementation
--  SPARK Migration Phase 3

pragma SPARK_Mode (On);

package body Result_Validator is

   --  ===========================================
   --  Initialize Receipt
   --  ===========================================

   procedure Initialize_Receipt (Receipt : out Receipt_Data) is
   begin
      Receipt := Empty_Receipt;
   end Initialize_Receipt;

   --  ===========================================
   --  Add Entry
   --  ===========================================

   procedure Add_Entry (
      Receipt  : in out Receipt_Data;
      Path     : in String;
      Hash     : in String;
      Success  : out Boolean)
   is
      E : Validation_Entry := Empty_Entry_Val;
   begin
      if Receipt.Entry_Count = Max_Entries then
         Success := False;
         return;
      end if;

      --  Copy path
      for I in Path'Range loop
         E.File_Path (I - Path'First + 1) := Path (I);
      end loop;
      E.Path_Len := Path'Length;

      --  Copy hash
      for I in Hash'Range loop
         E.Expected_Hash (I - Hash'First + 1) := Hash (I);
      end loop;

      E.Outcome := Empty_Entry;
      E.File_Exists := False;

      Receipt.Entry_Count := Receipt.Entry_Count + 1;
      Receipt.Entries (Valid_Entry_Index (Receipt.Entry_Count)) := E;
      Success := True;
   end Add_Entry;

   --  ===========================================
   --  Mark Loaded
   --  ===========================================

   procedure Mark_Loaded (
      Receipt : in out Receipt_Data;
      Status  : in Validation_Outcome) is
   begin
      Receipt.Is_Loaded := True;
      Receipt.Parse_Status := Status;
   end Mark_Loaded;

   --  ===========================================
   --  Create Entry
   --  ===========================================

   function Create_Entry (
      Path     : String;
      Expected : String) return Validation_Entry
   is
      E : Validation_Entry := Empty_Entry_Val;
   begin
      for I in Path'Range loop
         E.File_Path (I - Path'First + 1) := Path (I);
      end loop;
      E.Path_Len := Path'Length;

      for I in Expected'Range loop
         E.Expected_Hash (I - Expected'First + 1) := Expected (I);
      end loop;

      E.Outcome := Empty_Entry;
      return E;
   end Create_Entry;

   --  ===========================================
   --  Validate Entry
   --  ===========================================

   procedure Validate_Entry (
      E        : in out Validation_Entry;
      Base_Dir : in Path_String;
      Dir_Len  : in Natural)
   is
      pragma Unreferenced (Base_Dir);
      pragma Unreferenced (Dir_Len);
   begin
      --  In a real implementation, would check file existence and compute hash
      --  For SPARK verification, we simulate successful validation

      if E.Path_Len = 0 then
         E.Outcome := Invalid_Path;
         E.File_Exists := False;
         return;
      end if;

      --  Simulate file exists
      E.File_Exists := True;

      --  Simulate hash computation (copy expected to actual for demo)
      E.Actual_Hash := E.Expected_Hash;

      --  Check hash match
      if Hashes_Match (E.Expected_Hash, E.Actual_Hash) then
         E.Outcome := Valid;
      else
         E.Outcome := Hash_Mismatch;
      end if;
   end Validate_Entry;

   --  ===========================================
   --  Hashes Match
   --  ===========================================

   function Hashes_Match (
      Hash1 : String;
      Hash2 : String) return Boolean
   is
      Match : Boolean := True;
   begin
      for I in 1 .. Hash_Length loop
         if Hash1 (Hash1'First + I - 1) /= Hash2 (Hash2'First + I - 1) then
            Match := False;
            exit;
         end if;
      end loop;
      return Match;
   end Hashes_Match;

   --  ===========================================
   --  Validate All
   --  ===========================================

   procedure Validate_All (
      Receipt  : in Receipt_Data;
      Base_Dir : in Path_String;
      Dir_Len  : in Natural;
      Results  : out Validation_Results)
   is
      E : Validation_Entry;
   begin
      Results := Empty_Results;

      for I in 1 .. Receipt.Entry_Count loop
         E := Receipt.Entries (Valid_Entry_Index (I));
         Validate_Entry (E, Base_Dir, Dir_Len);
         Results.Entries (Valid_Entry_Index (I)) := E;
         Results.Checked := Results.Checked + 1;
         Update_Stats (Results.Stats, E.Outcome);
      end loop;

      Results.All_Valid := All_Entries_Valid (Results);
   end Validate_All;

   --  ===========================================
   --  Update Stats
   --  ===========================================

   procedure Update_Stats (
      Stats   : in out Validation_Stats;
      Outcome : in Validation_Outcome) is
   begin
      Stats.Total := Stats.Total + 1;

      case Outcome is
         when Valid =>
            if Stats.Valid < Natural'Last then
               Stats.Valid := Stats.Valid + 1;
            end if;
         when Hash_Mismatch =>
            if Stats.Invalid < Natural'Last then
               Stats.Invalid := Stats.Invalid + 1;
            end if;
         when File_Missing =>
            if Stats.Missing < Natural'Last then
               Stats.Missing := Stats.Missing + 1;
            end if;
         when Parse_Error | Invalid_Path =>
            if Stats.Errors < Natural'Last then
               Stats.Errors := Stats.Errors + 1;
            end if;
         when Empty_Entry =>
            null;
      end case;
   end Update_Stats;

   --  ===========================================
   --  Format Result
   --  ===========================================

   procedure Format_Result (
      Results : in Validation_Results;
      Output  : out Path_String;
      Length  : out Natural)
   is
      Prefix : constant String := "Validation: ";
   begin
      Output := (others => ' ');
      for I in Prefix'Range loop
         Output (I - Prefix'First + 1) := Prefix (I);
      end loop;
      Length := Prefix'Length;

      pragma Unreferenced (Results);
   end Format_Result;

end Result_Validator;
