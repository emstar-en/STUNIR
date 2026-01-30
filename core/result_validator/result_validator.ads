--  STUNIR Result Validator
--  SPARK Migration Phase 3 - Test Infrastructure
--  Validates build artifacts against receipts

pragma SPARK_Mode (On);

with Validator_Types; use Validator_Types;
with Stunir_Hashes; use Stunir_Hashes;

package Result_Validator is

   --  ===========================================
   --  Receipt Loading
   --  ===========================================

   --  Initialize empty receipt
   procedure Initialize_Receipt (Receipt : out Receipt_Data)
      with Post => Receipt.Entry_Count = 0 and not Receipt.Is_Loaded;

   --  Add entry to receipt (for testing/simulation)
   procedure Add_Entry (
      Receipt  : in out Receipt_Data;
      Path     : in String;
      Hash     : in String;
      Success  : out Boolean)
      with Pre  => Receipt.Entry_Count < Max_Entries and
                   Path'Length > 0 and Path'Length <= Max_Path_Length and
                   Hash'Length = Hash_Length,
           Post => (if Success then
                       Receipt.Entry_Count = Receipt.Entry_Count'Old + 1
                    else Receipt.Entry_Count = Receipt.Entry_Count'Old);

   --  Mark receipt as loaded
   procedure Mark_Loaded (
      Receipt : in out Receipt_Data;
      Status  : in Validation_Outcome)
      with Post => Receipt.Is_Loaded and Receipt.Parse_Status = Status;

   --  ===========================================
   --  Single Entry Validation
   --  ===========================================

   --  Create validation entry
   function Create_Entry (
      Path     : String;
      Expected : String) return Validation_Entry
      with Pre  => Path'Length > 0 and Path'Length <= Max_Path_Length and
                   Expected'Length = Hash_Length,
           Post => Create_Entry'Result.Path_Len = Path'Length;

   --  Validate single entry (check file and hash)
   procedure Validate_Entry (
      E        : in out Validation_Entry;
      Base_Dir : in Path_String;
      Dir_Len  : in Natural)
      with Pre  => E.Path_Len > 0 and Dir_Len <= Max_Path_Length,
           Post => E.Outcome /= Empty_Entry;

   --  Compare two hashes
   function Hashes_Match (
      Hash1 : String;
      Hash2 : String) return Boolean
      with Pre => Hash1'Length = Hash_Length and Hash2'Length = Hash_Length;

   --  ===========================================
   --  Batch Validation
   --  ===========================================

   --  Validate all entries in receipt
   procedure Validate_All (
      Receipt  : in Receipt_Data;
      Base_Dir : in Path_String;
      Dir_Len  : in Natural;
      Results  : out Validation_Results)
      with Pre  => Receipt.Is_Loaded and Dir_Len <= Max_Path_Length,
           Post => Results.Checked = Receipt.Entry_Count;

   --  Update validation statistics
   procedure Update_Stats (
      Stats   : in out Validation_Stats;
      Outcome : in Validation_Outcome)
      with Pre => Stats.Total < Natural'Last,
           Post => Stats.Total = Stats.Total'Old + 1;

   --  ===========================================
   --  Reporting
   --  ===========================================

   --  Check if all entries are valid
   function All_Entries_Valid (Results : Validation_Results) return Boolean is
      (Results.Stats.Invalid = 0 and Results.Stats.Missing = 0 and
       Results.Stats.Errors = 0);

   --  Get count of failed entries
   function Failed_Count (Results : Validation_Results) return Natural is
      (Results.Stats.Invalid + Results.Stats.Missing + Results.Stats.Errors);

   --  Format result summary
   procedure Format_Result (
      Results : in Validation_Results;
      Output  : out Path_String;
      Length  : out Natural)
      with Post => Length <= Max_Path_Length;

end Result_Validator;
