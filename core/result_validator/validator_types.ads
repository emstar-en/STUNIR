--  STUNIR Result Validator Types
--  SPARK Migration Phase 3 - Test Infrastructure
--  Validation types and contracts for receipt verification

pragma SPARK_Mode (On);

with Stunir_Hashes; use Stunir_Hashes;

package Validator_Types is

   --  ===========================================
   --  Constants
   --  ===========================================

   Max_Entries : constant := 128;
   Max_Path_Length : constant := 256;

   --  ===========================================
   --  Validation Outcome
   --  ===========================================

   type Validation_Outcome is (
      Valid,           --  Hash matches
      Hash_Mismatch,   --  File exists but hash differs
      File_Missing,    --  File not found
      Parse_Error,     --  Receipt parsing failed
      Invalid_Path,    --  Path validation failed
      Empty_Entry      --  Entry is empty
   );

   --  ===========================================
   --  Path String Type
   --  ===========================================

   subtype Path_String is String (1 .. Max_Path_Length);

   --  ===========================================
   --  Validation Entry
   --  ===========================================

   type Validation_Entry is record
      File_Path     : Path_String;
      Path_Len      : Natural;
      Expected_Hash : String (1 .. Hash_Length);
      Actual_Hash   : String (1 .. Hash_Length);
      Outcome       : Validation_Outcome;
      File_Exists   : Boolean;
   end record;

   --  Empty entry constant
   Empty_Entry_Val : constant Validation_Entry := (
      File_Path     => (others => ' '),
      Path_Len      => 0,
      Expected_Hash => (others => '0'),
      Actual_Hash   => (others => '0'),
      Outcome       => Empty_Entry,
      File_Exists   => False
   );

   --  ===========================================
   --  Entry Array
   --  ===========================================

   type Entry_Index is range 0 .. Max_Entries;
   subtype Valid_Entry_Index is Entry_Index range 1 .. Max_Entries;

   type Entry_Array is array (Valid_Entry_Index) of Validation_Entry;

   --  ===========================================
   --  Receipt Data
   --  ===========================================

   type Receipt_Data is record
      Entries      : Entry_Array;
      Entry_Count  : Entry_Index;
      Is_Loaded    : Boolean;
      Parse_Status : Validation_Outcome;
   end record;

   --  ===========================================
   --  Validation Statistics
   --  ===========================================

   type Validation_Stats is record
      Total    : Natural := 0;
      Valid    : Natural := 0;
      Invalid  : Natural := 0;
      Missing  : Natural := 0;
      Errors   : Natural := 0;
   end record;

   --  Empty stats constant
   Empty_Validation_Stats : constant Validation_Stats := (
      Total   => 0,
      Valid   => 0,
      Invalid => 0,
      Missing => 0,
      Errors  => 0
   );

   --  ===========================================
   --  Validation Results
   --  ===========================================

   type Validation_Results is record
      Entries   : Entry_Array;
      Checked   : Entry_Index;
      Stats     : Validation_Stats;
      All_Valid : Boolean;
   end record;

   --  Init functions
   function Init_Entry_Array return Entry_Array;
   function Empty_Receipt return Receipt_Data;
   function Empty_Results return Validation_Results;

   --  ===========================================
   --  Helper Functions
   --  ===========================================

   function Is_Success_Outcome (O : Validation_Outcome) return Boolean is
      (O = Valid);

   function Is_Error_Outcome (O : Validation_Outcome) return Boolean is
      (O in Hash_Mismatch | File_Missing | Parse_Error | Invalid_Path);

   function Get_Validation_Rate (Stats : Validation_Stats) return Natural
      with Pre => Stats.Total > 0,
           Post => Get_Validation_Rate'Result <= 100;

end Validator_Types;
