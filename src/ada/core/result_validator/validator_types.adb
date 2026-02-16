--  STUNIR Result Validator Types - Implementation
--  SPARK Migration Phase 3

pragma SPARK_Mode (On);

package body Validator_Types is

   --  ===========================================
   --  Init Entry Array
   --  ===========================================

   function Init_Entry_Array return Entry_Array is
      Result : Entry_Array;
   begin
      for I in Valid_Entry_Index loop
         Result (I) := Empty_Entry_Val;
      end loop;
      return Result;
   end Init_Entry_Array;

   --  ===========================================
   --  Empty Receipt
   --  ===========================================

   function Empty_Receipt return Receipt_Data is
   begin
      return (
         Entries      => Init_Entry_Array,
         Entry_Count  => 0,
         Is_Loaded    => False,
         Parse_Status => Empty_Entry
      );
   end Empty_Receipt;

   --  ===========================================
   --  Empty Results
   --  ===========================================

   function Empty_Results return Validation_Results is
   begin
      return (
         Entries   => Init_Entry_Array,
         Checked   => 0,
         Stats     => Empty_Validation_Stats,
         All_Valid => True
      );
   end Empty_Results;

   --  ===========================================
   --  Get Validation Rate
   --  ===========================================

   function Get_Validation_Rate (Stats : Validation_Stats) return Natural is
      Rate : Natural;
   begin
      if Stats.Valid = 0 then
         Rate := 0;
      elsif Stats.Valid >= Stats.Total then
         Rate := 100;
      else
         Rate := (Stats.Valid * 100) / Stats.Total;
      end if;
      return Rate;
   end Get_Validation_Rate;

end Validator_Types;
