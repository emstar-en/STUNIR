--  STUNIR Test Harness Types - Implementation
--  SPARK Migration Phase 3

pragma SPARK_Mode (On);

package body Test_Harness_Types is

   --  ===========================================
   --  Init Test Queue
   --  ===========================================

   function Init_Test_Queue return Test_Queue is
      Result : Test_Queue;
   begin
      for I in Valid_Test_Index loop
         Result (I) := Empty_Test;
      end loop;
      return Result;
   end Init_Test_Queue;

   --  ===========================================
   --  Init Result Queue
   --  ===========================================

   function Init_Result_Queue return Result_Queue is
      Result : Result_Queue;
   begin
      for I in Valid_Test_Index loop
         Result (I) := Empty_Result;
      end loop;
      return Result;
   end Init_Result_Queue;

   --  ===========================================
   --  Empty Suite
   --  ===========================================

   function Empty_Suite return Test_Suite is
   begin
      return (
         Tests   => Init_Test_Queue,
         Results => Init_Result_Queue,
         Count   => 0,
         Stats   => Empty_Stats
      );
   end Empty_Suite;

   --  ===========================================
   --  Get Success Rate
   --  ===========================================

   function Get_Success_Rate (Stats : Suite_Stats) return Natural is
      Rate : Natural;
   begin
      if Stats.Passed = 0 then
         Rate := 0;
      elsif Stats.Passed >= Stats.Total then
         Rate := 100;
      else
         Rate := (Stats.Passed * 100) / Stats.Total;
      end if;
      return Rate;
   end Get_Success_Rate;

end Test_Harness_Types;
