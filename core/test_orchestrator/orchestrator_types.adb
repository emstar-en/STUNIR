--  STUNIR Test Orchestrator Types - Implementation
--  SPARK Migration Phase 3

pragma SPARK_Mode (On);

package body Orchestrator_Types is

   --  ===========================================
   --  Initialize Tool Array
   --  ===========================================

   function Init_Tool_Array return Tool_Array is
      Result : Tool_Array;
   begin
      for I in Valid_Tool_Index loop
         Result (I) := Empty_Tool_Entry;
      end loop;
      return Result;
   end Init_Tool_Array;

   --  ===========================================
   --  Initialize Output Array
   --  ===========================================

   function Init_Output_Array return Output_Array is
      Result : Output_Array;
   begin
      for I in Valid_Tool_Index loop
         Result (I) := Empty_Output;
      end loop;
      return Result;
   end Init_Output_Array;

   --  ===========================================
   --  Empty Session
   --  ===========================================

   function Empty_Session return Orchestration_Session is
   begin
      return (
         Tools       => Init_Tool_Array,
         Outputs     => Init_Output_Array,
         Tool_Count  => 0,
         Stats       => Empty_Orch_Stats,
         Reference   => 0,
         Is_Complete => False
      );
   end Empty_Session;

end Orchestrator_Types;
