--  Test: Protected Types and Tasks
--  Purpose: Test extraction of protected types and task declarations
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

package Protected_Types is

   --  Protected type declaration
   protected type Counter is
      procedure Increment;
      procedure Decrement;
      function Get_Value return Integer;
      procedure Reset;
   private
      Value : Integer := 0;
   end Counter;

   --  Protected object
   protected Shared_State is
      procedure Set (New_Value : in Integer);
      function Get return Integer;
      entry Wait_For_Zero;
   private
      State : Integer := 0;
   end Shared_State;

   --  Task type declaration
   task type Worker is
      entry Start (Id : in Integer);
      entry Stop;
   end Worker;

   --  Single task declaration
   task Background_Task;

   --  Regular subprograms for comparison
   procedure Initialize_Counter (C : out Counter);
   function Create_Worker (Id : Integer) return Worker;

end Protected_Types;
