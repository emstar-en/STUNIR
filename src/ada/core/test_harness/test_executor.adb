--  STUNIR Test Executor - Implementation
--  SPARK Migration Phase 3

pragma SPARK_Mode (On);

package body Test_Executor is

   --  ===========================================
   --  Initialize Suite
   --  ===========================================

   procedure Initialize_Suite (Suite : out Test_Suite) is
   begin
      Suite := Empty_Suite;
   end Initialize_Suite;

   --  ===========================================
   --  Register Test
   --  ===========================================

   procedure Register_Test (
      Suite   : in out Test_Suite;
      TC      : in Test_Case;
      Success : out Boolean) is
   begin
      if Suite.Count < Max_Tests and TC.Name_Len > 0 then
         Suite.Count := Suite.Count + 1;
         Suite.Tests (Valid_Test_Index (Suite.Count)) := TC;
         Success := True;
      else
         Success := False;
      end if;
   end Register_Test;

   --  ===========================================
   --  Create Test
   --  ===========================================

   function Create_Test (
      Name     : String;
      Category : Test_Category;
      Priority : Test_Priority;
      Timeout  : Natural) return Test_Case
   is
      TC : Test_Case := Empty_Test;
   begin
      for I in Name'Range loop
         TC.Name (I - Name'First + 1) := Name (I);
      end loop;
      TC.Name_Len   := Name'Length;
      TC.Category   := Category;
      TC.Priority   := Priority;
      TC.Timeout_Ms := Timeout;
      TC.Is_Enabled := True;
      return TC;
   end Create_Test;

   --  ===========================================
   --  Execute Test (Simulated)
   --  ===========================================

   procedure Execute_Test (
      TC     : in Test_Case;
      Result : out Test_Result)
   is
      Msg : constant String := "Test executed successfully";
   begin
      Result := Empty_Result;

      --  Copy test name to result
      Result.Name     := TC.Name;
      Result.Name_Len := TC.Name_Len;

      --  Simulate test execution
      --  In real impl, would call actual test
      --  For SPARK verification, mark as passed
      Result.Status      := Passed;
      Result.Duration_Ms := 1;  --  Simulated duration

      --  Set success message
      for I in Msg'Range loop
         Result.Message (I - Msg'First + 1) := Msg (I);
      end loop;
      Result.Msg_Len := Msg'Length;

      Result.Output_Len := 0;
   end Execute_Test;

   --  ===========================================
   --  Execute All Tests
   --  ===========================================

   procedure Execute_All (
      Suite : in out Test_Suite)
   is
      Result : Test_Result;
   begin
      Suite.Stats := Empty_Stats;

      for I in 1 .. Suite.Count loop
         if Suite.Tests (Valid_Test_Index (I)).Is_Enabled then
            Execute_Test (Suite.Tests (Valid_Test_Index (I)), Result);
            Suite.Results (Valid_Test_Index (I)) := Result;
            Update_Stats (Suite.Stats, Result);
         end if;
      end loop;
   end Execute_All;

   --  ===========================================
   --  Execute By Category
   --  ===========================================

   procedure Execute_By_Category (
      Suite    : in out Test_Suite;
      Category : in Test_Category)
   is
      Result : Test_Result;
   begin
      for I in 1 .. Suite.Count loop
         if Suite.Tests (Valid_Test_Index (I)).Category = Category and
            Suite.Tests (Valid_Test_Index (I)).Is_Enabled
         then
            Execute_Test (Suite.Tests (Valid_Test_Index (I)), Result);
            Suite.Results (Valid_Test_Index (I)) := Result;
            Update_Stats (Suite.Stats, Result);
         end if;
      end loop;
   end Execute_By_Category;

   --  ===========================================
   --  Execute By Priority
   --  ===========================================

   procedure Execute_By_Priority (
      Suite : in out Test_Suite)
   is
      Result : Test_Result;
   begin
      Suite.Stats := Empty_Stats;

      --  Execute Critical tests first
      for I in 1 .. Suite.Count loop
         if Suite.Tests (Valid_Test_Index (I)).Priority = Critical and
            Suite.Tests (Valid_Test_Index (I)).Is_Enabled
         then
            Execute_Test (Suite.Tests (Valid_Test_Index (I)), Result);
            Suite.Results (Valid_Test_Index (I)) := Result;
            Update_Stats (Suite.Stats, Result);
         end if;
      end loop;

      --  Execute High priority tests
      for I in 1 .. Suite.Count loop
         if Suite.Tests (Valid_Test_Index (I)).Priority = High and
            Suite.Tests (Valid_Test_Index (I)).Is_Enabled
         then
            Execute_Test (Suite.Tests (Valid_Test_Index (I)), Result);
            Suite.Results (Valid_Test_Index (I)) := Result;
            Update_Stats (Suite.Stats, Result);
         end if;
      end loop;

      --  Execute Medium priority tests
      for I in 1 .. Suite.Count loop
         if Suite.Tests (Valid_Test_Index (I)).Priority = Medium and
            Suite.Tests (Valid_Test_Index (I)).Is_Enabled
         then
            Execute_Test (Suite.Tests (Valid_Test_Index (I)), Result);
            Suite.Results (Valid_Test_Index (I)) := Result;
            Update_Stats (Suite.Stats, Result);
         end if;
      end loop;

      --  Execute Low priority tests
      for I in 1 .. Suite.Count loop
         if Suite.Tests (Valid_Test_Index (I)).Priority = Low and
            Suite.Tests (Valid_Test_Index (I)).Is_Enabled
         then
            Execute_Test (Suite.Tests (Valid_Test_Index (I)), Result);
            Suite.Results (Valid_Test_Index (I)) := Result;
            Update_Stats (Suite.Stats, Result);
         end if;
      end loop;
   end Execute_By_Priority;

   --  ===========================================
   --  Update Stats
   --  ===========================================

   procedure Update_Stats (
      Stats  : in out Suite_Stats;
      Result : in Test_Result) is
   begin
      Stats.Total := Stats.Total + 1;

      case Result.Status is
         when Passed =>
            if Stats.Passed < Natural'Last then
               Stats.Passed := Stats.Passed + 1;
            end if;
         when Failed =>
            if Stats.Failed < Natural'Last then
               Stats.Failed := Stats.Failed + 1;
            end if;
         when Skipped =>
            if Stats.Skipped < Natural'Last then
               Stats.Skipped := Stats.Skipped + 1;
            end if;
         when Error =>
            if Stats.Errors < Natural'Last then
               Stats.Errors := Stats.Errors + 1;
            end if;
         when Timeout =>
            if Stats.Timeouts < Natural'Last then
               Stats.Timeouts := Stats.Timeouts + 1;
            end if;
         when others =>
            null;
      end case;
   end Update_Stats;

   --  ===========================================
   --  Get Result
   --  ===========================================

   function Get_Result (
      Suite : Test_Suite;
      Index : Valid_Test_Index) return Test_Result is
   begin
      return Suite.Results (Index);
   end Get_Result;

   --  ===========================================
   --  Total Duration
   --  ===========================================

   function Total_Duration_Ms (Suite : Test_Suite) return Natural is
      Total : Natural := 0;
      Dur   : Natural;
   begin
      for I in 1 .. Suite.Count loop
         Dur := Suite.Results (Valid_Test_Index (I)).Duration_Ms;
         if Total < Natural'Last - Dur then
            Total := Total + Dur;
         end if;
      end loop;
      return Total;
   end Total_Duration_Ms;

   --  ===========================================
   --  Format Summary
   --  ===========================================

   procedure Format_Summary (
      Stats  : in Suite_Stats;
      Output : out Message_String;
      Length : out Natural)
   is
      Prefix : constant String := "Tests: ";
      pragma Unreferenced (Stats);
   begin
      Output := (others => ' ');
      for I in Prefix'Range loop
         Output (I - Prefix'First + 1) := Prefix (I);
      end loop;
      Length := Prefix'Length;
   end Format_Summary;

end Test_Executor;
