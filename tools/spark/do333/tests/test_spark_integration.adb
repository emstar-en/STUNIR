--  STUNIR DO-333 SPARK Integration Tests
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

with Ada.Text_IO;       use Ada.Text_IO;
with SPARK_Integration; use SPARK_Integration;
with GNATprove_Wrapper; use GNATprove_Wrapper;

procedure Test_SPARK_Integration is

   Total_Tests  : Natural := 0;
   Passed_Tests : Natural := 0;

   procedure Test (Name : String; Condition : Boolean) is
   begin
      Total_Tests := Total_Tests + 1;
      if Condition then
         Passed_Tests := Passed_Tests + 1;
         Put_Line ("  [PASS] " & Name);
      else
         Put_Line ("  [FAIL] " & Name);
      end if;
   end Test;

begin
   Put_Line ("=== SPARK Integration Tests ===");
   New_Line;

   --  Test 1: Default configuration
   declare
      Cfg : constant SPARK_Config := Default_Config;
   begin
      Test ("Default mode is Mode_All", Cfg.Mode = Mode_All);
      Test ("Default prover is Prover_All", Cfg.Prover = Prover_All);
      Test ("Default level is 2", Cfg.Level = 2);
      Test ("Default timeout is 60", Cfg.Timeout = 60);
      Test ("Default config is valid", Is_Valid_Config (Cfg));
   end;

   --  Test 2: High assurance configuration
   declare
      Cfg : constant SPARK_Config := High_Assurance_Config;
   begin
      Test ("HA mode is Mode_All", Cfg.Mode = Mode_All);
      Test ("HA level is 4", Cfg.Level = 4);
      Test ("HA timeout is 120", Cfg.Timeout = 120);
      Test ("HA replay is True", Cfg.Replay);
      Test ("HA config is valid", Is_Valid_Config (Cfg));
   end;

   --  Test 3: Quick configuration
   declare
      Cfg : constant SPARK_Config := Quick_Config;
   begin
      Test ("Quick mode is Mode_Flow", Cfg.Mode = Mode_Flow);
      Test ("Quick prover is Z3", Cfg.Prover = Prover_Z3);
      Test ("Quick level is 0", Cfg.Level = 0);
   end;

   --  Test 4: Mode names
   Test ("Mode_Flow name", Mode_Name (Mode_Flow) = "flow");
   Test ("Mode_Prove name", Mode_Name (Mode_Prove) = "prove");
   Test ("Mode_All name", Mode_Name (Mode_All) = "all");

   --  Test 5: Prover names
   Test ("Prover_Z3 name", Prover_Name (Prover_Z3) = "z3");
   Test ("Prover_CVC4 name", Prover_Name (Prover_CVC4) = "cvc4");
   Test ("Prover_Alt_Ergo name", Prover_Name (Prover_Alt_Ergo) = "altergo");

   --  Test 6: Tool availability
   Test ("GNATprove availability check", Is_GNATprove_Available);
   Test ("Z3 availability check", Is_Prover_Available (Prover_Z3));
   Test ("CVC4 availability check", Is_Prover_Available (Prover_CVC4));

   --  Test 7: Empty result
   declare
      R : constant SPARK_Result := Empty_Result;
   begin
      Test ("Empty result not successful", not R.Success);
      Test ("Empty result exit code is -1", R.Exit_Code = -1);
      Test ("Empty result total VCs is 0", R.Total_VCs = 0);
      Test ("Empty result not all proved", not All_Proved (R));
   end;

   --  Test 8: Result analysis
   declare
      R : SPARK_Result := Empty_Result;
   begin
      R.Success := True;
      R.Total_VCs := 100;
      R.Proved_VCs := 95;
      R.Unproved_VCs := 5;
      R.Flow_Errors := 0;

      Test ("Proof percentage is 95", Proof_Percentage (R) = 95);
      Test ("Flow passed", Flow_Passed (R));
      Test ("Not all proved (5 unproved)", not All_Proved (R));
   end;

   --  Test 9: Perfect result
   declare
      R : SPARK_Result := Empty_Result;
   begin
      R.Success := True;
      R.Total_VCs := 50;
      R.Proved_VCs := 50;
      R.Unproved_VCs := 0;
      R.Flow_Errors := 0;

      Test ("All proved", All_Proved (R));
      Test ("Proof percentage is 100", Proof_Percentage (R) = 100);
   end;

   --  Test 10: Build command
   declare
      Cmd    : Command_String;
      Length : Command_Length;
      Cfg    : constant SPARK_Config := Default_Config;
   begin
      Build_Command ("test.gpr", Cfg, Cmd, Length);
      Test ("Command length > 0", Length > 0);
      --  Check command contains gnatprove
      Test ("Command starts with gnatprove",
            Length >= 9 and then Cmd (1 .. 9) = "gnatprove");
   end;

   --  Test 11: Flow command
   declare
      Cmd    : Command_String;
      Length : Command_Length;
   begin
      Build_Flow_Command ("myproject.gpr", Cmd, Length);
      Test ("Flow command length > 0", Length > 0);
   end;

   --  Test 12: Output buffer
   declare
      Buffer  : Output_Buffer;
      Success : Boolean;
   begin
      Clear_Buffer (Buffer);
      Test ("Cleared buffer count is 0", Buffer.Count = 0);

      Add_Line (Buffer, "test line 1", Success);
      Test ("Add line succeeds", Success);
      Test ("Buffer count is 1", Buffer.Count = 1);

      Add_Line (Buffer, "test line 2", Success);
      Test ("Buffer count is 2", Buffer.Count = 2);
   end;

   --  Test 13: Contains pattern
   declare
      Buffer  : Output_Buffer;
      Success : Boolean;
   begin
      Clear_Buffer (Buffer);
      Add_Line (Buffer, "Summary: total: 100, proved: 95", Success);
      Add_Line (Buffer, "error: some error message", Success);

      Test ("Contains 'total:'", Contains_Pattern (Buffer, "total:"));
      Test ("Contains 'error:'", Contains_Pattern (Buffer, "error:"));
      Test ("Does not contain 'foobar'",
            not Contains_Pattern (Buffer, "foobar"));
   end;

   --  Summary
   New_Line;
   Put_Line ("=== Test Summary ===");
   Put_Line ("Passed:" & Natural'Image (Passed_Tests) &
             " /" & Natural'Image (Total_Tests));

   if Passed_Tests = Total_Tests then
      Put_Line ("All tests PASSED!");
   else
      Put_Line ("Some tests FAILED!");
   end if;

end Test_SPARK_Integration;
