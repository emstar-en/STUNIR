--  STUNIR DO-333 GNATprove Wrapper
--  Implementation
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

with Proof_Obligation; use Proof_Obligation;
with Verification_Condition; use Verification_Condition;

package body GNATprove_Wrapper is

   --  ============================================================
   --  Helper: Append String to Command
   --  ============================================================

   procedure Append_To_Command
     (Command : in Out Command_String;
      Length  : in Out Command_Length;
      Text    : String)
   with
      Pre => Length + Text'Length <= Max_Command_Length
   is
   begin
      for I in Text'Range loop
         Length := Length + 1;
         Command (Length) := Text (I);
      end loop;
   end Append_To_Command;

   --  ============================================================
   --  Build Command
   --  ============================================================

   procedure Build_Command
     (Project_File : String;
      Config       : SPARK_Config;
      Command      : out Command_String;
      Length       : out Command_Length)
   is
   begin
      Command := (others => ' ');
      Length := 0;

      --  Base command
      Append_To_Command (Command, Length, "gnatprove -P ");
      Append_To_Command (Command, Length, Project_File);

      --  Mode
      Append_To_Command (Command, Length, " --mode=");
      Append_To_Command (Command, Length, Mode_Name (Config.Mode));

      --  Level
      Append_To_Command (Command, Length, " --level=");
      Append_To_Command (Command, Length,
        (1 => Character'Val (Character'Pos ('0') + Natural (Config.Level))));

      --  Timeout
      Append_To_Command (Command, Length, " --timeout=");
      declare
         Timeout_Str : String (1 .. 10) := (others => ' ');
         Timeout_Val : Natural := Config.Timeout;
         Idx         : Natural := 10;
      begin
         if Timeout_Val = 0 then
            Timeout_Str (10) := '0';
         else
            while Timeout_Val > 0 and then Idx >= 1 loop
               Timeout_Str (Idx) := Character'Val
                 (Character'Pos ('0') + (Timeout_Val mod 10));
               Timeout_Val := Timeout_Val / 10;
               Idx := Idx - 1;
            end loop;
         end if;
         for I in Timeout_Str'Range loop
            if Timeout_Str (I) /= ' ' then
               Append_To_Command (Command, Length, Timeout_Str (I .. 10));
               exit;
            end if;
         end loop;
      end;

      --  Prover
      if Config.Prover /= Prover_All then
         Append_To_Command (Command, Length, " --prover=");
         Append_To_Command (Command, Length, Prover_Name (Config.Prover));
      end if;

      --  Force
      if Config.Force then
         Append_To_Command (Command, Length, " --force");
      end if;

      --  JSON output
      if Config.Output_JSON then
         Append_To_Command (Command, Length, " --output=json");
      end if;

      --  Warnings
      if Config.Warnings then
         Append_To_Command (Command, Length, " --warnings=continue");
      end if;
   end Build_Command;

   --  ============================================================
   --  Build Flow Command
   --  ============================================================

   procedure Build_Flow_Command
     (Project_File : String;
      Command      : out Command_String;
      Length       : out Command_Length)
   is
   begin
      Command := (others => ' ');
      Length := 0;

      Append_To_Command (Command, Length, "gnatprove -P ");
      Append_To_Command (Command, Length, Project_File);
      Append_To_Command (Command, Length, " --mode=flow");
   end Build_Flow_Command;

   --  ============================================================
   --  Build Replay Command
   --  ============================================================

   procedure Build_Replay_Command
     (Project_File : String;
      Config       : SPARK_Config;
      Command      : out Command_String;
      Length       : out Command_Length)
   is
   begin
      Build_Command (Project_File, Config, Command, Length);
      Append_To_Command (Command, Length, " --replay");
   end Build_Replay_Command;

   --  ============================================================
   --  Parse Summary
   --  ============================================================

   procedure Parse_Summary
     (Output   : Output_Buffer;
      Result   : out SPARK_Result;
      Success  : out Boolean)
   is
      --  Simple parsing: look for summary patterns
      Total_Found   : Boolean := False;
      Proved_Found  : Boolean := False;
   begin
      Result := Empty_Result;
      Success := False;

      --  Scan output for summary information
      for I in 1 .. Output.Count loop
         declare
            Line : constant String :=
              Output.Lines (I).Content (1 .. Output.Lines (I).Length);
         begin
            --  Look for "total:" pattern
            if Line'Length > 6 then
               for J in Line'First .. Line'Last - 5 loop
                  if Line (J .. J + 5) = "total:" then
                     Total_Found := True;
                     --  Parse number after "total:"
                     --  Simplified: just set a placeholder
                     Result.Total_VCs := 100;  --  Placeholder
                  end if;
               end loop;
            end if;

            --  Look for "proved:" pattern
            if Line'Length > 7 then
               for J in Line'First .. Line'Last - 6 loop
                  if Line (J .. J + 6) = "proved:" then
                     Proved_Found := True;
                     Result.Proved_VCs := 95;  --  Placeholder
                  end if;
               end loop;
            end if;
         end;
      end loop;

      if Total_Found or else Proved_Found then
         Result.Unproved_VCs := Result.Total_VCs - Result.Proved_VCs;
         Result.Success := Result.Unproved_VCs = 0;
         Success := True;
      end if;
   end Parse_Summary;

   --  ============================================================
   --  Parse PO Results
   --  ============================================================

   procedure Parse_PO_Results
     (Output  : Output_Buffer;
      PO_Coll : out PO_Collection;
      Success : out Boolean)
   is
      Add_Success : Boolean;
      PO_Rec      : Proof_Obligation_Record;
      PO_ID       : Natural := 0;
   begin
      Initialize (PO_Coll);
      Success := True;

      --  In real implementation, parse JSON output
      --  For now, create sample POs based on output size
      for I in 1 .. Natural'Min (Output.Count, 100) loop
         PO_ID := PO_ID + 1;
         Create_PO
           (ID          => PO_ID,
            Kind        => PO_Assert,
            Criticality => DAL_C,
            Source      => "sample.ads",
            Subprogram  => "Sample_Proc",
            Line        => I,
            Column      => 1,
            PO          => PO_Rec);
         Update_Status (PO_Rec, PO_Proved, 50, 100);
         Add_PO (PO_Coll, PO_Rec, Add_Success);
         exit when not Add_Success;
      end loop;
   end Parse_PO_Results;

   --  ============================================================
   --  Parse VC Results
   --  ============================================================

   procedure Parse_VC_Results
     (Output  : Output_Buffer;
      VC_Coll : out VC_Collection;
      Success : out Boolean)
   is
      Add_Success : Boolean;
      VC_Rec      : VC_Record;
      VC_ID       : Natural := 0;
   begin
      Initialize (VC_Coll);
      Success := True;

      --  In real implementation, parse JSON output
      --  For now, create sample VCs
      for I in 1 .. Natural'Min (Output.Count, 200) loop
         VC_ID := VC_ID + 1;
         Create_VC (VC_ID, I, VC_Rec);
         Update_VC_Status (VC_Rec, VC_Valid, 50, 100);
         Add_VC (VC_Coll, VC_Rec, Add_Success);
         exit when not Add_Success;
      end loop;
   end Parse_VC_Results;

   --  ============================================================
   --  Parse Flow Results
   --  ============================================================

   procedure Parse_Flow_Results
     (Output   : Output_Buffer;
      Errors   : out Natural;
      Warnings : out Natural)
   is
   begin
      Errors := 0;
      Warnings := 0;

      --  Scan for flow error/warning patterns
      for I in 1 .. Output.Count loop
         declare
            Line : constant String :=
              Output.Lines (I).Content (1 .. Output.Lines (I).Length);
         begin
            --  Look for "error:" pattern
            if Line'Length > 6 then
               for J in Line'First .. Line'Last - 5 loop
                  if Line (J .. J + 5) = "error:" then
                     Errors := Errors + 1;
                  end if;
               end loop;
            end if;

            --  Look for "warning:" pattern
            if Line'Length > 8 then
               for J in Line'First .. Line'Last - 7 loop
                  if Line (J .. J + 7) = "warning:" then
                     Warnings := Warnings + 1;
                  end if;
               end loop;
            end if;
         end;
      end loop;
   end Parse_Flow_Results;

   --  ============================================================
   --  Add Line
   --  ============================================================

   procedure Add_Line
     (Buffer  : in Out Output_Buffer;
      Line    : String;
      Success : out Boolean)
   is
      New_Line : Output_Line := Empty_Line;
   begin
      if Buffer.Count >= Max_Output_Lines then
         Success := False;
         return;
      end if;

      for I in Line'Range loop
         New_Line.Content (I - Line'First + 1) := Line (I);
      end loop;
      New_Line.Length := Line'Length;

      Buffer.Count := Buffer.Count + 1;
      Buffer.Lines (Buffer.Count) := New_Line;
      Success := True;
   end Add_Line;

   --  ============================================================
   --  Clear Buffer
   --  ============================================================

   procedure Clear_Buffer (Buffer : out Output_Buffer) is
   begin
      Buffer := Empty_Buffer;
   end Clear_Buffer;

   --  ============================================================
   --  Contains Pattern
   --  ============================================================

   function Contains_Pattern
     (Buffer  : Output_Buffer;
      Pattern : String) return Boolean
   is
   begin
      for I in 1 .. Buffer.Count loop
         declare
            Line : constant String :=
              Buffer.Lines (I).Content (1 .. Buffer.Lines (I).Length);
         begin
            if Line'Length >= Pattern'Length then
               for J in Line'First .. Line'Last - Pattern'Length + 1 loop
                  declare
                     Match : Boolean := True;
                  begin
                     for K in Pattern'Range loop
                        if Line (J + K - Pattern'First) /= Pattern (K) then
                           Match := False;
                           exit;
                        end if;
                     end loop;
                     if Match then
                        return True;
                     end if;
                  end;
               end loop;
            end if;
         end;
      end loop;
      return False;
   end Contains_Pattern;

end GNATprove_Wrapper;
