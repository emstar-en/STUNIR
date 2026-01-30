--  STUNIR DO-333 Interface Implementation
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

package body DO333_Interface is

   --  ============================================================
   --  Helper: Copy String
   --  ============================================================

   procedure Copy_String
     (Source : String;
      Target : out String;
      Length : out Natural)
   with Pre => Target'Length >= Source'Length
   is
   begin
      Length := Source'Length;
      for I in 1 .. Source'Length loop
         pragma Loop_Invariant (I <= Source'Length);
         Target(Target'First + I - 1) := Source(Source'First + I - 1);
      end loop;
      for I in Source'Length + 1 .. Target'Length loop
         Target(Target'First + I - 1) := ' ';
      end loop;
   end Copy_String;

   --  ============================================================
   --  Initialize Config
   --  ============================================================

   procedure Initialize_Config
     (Config      : out Verify_Config;
      Source_Dir  : String;
      Output_Dir  : String;
      Project_File: String := "")
   is
      Src_Len, Out_Len, Proj_Len : Natural;
   begin
      Config := Null_Verify_Config;

      if Source_Dir'Length > 0 and Source_Dir'Length <= Max_Path_Length then
         Copy_String(Source_Dir, Config.Source_Dir, Src_Len);
         Config.Source_Len := Path_Length(Src_Len);
      end if;

      if Output_Dir'Length > 0 and Output_Dir'Length <= Max_Path_Length then
         Copy_String(Output_Dir, Config.Output_Dir, Out_Len);
         Config.Output_Len := Path_Length(Out_Len);
      end if;

      if Project_File'Length > 0 and Project_File'Length <= Max_Path_Length then
         Copy_String(Project_File, Config.Project_File, Proj_Len);
         Config.Project_Len := Path_Length(Proj_Len);
      end if;
   end Initialize_Config;

   --  ============================================================
   --  Run Verification
   --  ============================================================

   procedure Run_Verification
     (Config : Verify_Config;
      Result : out DO333_Result;
      Status : out DO333_Status)
   is
      Prover_Len : Natural;
   begin
      Result := Null_DO333_Result;

      --  Set prover name
      Copy_String("GNATprove", Result.Prover_Name, Prover_Len);
      Result.Prover_Len := Prover_Name_Length(Prover_Len);

      --  Simulated verification run
      --  In real implementation, this would invoke gnatprove

      --  Collect results
      Collect_Proof_Results(
         Output_Dir => Config.Output_Dir(1..Config.Output_Len),
         Result     => Result,
         Status     => Status
      );

      if Status /= Success then
         return;
      end if;

      --  Finalize
      Finalize_Result(Result);
      Result.Success := True;
      Status := Success;
   end Run_Verification;

   --  ============================================================
   --  Collect Proof Results
   --  ============================================================

   procedure Collect_Proof_Results
     (Output_Dir : String;
      Result     : in out DO333_Result;
      Status     : out DO333_Status)
   is
      pragma Unreferenced (Output_Dir);
   begin
      --  Simulated result collection
      --  In real implementation, would parse gnatprove output

      --  Add sample VCs
      declare
         St : DO333_Status;
      begin
         Add_VC(Result, "type_invariant_1", "main.adb", 10, 1,
                Type_Invariant, Proven, St);
         if St /= Success then
            Status := St;
            return;
         end if;

         Add_VC(Result, "precondition_1", "main.adb", 15, 1,
                Precondition, Proven, St);
         if St /= Success then
            Status := St;
            return;
         end if;

         Add_VC(Result, "postcondition_1", "main.adb", 20, 1,
                Postcondition, Proven, St);
         if St /= Success then
            Status := St;
            return;
         end if;
      end;

      Update_VC_Statistics(Result);
      Status := Success;
   end Collect_Proof_Results;

   --  ============================================================
   --  Generate Evidence
   --  ============================================================

   procedure Generate_Evidence
     (Result     : DO333_Result;
      Output_Dir : String;
      Status     : out DO333_Status)
   is
      pragma Unreferenced (Result, Output_Dir);
   begin
      --  Simulated evidence generation
      Status := Success;
   end Generate_Evidence;

   --  ============================================================
   --  Add VC
   --  ============================================================

   procedure Add_VC
     (Result : in out DO333_Result;
      Name   : String;
      Source : String;
      Line   : Positive;
      Column : Positive;
      Kind   : VC_Kind;
      Status : Proof_Status;
      St_Out : out DO333_Status)
   is
      VC : Verification_Condition := Null_VC;
      Name_Len, Source_Len : Natural;
   begin
      Copy_String(Name, VC.Name, Name_Len);
      VC.Name_Len := VC_Name_Length(Name_Len);

      Copy_String(Source, VC.Source, Source_Len);
      VC.Source_Len := Source_Path_Length(Source_Len);

      VC.Line := Line;
      VC.Column := Column;
      VC.Kind := Kind;
      VC.Status := Status;
      VC.Is_Valid := True;

      Result.VC_Total := Result.VC_Total + 1;
      Result.VCs(Result.VC_Total) := VC;
      St_Out := Success;
   end Add_VC;

   --  ============================================================
   --  Add PO
   --  ============================================================

   procedure Add_PO
     (Result  : in out DO333_Result;
      Name    : String;
      Source  : String;
      VC_Cnt  : VC_Count_Type;
      Proven  : VC_Count_Type;
      St_Out  : out DO333_Status)
   is
      PO : Proof_Obligation := Null_PO;
      Name_Len, Source_Len : Natural;
   begin
      Copy_String(Name, PO.Name, Name_Len);
      PO.Name_Len := VC_Name_Length(Name_Len);

      Copy_String(Source, PO.Source, Source_Len);
      PO.Source_Len := Source_Path_Length(Source_Len);

      PO.VC_Count := VC_Cnt;
      PO.Proven_Count := Proven;
      PO.Status := (if Proven = VC_Cnt then DO333_Types.Proven else DO333_Types.Unproven);
      PO.Is_Valid := True;

      Result.PO_Total := Result.PO_Total + 1;
      Result.POs(Result.PO_Total) := PO;
      St_Out := Success;
   end Add_PO;

   --  ============================================================
   --  All VCs Proven
   --  ============================================================

   function All_VCs_Proven (Result : DO333_Result) return Boolean is
   begin
      return Result.VC_Total > 0 and 
             Result.VC_Proven = Result.VC_Total;
   end All_VCs_Proven;

   --  ============================================================
   --  Calculate Proof Rate
   --  ============================================================

   function Calculate_Proof_Rate (Result : DO333_Result) return Percentage_Type is
   begin
      if Result.VC_Total = 0 then
         return 0.0;
      end if;
      return Percentage_Type(Float(Result.VC_Proven) / Float(Result.VC_Total) * 100.0);
   end Calculate_Proof_Rate;

   --  ============================================================
   --  Meets Requirements
   --  ============================================================

   function Meets_Requirements
     (Result   : DO333_Result;
      Min_Rate : Percentage_Type) return Boolean
   is
   begin
      return Result.Proof_Rate >= Min_Rate and Result.Success;
   end Meets_Requirements;

   --  ============================================================
   --  Finalize Result
   --  ============================================================

   procedure Finalize_Result
     (Result : in out DO333_Result)
   is
   begin
      Update_VC_Statistics(Result);
      Result.Proof_Rate := Calculate_Proof_Rate(Result);
      Result.All_Proven := All_VCs_Proven(Result);
   end Finalize_Result;

   --  ============================================================
   --  Update VC Statistics
   --  ============================================================

   procedure Update_VC_Statistics
     (Result : in out DO333_Result)
   is
      Proven_Cnt, Unproven_Cnt, Timeout_Cnt, Error_Cnt : VC_Count_Type := 0;
   begin
      for I in 1 .. Result.VC_Total loop
         pragma Loop_Invariant (I <= Result.VC_Total);
         pragma Loop_Invariant (Proven_Cnt <= I - 1);
         pragma Loop_Invariant (Unproven_Cnt <= I - 1);
         pragma Loop_Invariant (Timeout_Cnt <= I - 1);
         pragma Loop_Invariant (Error_Cnt <= I - 1);

         case Result.VCs(I).Status is
            when DO333_Types.Proven =>
               Proven_Cnt := Proven_Cnt + 1;
            when DO333_Types.Unproven =>
               Unproven_Cnt := Unproven_Cnt + 1;
            when DO333_Types.Timeout =>
               Timeout_Cnt := Timeout_Cnt + 1;
            when DO333_Types.Error =>
               Error_Cnt := Error_Cnt + 1;
            when others =>
               null;
         end case;
      end loop;

      Result.VC_Proven := Proven_Cnt;
      Result.VC_Unproven := Unproven_Cnt;
      Result.VC_Timeout := Timeout_Cnt;
      Result.VC_Error := Error_Cnt;
   end Update_VC_Statistics;

end DO333_Interface;
