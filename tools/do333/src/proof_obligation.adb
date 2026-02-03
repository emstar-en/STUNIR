--  STUNIR DO-333 Proof Obligation Types
--  Implementation
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

package body Proof_Obligation is

   --  ============================================================
   --  Create PO
   --  ============================================================

   procedure Create_PO
     (ID          : Natural;
      Kind        : PO_Kind;
      Criticality : Criticality_Level;
      Source      : String;
      Subprogram  : String;
      Line        : Natural;
      Column      : Natural;
      PO          : out Proof_Obligation_Record)
   is
      Src_Buf  : Source_String := (others => ' ');
      Subp_Buf : Subp_String := (others => ' ');
   begin
      --  Copy source file name
      for I in Source'Range loop
         Src_Buf (I - Source'First + 1) := Source (I);
      end loop;

      --  Copy subprogram name
      for I in Subprogram'Range loop
         Subp_Buf (I - Subprogram'First + 1) := Subprogram (I);
      end loop;

      PO := (
         ID          => ID,
         Kind        => Kind,
         Status      => PO_Unproved,
         Criticality => Criticality,
         Strategy    => Strategy_Auto,
         Source_File => Src_Buf,
         Source_Len  => Source'Length,
         Subprogram  => Subp_Buf,
         Subp_Len    => Subprogram'Length,
         Line        => Line,
         Column      => Column,
         Context     => (others => ' '),
         Context_Len => 0,
         Prover_Time => 0,
         Step_Count  => 0,
         Prover_Name => (others => ' '),
         Prover_Len  => 0
      );
   end Create_PO;

   --  ============================================================
   --  Update Status
   --  ============================================================

   procedure Update_Status
     (PO     : in Out Proof_Obligation_Record;
      Status : PO_Status;
      Time   : Natural;
      Steps  : Natural)
   is
   begin
      PO.Status      := Status;
      PO.Prover_Time := Time;
      PO.Step_Count  := Steps;
   end Update_Status;

   --  ============================================================
   --  Set Prover
   --  ============================================================

   procedure Set_Prover
     (PO   : in Out Proof_Obligation_Record;
      Name : String)
   is
      Prover_Buf : Source_String := (others => ' ');
   begin
      for I in Name'Range loop
         Prover_Buf (I - Name'First + 1) := Name (I);
      end loop;
      PO.Prover_Name := Prover_Buf;
      PO.Prover_Len  := Name'Length;
   end Set_Prover;

   --  ============================================================
   --  Status Name
   --  ============================================================

   function Status_Name (S : PO_Status) return String is
   begin
      case S is
         when PO_Unproved          => return "Unproved";
         when PO_Proved            => return "Proved";
         when PO_Timeout           => return "Timeout";
         when PO_Error             => return "Error";
         when PO_Manually_Justified => return "Manually_Justified";
         when PO_Flow_Proved       => return "Flow_Proved";
         when PO_Skipped           => return "Skipped";
      end case;
   end Status_Name;

   --  ============================================================
   --  Kind Name
   --  ============================================================

   function Kind_Name (K : PO_Kind) return String is
   begin
      case K is
         when PO_Precondition             => return "Precondition";
         when PO_Postcondition            => return "Postcondition";
         when PO_Loop_Invariant_Init      => return "Loop_Invariant_Init";
         when PO_Loop_Invariant_Preserv   => return "Loop_Invariant_Preserv";
         when PO_Loop_Variant             => return "Loop_Variant";
         when PO_Assert                   => return "Assert";
         when PO_Range_Check              => return "Range_Check";
         when PO_Overflow_Check           => return "Overflow_Check";
         when PO_Division_Check           => return "Division_Check";
         when PO_Index_Check              => return "Index_Check";
         when PO_Discriminant_Check       => return "Discriminant_Check";
         when PO_Predicate_Check          => return "Predicate_Check";
         when PO_Default_Initial_Condition => return "Default_Initial_Condition";
         when PO_Contract_Cases           => return "Contract_Cases";
         when PO_Initial_Condition        => return "Initial_Condition";
         when PO_Type_Invariant           => return "Type_Invariant";
         when PO_Tag_Check                => return "Tag_Check";
         when PO_Ceiling_Priority         => return "Ceiling_Priority";
         when PO_Subprogram_Variant       => return "Subprogram_Variant";
         when PO_Other                    => return "Other";
      end case;
   end Kind_Name;

   --  ============================================================
   --  Criticality Name
   --  ============================================================

   function Criticality_Name (C : Criticality_Level) return String is
   begin
      case C is
         when DAL_A => return "DAL-A (Catastrophic)";
         when DAL_B => return "DAL-B (Hazardous)";
         when DAL_C => return "DAL-C (Major)";
         when DAL_D => return "DAL-D (Minor)";
         when DAL_E => return "DAL-E (No Effect)";
      end case;
   end Criticality_Name;

end Proof_Obligation;
