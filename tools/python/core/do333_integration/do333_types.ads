--  STUNIR DO-333 Integration Types Specification
--  Formal Methods Data Types
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0
--
--  This package defines types for DO-333 formal verification:
--  - Proof obligations
--  - Verification conditions
--  - Evidence generation
--  - Prover integration

pragma SPARK_Mode (On);

package DO333_Types is

   --  ============================================================
   --  Constants
   --  ============================================================

   Max_VC_Name_Length     : constant := 128;
   Max_Source_Path_Length : constant := 512;
   Max_VC_Count           : constant := 4096;
   Max_PO_Count           : constant := 1024;
   Max_Message_Length     : constant := 512;
   Max_Prover_Name_Length : constant := 64;

   --  ============================================================
   --  String Types
   --  ============================================================

   subtype VC_Name_Index is Positive range 1 .. Max_VC_Name_Length;
   subtype VC_Name_Length is Natural range 0 .. Max_VC_Name_Length;
   subtype VC_Name_String is String (VC_Name_Index);

   subtype Source_Path_Index is Positive range 1 .. Max_Source_Path_Length;
   subtype Source_Path_Length is Natural range 0 .. Max_Source_Path_Length;
   subtype Source_Path_String is String (Source_Path_Index);

   subtype Message_Index is Positive range 1 .. Max_Message_Length;
   subtype Message_Length is Natural range 0 .. Max_Message_Length;
   subtype Message_String is String (Message_Index);

   subtype Prover_Name_Index is Positive range 1 .. Max_Prover_Name_Length;
   subtype Prover_Name_Length is Natural range 0 .. Max_Prover_Name_Length;
   subtype Prover_Name_String is String (Prover_Name_Index);

   --  ============================================================
   --  Proof Status Types
   --  ============================================================

   type Proof_Status is (
      Proven,
      Unproven,
      Timeout,
      Error,
      Skipped,
      Not_Attempted
   );

   type VC_Kind is (
      Precondition,
      Postcondition,
      Assert,
      Loop_Invariant,
      Loop_Variant,
      Range_Check,
      Overflow_Check,
      Division_Check,
      Index_Check,
      Discriminant_Check,
      Type_Invariant,
      Contract_Case,
      Subprogram_Variant
   );

   --  ============================================================
   --  Verification Condition
   --  ============================================================

   type Verification_Condition is record
      Name       : VC_Name_String;
      Name_Len   : VC_Name_Length;
      Source     : Source_Path_String;
      Source_Len : Source_Path_Length;
      Line       : Positive;
      Column     : Positive;
      Kind       : VC_Kind;
      Status     : Proof_Status;
      Time_MS    : Natural;
      Prover     : Prover_Name_String;
      Prover_Len : Prover_Name_Length;
      Is_Valid   : Boolean;
   end record;

   Null_VC : constant Verification_Condition := (
      Name       => (others => ' '),
      Name_Len   => 0,
      Source     => (others => ' '),
      Source_Len => 0,
      Line       => 1,
      Column     => 1,
      Kind       => Assert,
      Status     => Not_Attempted,
      Time_MS    => 0,
      Prover     => (others => ' '),
      Prover_Len => 0,
      Is_Valid   => False
   );

   subtype VC_Index is Positive range 1 .. Max_VC_Count;
   subtype VC_Count_Type is Natural range 0 .. Max_VC_Count;
   type VC_Array is array (VC_Index) of Verification_Condition;

   --  ============================================================
   --  Proof Obligation
   --  ============================================================

   type Proof_Obligation is record
      Name       : VC_Name_String;
      Name_Len   : VC_Name_Length;
      Source     : Source_Path_String;
      Source_Len : Source_Path_Length;
      VC_Count   : VC_Count_Type;
      Proven_Count: VC_Count_Type;
      Message    : Message_String;
      Message_Len: Message_Length;
      Status     : Proof_Status;
      Is_Valid   : Boolean;
   end record;

   Null_PO : constant Proof_Obligation := (
      Name        => (others => ' '),
      Name_Len    => 0,
      Source      => (others => ' '),
      Source_Len  => 0,
      VC_Count    => 0,
      Proven_Count=> 0,
      Message     => (others => ' '),
      Message_Len => 0,
      Status      => Not_Attempted,
      Is_Valid    => False
   );

   subtype PO_Index is Positive range 1 .. Max_PO_Count;
   subtype PO_Count_Type is Natural range 0 .. Max_PO_Count;
   type PO_Array is array (PO_Index) of Proof_Obligation;

   --  ============================================================
   --  Percentage Type
   --  ============================================================

   subtype Percentage_Type is Float range 0.0 .. 100.0;

   --  ============================================================
   --  DO-333 Result
   --  ============================================================

   type DO333_Result is record
      --  Verification conditions
      VCs          : VC_Array;
      VC_Total     : VC_Count_Type;
      VC_Proven    : VC_Count_Type;
      VC_Unproven  : VC_Count_Type;
      VC_Timeout   : VC_Count_Type;
      VC_Error     : VC_Count_Type;

      --  Proof obligations
      POs          : PO_Array;
      PO_Total     : PO_Count_Type;
      PO_Proven    : PO_Count_Type;

      --  Metrics
      Proof_Rate   : Percentage_Type;
      Total_Time_MS: Natural;

      --  Prover info
      Prover_Name  : Prover_Name_String;
      Prover_Len   : Prover_Name_Length;

      --  Status
      Success      : Boolean;
      All_Proven   : Boolean;
   end record;

   Null_DO333_Result : constant DO333_Result := (
      VCs          => (others => Null_VC),
      VC_Total     => 0,
      VC_Proven    => 0,
      VC_Unproven  => 0,
      VC_Timeout   => 0,
      VC_Error     => 0,
      POs          => (others => Null_PO),
      PO_Total     => 0,
      PO_Proven    => 0,
      Proof_Rate   => 0.0,
      Total_Time_MS=> 0,
      Prover_Name  => (others => ' '),
      Prover_Len   => 0,
      Success      => False,
      All_Proven   => False
   );

   --  ============================================================
   --  Status
   --  ============================================================

   type DO333_Status is (
      Success,
      Prover_Not_Found,
      Parse_Error,
      Proof_Failed,
      Timeout_Exceeded,
      Resource_Exhausted,
      Configuration_Error,
      IO_Error
   );

   --  ============================================================
   --  Utility Functions
   --  ============================================================

   function Status_Message (Status : DO333_Status) return String;
   function Proof_Status_Name (Status : Proof_Status) return String;
   function VC_Kind_Name (Kind : VC_Kind) return String;

end DO333_Types;
