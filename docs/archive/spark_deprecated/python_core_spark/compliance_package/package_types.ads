--  STUNIR Compliance Package Types Specification
--  DO-330 Tool Qualification Package Types
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0
--
--  This package defines types for DO-330 compliance packages:
--  - Artifact collection
--  - Traceability matrices
--  - Configuration index
--  - Package generation

pragma SPARK_Mode (On);

package Package_Types is

   --  ============================================================
   --  Constants
   --  ============================================================

   Max_Name_Length        : constant := 128;
   Max_Path_Length        : constant := 512;
   Max_Hash_Length        : constant := 64;  --  SHA256 hex
   Max_Artifacts          : constant := 256;
   Max_Trace_Entries      : constant := 1024;
   Max_Config_Items       : constant := 128;
   Max_Description_Length : constant := 512;

   --  ============================================================
   --  TQL and DAL Levels
   --  ============================================================

   type TQL_Level is (TQL_1, TQL_2, TQL_3, TQL_4, TQL_5);
   type DAL_Level is (DAL_A, DAL_B, DAL_C, DAL_D, DAL_E);

   --  ============================================================
   --  String Types
   --  ============================================================

   subtype Name_Index is Positive range 1 .. Max_Name_Length;
   subtype Name_Length is Natural range 0 .. Max_Name_Length;
   subtype Name_String is String (Name_Index);

   subtype Path_Index is Positive range 1 .. Max_Path_Length;
   subtype Path_Length is Natural range 0 .. Max_Path_Length;
   subtype Path_String is String (Path_Index);

   subtype Hash_Index is Positive range 1 .. Max_Hash_Length;
   subtype Hash_Length is Natural range 0 .. Max_Hash_Length;
   subtype Hash_String is String (Hash_Index);

   subtype Desc_Index is Positive range 1 .. Max_Description_Length;
   subtype Desc_Length is Natural range 0 .. Max_Description_Length;
   subtype Desc_String is String (Desc_Index);

   --  ============================================================
   --  Artifact Types
   --  ============================================================

   type Artifact_Kind is (
      Source_Code,
      Object_Code,
      Executable,
      Test_Case,
      Test_Result,
      Coverage_Data,
      Proof_Result,
      Document,
      Configuration,
      Receipt,
      Manifest
   );

   type Artifact_Entry is record
      Name        : Name_String;
      Name_Len    : Name_Length;
      Path        : Path_String;
      Path_Len    : Path_Length;
      Hash        : Hash_String;
      Hash_Len    : Hash_Length;
      Kind        : Artifact_Kind;
      Size_Bytes  : Natural;
      Is_Valid    : Boolean;
   end record;

   Null_Artifact_Entry : constant Artifact_Entry := (
      Name       => (others => ' '),
      Name_Len   => 0,
      Path       => (others => ' '),
      Path_Len   => 0,
      Hash       => (others => ' '),
      Hash_Len   => 0,
      Kind       => Source_Code,
      Size_Bytes => 0,
      Is_Valid   => False
   );

   subtype Artifact_Index is Positive range 1 .. Max_Artifacts;
   subtype Artifact_Count is Natural range 0 .. Max_Artifacts;
   type Artifact_Array is array (Artifact_Index) of Artifact_Entry;

   --  ============================================================
   --  Traceability Types
   --  ============================================================

   type Trace_Kind is (
      Req_To_Design,
      Design_To_Code,
      Code_To_Test,
      Test_To_Result,
      Req_To_Test,
      Code_To_Proof
   );

   type Trace_Entry is record
      Source_ID   : Name_String;
      Source_Len  : Name_Length;
      Target_ID   : Name_String;
      Target_Len  : Name_Length;
      Kind        : Trace_Kind;
      Verified    : Boolean;
      Is_Valid    : Boolean;
   end record;

   Null_Trace_Entry : constant Trace_Entry := (
      Source_ID  => (others => ' '),
      Source_Len => 0,
      Target_ID  => (others => ' '),
      Target_Len => 0,
      Kind       => Req_To_Design,
      Verified   => False,
      Is_Valid   => False
   );

   subtype Trace_Index is Positive range 1 .. Max_Trace_Entries;
   subtype Trace_Count is Natural range 0 .. Max_Trace_Entries;
   type Trace_Array is array (Trace_Index) of Trace_Entry;

   --  ============================================================
   --  Configuration Item
   --  ============================================================

   type Config_Item is record
      Name        : Name_String;
      Name_Len    : Name_Length;
      Version     : Name_String;
      Version_Len : Name_Length;
      Hash        : Hash_String;
      Hash_Len    : Hash_Length;
      Description : Desc_String;
      Desc_Len    : Desc_Length;
      Is_Valid    : Boolean;
   end record;

   Null_Config_Item : constant Config_Item := (
      Name        => (others => ' '),
      Name_Len    => 0,
      Version     => (others => ' '),
      Version_Len => 0,
      Hash        => (others => ' '),
      Hash_Len    => 0,
      Description => (others => ' '),
      Desc_Len    => 0,
      Is_Valid    => False
   );

   subtype Config_Index is Positive range 1 .. Max_Config_Items;
   subtype Config_Count is Natural range 0 .. Max_Config_Items;
   type Config_Array is array (Config_Index) of Config_Item;

   --  ============================================================
   --  Compliance Package
   --  ============================================================

   type Compliance_Package is record
      --  Package identification
      Tool_Name    : Name_String;
      Tool_Name_Len: Name_Length;
      Version      : Name_String;
      Version_Len  : Name_Length;
      TQL          : TQL_Level;
      DAL          : DAL_Level;

      --  Artifacts
      Artifacts    : Artifact_Array;
      Artifact_Total: Artifact_Count;

      --  Traceability
      Traces       : Trace_Array;
      Trace_Total  : Trace_Count;

      --  Configuration
      Config_Items : Config_Array;
      Config_Total : Config_Count;

      --  Status
      Is_Complete  : Boolean;
      Is_Valid     : Boolean;
   end record;

   Null_Compliance_Package : constant Compliance_Package := (
      Tool_Name     => (others => ' '),
      Tool_Name_Len => 0,
      Version       => (others => ' '),
      Version_Len   => 0,
      TQL           => TQL_5,
      DAL           => DAL_E,
      Artifacts     => (others => Null_Artifact_Entry),
      Artifact_Total=> 0,
      Traces        => (others => Null_Trace_Entry),
      Trace_Total   => 0,
      Config_Items  => (others => Null_Config_Item),
      Config_Total  => 0,
      Is_Complete   => False,
      Is_Valid      => False
   );

   --  ============================================================
   --  Status
   --  ============================================================

   type Package_Status is (
      Success,
      Invalid_Config,
      Artifact_Missing,
      Hash_Mismatch,
      Trace_Incomplete,
      Generation_Failed,
      Validation_Failed,
      IO_Error
   );

   --  ============================================================
   --  Utility Functions
   --  ============================================================

   function Status_Message (Status : Package_Status) return String;
   function TQL_Name (Level : TQL_Level) return String;
   function DAL_Name (Level : DAL_Level) return String;
   function Artifact_Kind_Name (Kind : Artifact_Kind) return String;
   function Trace_Kind_Name (Kind : Trace_Kind) return String;

end Package_Types;
