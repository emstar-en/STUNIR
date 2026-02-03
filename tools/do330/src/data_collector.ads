--  STUNIR DO-330 Data Collector Specification
--  Collects Qualification Data from DO-331/332/333 Tools
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0
--
--  This package collects tool qualification data from:
--  - DO-331 Model-Based Development outputs
--  - DO-332 Object-Oriented Technology outputs
--  - DO-333 Formal Methods outputs
--  - Test results and coverage data
--  - Build and configuration manifests
--
--  DO-330 Objective: T-2 (Tool Qualification)

pragma SPARK_Mode (On);

with Templates; use Templates;

package Data_Collector is

   --  ============================================================
   --  Constants
   --  ============================================================

   Max_Tools           : constant := 32;
   Max_Entries         : constant := 256;
   Max_Hash_Length     : constant := 64;  --  SHA256 hex
   Max_Version_Length  : constant := 32;

   --  ============================================================
   --  Coverage and Percentage Types
   --  ============================================================

   subtype Coverage_Percentage is Float range 0.0 .. 100.0;
   subtype Test_Count is Natural range 0 .. 99999;

   --  ============================================================
   --  Hash and Version Strings
   --  ============================================================

   subtype Hash_Index is Positive range 1 .. Max_Hash_Length;
   subtype Hash_Length_Type is Natural range 0 .. Max_Hash_Length;
   subtype Hash_String is String (Hash_Index);

   subtype Version_Index is Positive range 1 .. Max_Version_Length;
   subtype Version_Length_Type is Natural range 0 .. Max_Version_Length;
   subtype Version_String is String (Version_Index);

   --  ============================================================
   --  DO-331 Model-Based Data
   --  ============================================================

   type DO331_Data is record
      Available           : Boolean;
      Model_Count         : Natural;
      Total_Coverage      : Coverage_Percentage;
      Traceability_Links  : Natural;
      SysML_Exports       : Natural;
      XMI_Exports         : Natural;
      Data_Valid          : Boolean;
   end record;

   Null_DO331_Data : constant DO331_Data := (
      Available          => False,
      Model_Count        => 0,
      Total_Coverage     => 0.0,
      Traceability_Links => 0,
      SysML_Exports      => 0,
      XMI_Exports        => 0,
      Data_Valid         => False
   );

   --  ============================================================
   --  DO-332 OOP Data
   --  ============================================================

   type DO332_Data is record
      Available              : Boolean;
      Classes_Analyzed       : Natural;
      Inheritance_Verified   : Boolean;
      Polymorphism_Verified  : Boolean;
      Max_Inheritance_Depth  : Natural;
      Coupling_Metrics_Valid : Boolean;
      Data_Valid             : Boolean;
   end record;

   Null_DO332_Data : constant DO332_Data := (
      Available             => False,
      Classes_Analyzed      => 0,
      Inheritance_Verified  => False,
      Polymorphism_Verified => False,
      Max_Inheritance_Depth => 0,
      Coupling_Metrics_Valid => False,
      Data_Valid            => False
   );

   --  ============================================================
   --  DO-333 Formal Methods Data
   --  ============================================================

   type DO333_Data is record
      Available         : Boolean;
      Total_VCs         : Natural;
      Proven_VCs        : Natural;
      Unproven_VCs      : Natural;
      Proof_Coverage    : Coverage_Percentage;
      Prover_Used       : Name_String;
      Prover_Len        : Name_Length_Type;
      Data_Valid        : Boolean;
   end record;

   Null_DO333_Data : constant DO333_Data := (
      Available      => False,
      Total_VCs      => 0,
      Proven_VCs     => 0,
      Unproven_VCs   => 0,
      Proof_Coverage => 0.0,
      Prover_Used    => (others => ' '),
      Prover_Len     => 0,
      Data_Valid     => False
   );

   --  ============================================================
   --  Test Results Data
   --  ============================================================

   type Test_Data is record
      Available        : Boolean;
      Total_Tests      : Test_Count;
      Passed_Tests     : Test_Count;
      Failed_Tests     : Test_Count;
      Skipped_Tests    : Test_Count;
      Statement_Cov    : Coverage_Percentage;
      Branch_Cov       : Coverage_Percentage;
      MCDC_Cov         : Coverage_Percentage;
      Data_Valid       : Boolean;
   end record;

   Null_Test_Data : constant Test_Data := (
      Available     => False,
      Total_Tests   => 0,
      Passed_Tests  => 0,
      Failed_Tests  => 0,
      Skipped_Tests => 0,
      Statement_Cov => 0.0,
      Branch_Cov    => 0.0,
      MCDC_Cov      => 0.0,
      Data_Valid    => False
   );

   --  ============================================================
   --  Build Configuration Data
   --  ============================================================

   type Build_Data is record
      Available        : Boolean;
      Compiler_Version : Version_String;
      Compiler_Len     : Version_Length_Type;
      Build_Date       : Name_String;  --  ISO date
      Build_Date_Len   : Name_Length_Type;
      Git_Commit       : Hash_String;
      Git_Commit_Len   : Hash_Length_Type;
      Tool_Hash        : Hash_String;
      Tool_Hash_Len    : Hash_Length_Type;
      Data_Valid       : Boolean;
   end record;

   Null_Build_Data : constant Build_Data := (
      Available        => False,
      Compiler_Version => (others => ' '),
      Compiler_Len     => 0,
      Build_Date       => (others => ' '),
      Build_Date_Len   => 0,
      Git_Commit       => (others => ' '),
      Git_Commit_Len   => 0,
      Tool_Hash        => (others => ' '),
      Tool_Hash_Len    => 0,
      Data_Valid       => False
   );

   --  ============================================================
   --  Unified Tool Data (All sources combined)
   --  ============================================================

   type Tool_Data is record
      --  Tool identification
      Tool_Name      : Name_String;
      Tool_Name_Len  : Name_Length_Type;
      Tool_Version   : Version_String;
      Version_Len    : Version_Length_Type;
      TQL            : TQL_Level;
      DAL            : DAL_Level;

      --  Integration data
      DO331          : DO331_Data;
      DO332          : DO332_Data;
      DO333          : DO333_Data;
      Tests          : Test_Data;
      Build          : Build_Data;

      --  Status flags
      Is_Qualified   : Boolean;
      Data_Complete  : Boolean;
   end record;

   Null_Tool_Data : constant Tool_Data := (
      Tool_Name     => (others => ' '),
      Tool_Name_Len => 0,
      Tool_Version  => (others => ' '),
      Version_Len   => 0,
      TQL           => TQL_5,
      DAL           => DAL_E,
      DO331         => Null_DO331_Data,
      DO332         => Null_DO332_Data,
      DO333         => Null_DO333_Data,
      Tests         => Null_Test_Data,
      Build         => Null_Build_Data,
      Is_Qualified  => False,
      Data_Complete => False
   );

   --  ============================================================
   --  Collection Status
   --  ============================================================

   type Collect_Status is (
      Success,
      Source_Not_Found,
      Parse_Error,
      Incomplete_Data,
      Invalid_Format,
      IO_Error
   );

   --  ============================================================
   --  Collection Operations
   --  ============================================================

   --  Initialize tool data structure
   procedure Initialize_Tool_Data
     (Data      : out Tool_Data;
      Tool_Name : String;
      Version   : String;
      TQL       : TQL_Level;
      DAL       : DAL_Level)
   with Pre => Tool_Name'Length > 0 and Tool_Name'Length <= Max_Name_Length and
               Version'Length > 0 and Version'Length <= Max_Version_Length;

   --  Collect DO-331 data from source directory
   procedure Collect_DO331_Data
     (Data       : in out Tool_Data;
      Source_Dir : String;
      Status     : out Collect_Status)
   with Pre => Source_Dir'Length > 0 and Source_Dir'Length <= Max_Path_Length;

   --  Collect DO-332 data from source directory
   procedure Collect_DO332_Data
     (Data       : in out Tool_Data;
      Source_Dir : String;
      Status     : out Collect_Status)
   with Pre => Source_Dir'Length > 0 and Source_Dir'Length <= Max_Path_Length;

   --  Collect DO-333 data from source directory
   procedure Collect_DO333_Data
     (Data       : in out Tool_Data;
      Source_Dir : String;
      Status     : out Collect_Status)
   with Pre => Source_Dir'Length > 0 and Source_Dir'Length <= Max_Path_Length;

   --  Collect test results data
   procedure Collect_Test_Data
     (Data       : in out Tool_Data;
      Source_Dir : String;
      Status     : out Collect_Status)
   with Pre => Source_Dir'Length > 0 and Source_Dir'Length <= Max_Path_Length;

   --  Collect build configuration data
   procedure Collect_Build_Data
     (Data       : in out Tool_Data;
      Source_Dir : String;
      Status     : out Collect_Status)
   with Pre => Source_Dir'Length > 0 and Source_Dir'Length <= Max_Path_Length;

   --  Collect all data from all sources
   procedure Collect_All_Data
     (Data       : in out Tool_Data;
      Base_Dir   : String;
      Status     : out Collect_Status)
   with Pre => Base_Dir'Length > 0 and Base_Dir'Length <= Max_Path_Length;

   --  ============================================================
   --  Validation Operations
   --  ============================================================

   --  Validate collected data completeness
   function Is_Data_Complete (Data : Tool_Data) return Boolean;

   --  Check if tool meets qualification requirements for TQL level
   function Meets_TQL_Requirements
     (Data : Tool_Data;
      TQL  : TQL_Level) return Boolean;

   --  Calculate overall qualification score
   function Calculate_Qualification_Score (Data : Tool_Data) return Coverage_Percentage;

   --  ============================================================
   --  Summary Generation
   --  ============================================================

   --  Generate summary string for data
   procedure Generate_Data_Summary
     (Data    : Tool_Data;
      Summary : out Value_String;
      Sum_Len : out Value_Length_Type);

   --  Get status message
   function Status_Message (Status : Collect_Status) return String;

end Data_Collector;
