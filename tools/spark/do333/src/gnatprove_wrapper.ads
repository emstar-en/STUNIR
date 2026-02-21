--  STUNIR DO-333 GNATprove Wrapper
--  Low-level GNATprove invocation and result parsing
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0
--
--  This package provides GNATprove wrapper:
--  - Command line construction
--  - Invocation
--  - Result parsing
--
--  DO-333 Objectives: FM.2, FM.5 (Proofs, Integration)

pragma SPARK_Mode (On);

with SPARK_Integration; use SPARK_Integration;
with PO_Manager; use PO_Manager;
with VC_Tracker; use VC_Tracker;

package GNATprove_Wrapper is

   --  ============================================================
   --  Constants
   --  ============================================================

   Max_Command_Length : constant := 4096;
   Max_Output_Lines   : constant := 10000;
   Max_Line_Length    : constant := 1024;
   Max_Project_Len    : constant := 512;

   --  ============================================================
   --  Bounded Strings
   --  ============================================================

   subtype Command_Index is Positive range 1 .. Max_Command_Length;
   subtype Command_Length is Natural range 0 .. Max_Command_Length;
   subtype Command_String is String (Command_Index);

   subtype Line_Index is Positive range 1 .. Max_Line_Length;
   subtype Line_Length is Natural range 0 .. Max_Line_Length;
   subtype Line_String is String (Line_Index);

   subtype Project_Index is Positive range 1 .. Max_Project_Len;
   subtype Project_Length is Natural range 0 .. Max_Project_Len;
   subtype Project_String is String (Project_Index);

   --  ============================================================
   --  Output Line
   --  ============================================================

   type Output_Line is record
      Content : Line_String;
      Length  : Line_Length;
   end record;

   Empty_Line : constant Output_Line := (
      Content => (others => ' '),
      Length  => 0
   );

   --  ============================================================
   --  Output Buffer
   --  ============================================================

   subtype Output_Index is Positive range 1 .. Max_Output_Lines;
   subtype Output_Count is Natural range 0 .. Max_Output_Lines;

   type Output_Array is array (Output_Index) of Output_Line;

   type Output_Buffer is record
      Lines : Output_Array;
      Count : Output_Count;
   end record;

   Empty_Buffer : constant Output_Buffer := (
      Lines => (others => Empty_Line),
      Count => 0
   );

   --  ============================================================
   --  Command Construction
   --  ============================================================

   --  Build GNATprove command line
   procedure Build_Command
     (Project_File : String;
      Config       : SPARK_Config;
      Command      : out Command_String;
      Length       : out Command_Length)
   with
      Pre => Project_File'Length > 0 and then
             Project_File'Length <= Max_Project_Len and then
             Is_Valid_Config (Config);

   --  Build flow-only command
   procedure Build_Flow_Command
     (Project_File : String;
      Command      : out Command_String;
      Length       : out Command_Length)
   with
      Pre => Project_File'Length > 0 and then
             Project_File'Length <= Max_Project_Len;

   --  Build replay command
   procedure Build_Replay_Command
     (Project_File : String;
      Config       : SPARK_Config;
      Command      : out Command_String;
      Length       : out Command_Length)
   with
      Pre => Project_File'Length > 0 and then
             Project_File'Length <= Max_Project_Len;

   --  ============================================================
   --  Execution (placeholder - actual execution via shell)
   --  ============================================================

   --  Note: Actual execution happens through shell scripts
   --  These procedures prepare the command and parse results

   type Execution_Status is (
      Exec_Success,
      Exec_Error,
      Exec_Timeout,
      Exec_Not_Found
   );

   --  ============================================================
   --  Result Parsing
   --  ============================================================

   --  Parse GNATprove summary from output
   procedure Parse_Summary
     (Output   : Output_Buffer;
      Result   : out SPARK_Result;
      Success  : out Boolean);

   --  Parse proof obligations from JSON output
   procedure Parse_PO_Results
     (Output  : Output_Buffer;
      PO_Coll : out PO_Collection;
      Success : out Boolean);

   --  Parse verification conditions from JSON output
   procedure Parse_VC_Results
     (Output  : Output_Buffer;
      VC_Coll : out VC_Collection;
      Success : out Boolean);

   --  Parse flow analysis results
   procedure Parse_Flow_Results
     (Output   : Output_Buffer;
      Errors   : out Natural;
      Warnings : out Natural);

   --  ============================================================
   --  Output Buffer Operations
   --  ============================================================

   --  Add line to buffer
   procedure Add_Line
     (Buffer  : in Out Output_Buffer;
      Line    : String;
      Success : out Boolean)
   with
      Pre => Line'Length <= Max_Line_Length;

   --  Clear buffer
   procedure Clear_Buffer (Buffer : out Output_Buffer)
   with
      Post => Buffer.Count = 0;

   --  Check if buffer contains pattern
   function Contains_Pattern
     (Buffer  : Output_Buffer;
      Pattern : String) return Boolean
   with
      Pre => Pattern'Length > 0;

end GNATprove_Wrapper;
