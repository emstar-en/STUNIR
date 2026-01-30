--  STUNIR DO-331 Interface Specification
--  Model-Based Development Tool Integration
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0
--
--  This package provides the interface to DO-331 tools:
--  - IR to SysML 2.0 transformation
--  - Coverage data collection
--  - Traceability generation
--  - Model validation

pragma SPARK_Mode (On);

with DO331_Types; use DO331_Types;

package DO331_Interface is

   --  ============================================================
   --  Constants
   --  ============================================================

   Max_IR_Path_Length : constant := 512;
   Max_Output_Path_Length : constant := 512;

   --  ============================================================
   --  Path Types
   --  ============================================================

   subtype IR_Path_Index is Positive range 1 .. Max_IR_Path_Length;
   subtype IR_Path_Length is Natural range 0 .. Max_IR_Path_Length;
   subtype IR_Path_String is String (IR_Path_Index);

   subtype Output_Path_Index is Positive range 1 .. Max_Output_Path_Length;
   subtype Output_Path_Length is Natural range 0 .. Max_Output_Path_Length;
   subtype Output_Path_String is String (Output_Path_Index);

   --  ============================================================
   --  Transform Configuration
   --  ============================================================

   type Transform_Config is record
      IR_Path        : IR_Path_String;
      IR_Path_Len    : IR_Path_Length;
      Output_Path    : Output_Path_String;
      Output_Len     : Output_Path_Length;
      DAL            : DAL_Level;
      Include_Cov    : Boolean;
      Include_Trace  : Boolean;
      Generate_XMI   : Boolean;
   end record;

   Null_Transform_Config : constant Transform_Config := (
      IR_Path       => (others => ' '),
      IR_Path_Len   => 0,
      Output_Path   => (others => ' '),
      Output_Len    => 0,
      DAL           => DAL_C,
      Include_Cov   => True,
      Include_Trace => True,
      Generate_XMI  => False
   );

   --  ============================================================
   --  Transformation Operations
   --  ============================================================

   --  Initialize transform configuration
   procedure Initialize_Config
     (Config     : out Transform_Config;
      IR_Path    : String;
      Output_Path: String;
      DAL        : DAL_Level)
   with Pre  => IR_Path'Length > 0 and IR_Path'Length <= Max_IR_Path_Length and
                Output_Path'Length > 0 and Output_Path'Length <= Max_Output_Path_Length,
        Post => Config.DAL = DAL;

   --  Transform IR to SysML 2.0 model
   procedure Transform_To_SysML
     (Config : Transform_Config;
      Result : out DO331_Result;
      Status : out DO331_Status)
   with Pre  => Config.IR_Path_Len > 0 and Config.Output_Len > 0,
        Post => (if Status = Success then Result.Success);

   --  Collect coverage data from model
   procedure Collect_Coverage
     (Model_Path : String;
      Result     : in out DO331_Result;
      Status     : out DO331_Status)
   with Pre  => Model_Path'Length > 0 and Model_Path'Length <= Max_Output_Path_Length,
        Post => (if Status = Success then Result.Coverage_Total >= 0);

   --  Generate traceability links
   procedure Generate_Traceability
     (Config : Transform_Config;
      Result : in out DO331_Result;
      Status : out DO331_Status)
   with Pre  => Config.Include_Trace,
        Post => (if Status = Success then Result.Trace_Total >= 0);

   --  ============================================================
   --  Model Operations
   --  ============================================================

   --  Add model to result
   procedure Add_Model
     (Result : in out DO331_Result;
      Name   : String;
      Path   : String;
      Kind   : Model_Kind;
      Status : out DO331_Status)
   with Pre  => Name'Length > 0 and Name'Length <= Max_Model_Name_Length and
                Path'Length > 0 and Path'Length <= Max_Model_Path_Length and
                Result.Model_Total < Max_Model_Count,
        Post => (if Status = Success then Result.Model_Total = Result.Model_Total'Old + 1);

   --  Add coverage item
   procedure Add_Coverage_Item
     (Result     : in out DO331_Result;
      Element_ID : String;
      Kind       : Coverage_Kind;
      Covered    : Boolean;
      Status     : out DO331_Status)
   with Pre  => Element_ID'Length > 0 and Element_ID'Length <= Max_Element_ID_Length and
                Result.Coverage_Total < Max_Coverage_Items,
        Post => (if Status = Success then Result.Coverage_Total = Result.Coverage_Total'Old + 1);

   --  Add traceability link
   procedure Add_Trace_Link
     (Result    : in out DO331_Result;
      Source_ID : String;
      Target_ID : String;
      Direction : Trace_Direction;
      Status    : out DO331_Status)
   with Pre  => Source_ID'Length > 0 and Source_ID'Length <= Max_Element_ID_Length and
                Target_ID'Length > 0 and Target_ID'Length <= Max_Element_ID_Length and
                Result.Trace_Total < Max_Trace_Links,
        Post => (if Status = Success then Result.Trace_Total = Result.Trace_Total'Old + 1);

   --  ============================================================
   --  Validation Operations
   --  ============================================================

   --  Validate model completeness
   function Validate_Model_Completeness
     (Result : DO331_Result) return Boolean;

   --  Check DAL requirements
   function Meets_DAL_Requirements
     (Result : DO331_Result;
      DAL    : DAL_Level) return Boolean;

   --  Calculate overall coverage percentage
   function Calculate_Coverage_Percentage
     (Result : DO331_Result) return Percentage_Type;

   --  ============================================================
   --  Summary Operations
   --  ============================================================

   --  Finalize result with computed metrics
   procedure Finalize_Result
     (Result : in out DO331_Result)
   with Post => Result.Is_Complete;

end DO331_Interface;
