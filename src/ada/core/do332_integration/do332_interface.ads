--  STUNIR DO-332 Interface Specification
--  Object-Oriented Technology Verification
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0
--
--  This package provides the interface to DO-332 OOP analysis:
--  - Class hierarchy analysis
--  - Inheritance depth verification
--  - Polymorphism safety checking
--  - Coupling metrics calculation

pragma SPARK_Mode (On);

with DO332_Types; use DO332_Types;

package DO332_Interface is

   --  ============================================================
   --  Constants
   --  ============================================================

   Max_IR_Path_Length     : constant := 512;
   Max_Output_Path_Length : constant := 512;
   Default_Max_Depth      : constant := 8;
   Default_Max_Coupling   : constant := 10;

   --  ============================================================
   --  Path Types
   --  ============================================================

   subtype Path_Index is Positive range 1 .. Max_IR_Path_Length;
   subtype Path_Length is Natural range 0 .. Max_IR_Path_Length;
   subtype Path_String is String (Path_Index);

   --  ============================================================
   --  Analysis Configuration
   --  ============================================================

   type Analysis_Config is record
      IR_Path        : Path_String;
      IR_Path_Len    : Path_Length;
      Output_Path    : Path_String;
      Output_Len     : Path_Length;
      Max_Depth      : Natural;
      Max_Coupling   : Natural;
      Check_Virtual  : Boolean;
      Check_Coupling : Boolean;
   end record;

   Null_Analysis_Config : constant Analysis_Config := (
      IR_Path       => (others => ' '),
      IR_Path_Len   => 0,
      Output_Path   => (others => ' '),
      Output_Len    => 0,
      Max_Depth     => Default_Max_Depth,
      Max_Coupling  => Default_Max_Coupling,
      Check_Virtual => True,
      Check_Coupling=> True
   );

   --  ============================================================
   --  Analysis Operations
   --  ============================================================

   --  Initialize analysis configuration
   procedure Initialize_Config
     (Config      : out Analysis_Config;
      IR_Path     : String;
      Output_Path : String;
      Max_Depth   : Natural := Default_Max_Depth)
   with Pre  => IR_Path'Length > 0 and IR_Path'Length <= Max_IR_Path_Length and
                Output_Path'Length > 0 and Output_Path'Length <= Max_Output_Path_Length,
        Post => Config.Max_Depth = Max_Depth;

   --  Analyze OOP structures
   procedure Analyze_OOP
     (Config : Analysis_Config;
      Result : out DO332_Result;
      Status : out DO332_Status)
   with Pre  => Config.IR_Path_Len > 0,
        Post => (if Status = Success then Result.Success);

   --  Analyze class hierarchy
   procedure Analyze_Hierarchy
     (Config : Analysis_Config;
      Result : in out DO332_Result;
      Status : out DO332_Status)
   with Pre  => Config.IR_Path_Len > 0;

   --  Verify polymorphic calls
   procedure Verify_Polymorphism
     (Result : in out DO332_Result;
      Status : out DO332_Status);

   --  Calculate coupling metrics
   procedure Calculate_Coupling
     (Result : in out DO332_Result;
      Status : out DO332_Status);

   --  ============================================================
   --  Class Operations
   --  ============================================================

   --  Add class to result
   procedure Add_Class
     (Result      : in out DO332_Result;
      Name        : String;
      Parent      : String;
      Inheritance : Inheritance_Kind;
      Depth       : Natural;
      Status      : out DO332_Status)
   with Pre  => Name'Length > 0 and Name'Length <= Max_Class_Name_Length and
                Parent'Length <= Max_Class_Name_Length and
                Result.Class_Total < Max_Class_Count,
        Post => (if Status = Success then Result.Class_Total = Result.Class_Total'Old + 1);

   --  Add polymorphic call
   procedure Add_Polymorphic_Call
     (Result       : in out DO332_Result;
      Caller_Class : String;
      Target_Method: String;
      Is_Safe      : Boolean;
      Status       : out DO332_Status)
   with Pre  => Caller_Class'Length > 0 and Caller_Class'Length <= Max_Class_Name_Length and
                Target_Method'Length > 0 and Target_Method'Length <= Max_Class_Name_Length and
                Result.Poly_Total < Max_Polymorphic_Calls,
        Post => (if Status = Success then Result.Poly_Total = Result.Poly_Total'Old + 1);

   --  ============================================================
   --  Validation Operations
   --  ============================================================

   --  Check if inheritance depth is within limits
   function Check_Depth_Limits
     (Result    : DO332_Result;
      Max_Depth : Natural) return Boolean;

   --  Check if coupling is within acceptable limits
   function Check_Coupling_Limits
     (Result      : DO332_Result;
      Max_Coupling: Natural) return Boolean;

   --  Check if all polymorphic calls are safe
   function All_Polymorphism_Safe
     (Result : DO332_Result) return Boolean;

   --  ============================================================
   --  Summary Operations
   --  ============================================================

   --  Finalize result with verification status
   procedure Finalize_Result
     (Result     : in out DO332_Result;
      Max_Depth  : Natural;
      Max_Coupling: Natural);

end DO332_Interface;
