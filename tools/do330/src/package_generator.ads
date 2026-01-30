--  STUNIR DO-330 Package Generator Specification
--  Generates Certification Packages for Tool Qualification
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0
--
--  This package generates DO-330 certification packages containing:
--  - Tool Operational Requirements (TOR)
--  - Tool Qualification Plan (TQP)
--  - Tool Accomplishment Summary (TAS)
--  - Verification Cases and Procedures
--  - Configuration Index
--  - Traceability Matrices
--  - Integration data from DO-331/332/333
--
--  DO-330 Objective: T-0 through T-5 (Tool Qualification)

pragma SPARK_Mode (On);

with Templates; use Templates;
with Template_Engine; use Template_Engine;
with Data_Collector; use Data_Collector;

package Package_Generator is

   --  ============================================================
   --  Constants
   --  ============================================================

   Max_Package_Files    : constant := 32;
   Max_Error_Messages   : constant := 16;

   --  ============================================================
   --  Package Configuration
   --  ============================================================

   type Package_Config is record
      --  Tool identification
      Tool_Name       : Name_String;
      Tool_Name_Len   : Name_Length_Type;
      Tool_Version    : Version_String;
      Version_Len     : Version_Length_Type;
      TQL             : TQL_Level;
      DAL             : DAL_Level;

      --  Output configuration
      Output_Dir      : Path_String;
      Output_Dir_Len  : Path_Length_Type;
      Template_Dir    : Path_String;
      Template_Dir_Len : Path_Length_Type;

      --  Integration options
      Include_DO331   : Boolean;
      Include_DO332   : Boolean;
      Include_DO333   : Boolean;

      --  Package options
      Generate_TOR    : Boolean;
      Generate_TQP    : Boolean;
      Generate_TAS    : Boolean;
      Generate_VCP    : Boolean;
      Generate_CI     : Boolean;
      Generate_Trace  : Boolean;

      --  Author information
      Author          : Name_String;
      Author_Len      : Name_Length_Type;
      Date            : Name_String;
      Date_Len        : Name_Length_Type;
   end record;

   --  Default configuration
   Null_Package_Config : constant Package_Config := (
      Tool_Name        => (others => ' '),
      Tool_Name_Len    => 0,
      Tool_Version     => (others => ' '),
      Version_Len      => 0,
      TQL              => TQL_5,
      DAL              => DAL_E,
      Output_Dir       => (others => ' '),
      Output_Dir_Len   => 0,
      Template_Dir     => (others => ' '),
      Template_Dir_Len => 0,
      Include_DO331    => True,
      Include_DO332    => True,
      Include_DO333    => True,
      Generate_TOR     => True,
      Generate_TQP     => True,
      Generate_TAS     => True,
      Generate_VCP     => True,
      Generate_CI      => True,
      Generate_Trace   => True,
      Author           => (others => ' '),
      Author_Len       => 0,
      Date             => (others => ' '),
      Date_Len         => 0
   );

   --  ============================================================
   --  Package Generation Status
   --  ============================================================

   type Generate_Status is (
      Success,
      Config_Error,
      Template_Error,
      Data_Error,
      Output_Error,
      Validation_Error,
      IO_Error
   );

   --  ============================================================
   --  Validation Report
   --  ============================================================

   subtype Error_Index is Positive range 1 .. Max_Error_Messages;
   subtype Error_Count is Natural range 0 .. Max_Error_Messages;

   type Error_Message is record
      Code    : Natural;
      Message : Value_String;
      Msg_Len : Value_Length_Type;
   end record;

   Null_Error_Message : constant Error_Message := (
      Code    => 0,
      Message => (others => ' '),
      Msg_Len => 0
   );

   type Error_Array is array (Error_Index) of Error_Message;

   type Validation_Report is record
      Is_Valid      : Boolean;
      Error_Total   : Error_Count;
      Errors        : Error_Array;
      Warning_Total : Error_Count;
      Warnings      : Error_Array;
   end record;

   Null_Validation_Report : constant Validation_Report := (
      Is_Valid      => True,
      Error_Total   => 0,
      Errors        => (others => Null_Error_Message),
      Warning_Total => 0,
      Warnings      => (others => Null_Error_Message)
   );

   --  ============================================================
   --  Package Generator State
   --  ============================================================

   type Generator_State is record
      Config      : Package_Config;
      Data        : Tool_Data;
      Report      : Validation_Report;
      Initialized : Boolean;
      Data_Ready  : Boolean;
      Generated   : Boolean;
   end record;

   Null_Generator_State : constant Generator_State := (
      Config      => Null_Package_Config,
      Data        => Null_Tool_Data,
      Report      => Null_Validation_Report,
      Initialized => False,
      Data_Ready  => False,
      Generated   => False
   );

   --  ============================================================
   --  Initialization Operations
   --  ============================================================

   --  Initialize generator with configuration
   procedure Initialize_Generator
     (State  : out Generator_State;
      Config : Package_Config;
      Status : out Generate_Status)
   with Post => (if Status = Success then State.Initialized else not State.Initialized);

   --  Set default configuration for a tool
   procedure Set_Default_Config
     (Config    : out Package_Config;
      Tool_Name : String;
      Version   : String;
      TQL       : TQL_Level;
      DAL       : DAL_Level)
   with Pre => Tool_Name'Length > 0 and Tool_Name'Length <= Max_Name_Length and
               Version'Length > 0 and Version'Length <= Max_Version_Length;

   --  ============================================================
   --  Data Collection Operations
   --  ============================================================

   --  Collect all qualification data
   procedure Collect_Qualification_Data
     (State    : in out Generator_State;
      Base_Dir : String;
      Status   : out Generate_Status)
   with Pre => State.Initialized and
               Base_Dir'Length > 0 and Base_Dir'Length <= Max_Path_Length;

   --  Set tool data directly (for testing)
   procedure Set_Tool_Data
     (State  : in out Generator_State;
      Data   : Tool_Data;
      Status : out Generate_Status)
   with Pre => State.Initialized;

   --  ============================================================
   --  Document Generation Operations
   --  ============================================================

   --  Generate TOR (Tool Operational Requirements)
   procedure Generate_TOR
     (State  : in Out Generator_State;
      Output : out Output_Content;
      Length : out Output_Length_Type;
      Status : out Generate_Status)
   with Pre => State.Initialized and State.Data_Ready;

   --  Generate TQP (Tool Qualification Plan)
   procedure Generate_TQP
     (State  : in Out Generator_State;
      Output : out Output_Content;
      Length : out Output_Length_Type;
      Status : out Generate_Status)
   with Pre => State.Initialized and State.Data_Ready;

   --  Generate TAS (Tool Accomplishment Summary)
   procedure Generate_TAS
     (State  : in Out Generator_State;
      Output : out Output_Content;
      Length : out Output_Length_Type;
      Status : out Generate_Status)
   with Pre => State.Initialized and State.Data_Ready;

   --  Generate all documents
   procedure Generate_All_Documents
     (State  : in Out Generator_State;
      Status : out Generate_Status)
   with Pre => State.Initialized and State.Data_Ready;

   --  ============================================================
   --  Traceability Generation
   --  ============================================================

   --  Generate traceability matrix (TOR to Tests)
   procedure Generate_TOR_Traceability
     (State  : in Out Generator_State;
      Output : out Output_Content;
      Length : out Output_Length_Type;
      Status : out Generate_Status)
   with Pre => State.Initialized and State.Data_Ready;

   --  Generate DO-330 objectives traceability
   procedure Generate_DO330_Objectives_Trace
     (State  : in Out Generator_State;
      Output : out Output_Content;
      Length : out Output_Length_Type;
      Status : out Generate_Status)
   with Pre => State.Initialized and State.Data_Ready;

   --  ============================================================
   --  Configuration Index Generation
   --  ============================================================

   --  Generate configuration index
   procedure Generate_Config_Index
     (State  : in Out Generator_State;
      Output : out Output_Content;
      Length : out Output_Length_Type;
      Status : out Generate_Status)
   with Pre => State.Initialized and State.Data_Ready;

   --  ============================================================
   --  Integration Data Generation
   --  ============================================================

   --  Generate DO-331 integration summary
   procedure Generate_DO331_Summary
     (State  : in Out Generator_State;
      Output : out Output_Content;
      Length : out Output_Length_Type;
      Status : out Generate_Status)
   with Pre => State.Initialized and State.Data_Ready;

   --  Generate DO-332 integration summary
   procedure Generate_DO332_Summary
     (State  : in Out Generator_State;
      Output : out Output_Content;
      Length : out Output_Length_Type;
      Status : out Generate_Status)
   with Pre => State.Initialized and State.Data_Ready;

   --  Generate DO-333 integration summary
   procedure Generate_DO333_Summary
     (State  : in Out Generator_State;
      Output : out Output_Content;
      Length : out Output_Length_Type;
      Status : out Generate_Status)
   with Pre => State.Initialized and State.Data_Ready;

   --  ============================================================
   --  Package Validation
   --  ============================================================

   --  Validate generated package completeness
   procedure Validate_Package
     (State  : in Out Generator_State;
      Report : out Validation_Report)
   with Pre => State.Initialized;

   --  ============================================================
   --  Utility Functions
   --  ============================================================

   --  Get status message
   function Status_Message (Status : Generate_Status) return String;

   --  Check if generator is ready for generation
   function Is_Ready_For_Generation (State : Generator_State) return Boolean;

   --  Get package summary
   procedure Get_Package_Summary
     (State   : Generator_State;
      Summary : out Value_String;
      Length  : out Value_Length_Type)
   with Pre => State.Initialized;

end Package_Generator;
