--  STUNIR DO-330 Template Engine Specification
--  Template Processing for Tool Qualification Documents
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0
--
--  This package provides template processing capabilities:
--  - Load templates from files
--  - Set variable values for substitution
--  - Process templates with variable replacement
--  - Save processed output to files
--
--  Template variables use format: {{VARIABLE_NAME}}

pragma SPARK_Mode (On);

with Templates; use Templates;

package Template_Engine is

   --  ============================================================
   --  Constants
   --  ============================================================

   Max_Output_Size      : constant := 131072;  --  128 KB output limit
   Variable_Start_Delim : constant String := "{{";
   Variable_End_Delim   : constant String := "}}";

   --  ============================================================
   --  Template Context
   --  ============================================================

   subtype Template_Index is Positive range 1 .. Max_Template_Size;
   subtype Template_Length_Type is Natural range 0 .. Max_Template_Size;
   subtype Template_Content is String (1 .. Max_Template_Size);

   subtype Output_Index is Positive range 1 .. Max_Output_Size;
   subtype Output_Length_Type is Natural range 0 .. Max_Output_Size;
   subtype Output_Content is String (1 .. Max_Output_Size);

   type Template_Context is record
      Content       : Template_Content;
      Content_Len   : Template_Length_Type;
      Variables     : Variable_Array;
      Var_Count     : Variable_Count;
      Kind          : Template_Kind;
      Is_Loaded     : Boolean;
   end record;

   --  Default template context
   Null_Template_Context : constant Template_Context := (
      Content     => (others => ' '),
      Content_Len => 0,
      Variables   => (others => Null_Template_Variable),
      Var_Count   => 0,
      Kind        => TOR_Template,
      Is_Loaded   => False
   );

   --  ============================================================
   --  Template Processing Status
   --  ============================================================

   type Process_Status is (
      Success,
      Template_Not_Loaded,
      Variable_Not_Found,
      Output_Buffer_Overflow,
      Invalid_Template_Format,
      File_Read_Error,
      File_Write_Error
   );

   --  ============================================================
   --  Template Operations
   --  ============================================================

   --  Initialize a template context
   procedure Initialize_Context
     (Context : out Template_Context;
      Kind    : Template_Kind)
   with Post => not Context.Is_Loaded and
                Context.Var_Count = 0 and
                Context.Kind = Kind;

   --  Load template content from a string
   procedure Load_Template_Content
     (Context : in out Template_Context;
      Content : String;
      Status  : out Process_Status)
   with Pre  => Content'Length > 0 and Content'Length <= Max_Template_Size,
        Post => (if Status = Success then Context.Is_Loaded else not Context.Is_Loaded);

   --  Set a variable value in the context
   procedure Set_Variable
     (Context : in out Template_Context;
      Name    : String;
      Value   : String;
      Status  : out Process_Status)
   with Pre => Name'Length > 0 and Name'Length <= Max_Name_Length and
               Value'Length <= Max_Value_Length;

   --  Clear all variables from context
   procedure Clear_Variables
     (Context : in out Template_Context)
   with Post => Context.Var_Count = 0;

   --  Process template and generate output
   procedure Process_Template
     (Context : Template_Context;
      Output  : out Output_Content;
      Out_Len : out Output_Length_Type;
      Status  : out Process_Status)
   with Pre => Context.Is_Loaded;

   --  ============================================================
   --  Standard Variables
   --  ============================================================

   --  Set standard DO-330 variables
   procedure Set_DO330_Standard_Variables
     (Context    : in out Template_Context;
      Tool_Name  : String;
      Version    : String;
      TQL        : TQL_Level;
      DAL        : DAL_Level;
      Author     : String;
      Date       : String;
      Status     : out Process_Status)
   with Pre => Tool_Name'Length > 0 and Tool_Name'Length <= Max_Name_Length and
               Version'Length > 0 and Version'Length <= 32 and
               Author'Length > 0 and Author'Length <= 128 and
               Date'Length = 10;  --  YYYY-MM-DD format

   --  ============================================================
   --  Utility Functions
   --  ============================================================

   --  Find variable in context
   function Find_Variable
     (Context : Template_Context;
      Name    : String) return Variable_Count
   with Pre => Name'Length > 0 and Name'Length <= Max_Name_Length;

   --  Check if variable exists
   function Variable_Exists
     (Context : Template_Context;
      Name    : String) return Boolean
   with Pre => Name'Length > 0 and Name'Length <= Max_Name_Length;

   --  Get status message
   function Status_Message (Status : Process_Status) return String;

end Template_Engine;
