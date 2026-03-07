-------------------------------------------------------------------------------
--  STUNIR Semantic IR Emitter Package Specification
--  DO-178C Level A Compliant
--  SPARK 2014 Mode
--
--  This package provides emission capabilities for Semantic IR to JSON.
--  It serializes Semantic IR modules to JSON format for storage and
--  transmission.
--
--  Key features:
--  - Emits Semantic IR modules to JSON
--  - Preserves normal form ordering
--  - Includes content hashes for confluence
--
--  Copyright (c) 2026 STUNIR Project
--  License: MIT
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

with Semantic_IR.Modules; use Semantic_IR.Modules;
with Semantic_IR.Types; use Semantic_IR.Types;
with STUNIR_Types; use STUNIR_Types;

package Semantic_IR.Emitter is

   --  =========================================================================
   --  Emitter Status
   --  =========================================================================

   type Emitter_Status is
     (Status_Success, Status_Error_Invalid_Module, Status_Error_IO);

   --  =========================================================================
   --  Emission Configuration
   --  =========================================================================

   type Emitter_Config is record
      Pretty_Print    : Boolean := True;   --  Format JSON with indentation
      Include_Hashes  : Boolean := True;   --  Include content hashes
      Include_Loc     : Boolean := True;   --  Include source locations
      Include_Annots  : Boolean := True;   --  Include semantic annotations
      Enforce_Normal  : Boolean := True;   --  Require normal form
   end record;

   --  =========================================================================
   --  Emission Results
   --  =========================================================================

   type Emitter_Stats is record
      Nodes_Emitted    : Natural;
      Declarations_Emitted : Natural;
      Statements_Emitted : Natural;
      Expressions_Emitted : Natural;
      Hashes_Computed  : Natural;
   end record;

   type Emitter_Result is record
      Success      : Boolean;
      Stats        : Emitter_Stats;
      Message      : Error_String;
   end record;

   --  =========================================================================
   --  Core Emission Procedures
   --  =========================================================================

   --  Emit a Semantic IR module to a JSON file
   procedure Emit_Module_To_File
     (Module     : in     Semantic_Module;
      Output_Path : in     Path_String;
      Config     : in     Emitter_Config;
      Result     :    out Emitter_Result)
   with
      Pre  => Is_Valid_Module (Module),
      Post => (if Result.Success then Result.Stats.Nodes_Emitted > 0);

   --  Emit a Semantic IR module to a JSON string
   procedure Emit_Module_To_String
     (Module      : in     Semantic_Module;
      JSON_Output :    out JSON_String;
      Config      : in     Emitter_Config;
      Result      :    out Emitter_Result)
   with
      Pre  => Is_Valid_Module (Module),
      Post => (if Result.Success then
                JSON_String_Strings.Length (JSON_Output) > 0);

   --  =========================================================================
   --  Utility Functions
   --  =========================================================================

   --  Escape a string for JSON output
   function Escape_JSON_String (S : String) return String;

   --  Convert a Semantic_Node_Kind to JSON string
   function Kind_To_String (Kind : Semantic_Node_Kind) return String;

   --  Convert a Type_Kind to JSON string
   function Type_Kind_To_String (Kind : Type_Kind) return String;

   --  Convert a Safety_Level to JSON string
   function Safety_Level_To_String (Level : Safety_Level) return String;

   --  Convert a Target_Category to JSON string
   function Target_Category_To_String (Cat : Target_Category) return String;

end Semantic_IR.Emitter;