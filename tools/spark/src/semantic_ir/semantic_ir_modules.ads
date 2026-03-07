-------------------------------------------------------------------------------
--  STUNIR Semantic IR Modules Package Specification
--  DO-178C Level A Compliant
--  SPARK 2014 Mode
--
--  This package defines the module structure for the STUNIR Semantic IR.
--  Modules are the top-level containers for declarations and provide
--  semantic annotations for safety levels and target categories.
--
--  Key differences from flat IR modules:
--  - Explicit semantic annotations (safety level, target categories)
--  - Control flow graph entry points
--  - Normalized import/export ordering
--
--  Copyright (c) 2026 STUNIR Project
--  License: MIT
--
--  Normal Form: Modules must conform to the normal_form rules:
--  - Imports sorted lexicographically
--  - Exports sorted lexicographically
--  - Declarations sorted by kind, then name
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

with Semantic_IR.Types; use Semantic_IR.Types;
with Semantic_IR.Nodes; use Semantic_IR.Nodes;

package Semantic_IR.Modules with
   SPARK_Mode => On
is
   --  =========================================================================
   --  Import Statement
   --  =========================================================================

   --  Maximum symbols per import (reduced for stack usage)
   Max_Symbols : constant := 8;
   type Symbol_List is array (1 .. Max_Symbols) of IR_Name;

   type Import_Statement is record
      Module_Name  : IR_Name;
      Symbol_Count : Natural range 0 .. Max_Symbols := 0;
      Symbols      : Symbol_List;
      Import_All   : Boolean := False;
      Alias        : IR_Name;
   end record;

   --  =========================================================================
   --  Module Metadata (Semantic IR specific)
   --  =========================================================================

   --  Maximum target categories per module
   Max_Target_Categories : constant := 4;
   type Target_Category_List is array (1 .. Max_Target_Categories) of Target_Category;

   --  Module-level semantic metadata
   type Module_Metadata is record
      Target_Count       : Natural range 0 .. Max_Target_Categories := 0;
      Target_Categories  : Target_Category_List;
      Module_Safety      : Safety_Level := Level_None;
      Optimization_Level : Natural range 0 .. 3 := 0;  --  O0..O3
      Is_Entry_Point     : Boolean := False;  --  True if this is a program entry
   end record;

   --  =========================================================================
   --  Control Flow Graph Entry
   --  =========================================================================

   --  Entry point for the module's control flow graph
   type CFG_Entry is record
      Entry_Node   : Node_ID;       --  Entry node for the CFG
      Exit_Node    : Node_ID;       --  Exit node for the CFG
      Node_Count   : Natural := 0;  --  Total nodes in the CFG
   end record;

   --  =========================================================================
   --  Module Structure
   --  =========================================================================

   --  Module limits (reduced for stack usage)
   Max_Imports       : constant := 2;
   Max_Exports       : constant := 4;
   Max_Declarations  : constant := 8;

   type Import_List is array (1 .. Max_Imports) of Import_Statement;
   type Export_List is array (1 .. Max_Exports) of IR_Name;
   type Declaration_List is array (1 .. Max_Declarations) of Node_ID;

   --  Semantic IR Module structure
   type Semantic_Module is record
      --  Base node information
      Base            : Semantic_Node (Kind_Module);

      --  Module identification
      Module_Name     : IR_Name;
      Module_Hash     : IR_Hash;  --  Hash of module content

      --  Imports and exports (sorted lexicographically in normal form)
      Import_Count    : Natural range 0 .. Max_Imports := 0;
      Imports         : Import_List;
      Export_Count    : Natural range 0 .. Max_Exports := 0;
      Exports         : Export_List;

      --  Declarations (sorted by kind, then name in normal form)
      Decl_Count      : Natural range 0 .. Max_Declarations := 0;
      Declarations    : Declaration_List;

      --  Semantic metadata
      Metadata        : Module_Metadata;

      --  Control flow graph entry
      CFG             : CFG_Entry;
   end record;

   --  =========================================================================
   --  Module Validation
   --  =========================================================================

   --  Check if a module is valid
   function Is_Valid_Module (M : Semantic_Module) return Boolean
      with Post => (if Is_Valid_Module'Result then
                    Is_Valid_Semantic_Node (M.Base) and then
                    Name_Strings.Length (M.Module_Name) > 0);

   --  Check if a module is in normal form
   function Is_Normal_Form (M : Semantic_Module) return Boolean
      with Post => (if Is_Normal_Form'Result then Is_Valid_Module (M));

   --  =========================================================================
   --  Module Operations
   --  =========================================================================

   --  Add an import to the module
   procedure Add_Import (
      M       : in out Semantic_Module;
      Import  : in     Import_Statement;
      Success :    out Boolean
   ) with
      Pre  => Is_Valid_Module (M),
      Post => Is_Valid_Module (M);

   --  Add an export to the module
   procedure Add_Export (
      M       : in out Semantic_Module;
      Name    : in     IR_Name;
      Success :    out Boolean
   ) with
      Pre  => Is_Valid_Module (M) and then Name_Strings.Length (Name) > 0,
      Post => Is_Valid_Module (M);

   --  Add a declaration to the module
   procedure Add_Declaration (
      M        : in out Semantic_Module;
      Decl_ID  : in     Node_ID;
      Success  :    out Boolean
   ) with
      Pre  => Is_Valid_Module (M) and then Is_Valid_Node_ID (Decl_ID),
      Post => Is_Valid_Module (M);

   --  =========================================================================
   --  Normal Form Operations
   --  =========================================================================

   --  Sort imports lexicographically
   procedure Sort_Imports (M : in out Semantic_Module)
      with Pre  => Is_Valid_Module (M),
           Post => Is_Valid_Module (M) and then Is_Normal_Form (M);

   --  Sort exports lexicographically
   procedure Sort_Exports (M : in out Semantic_Module)
      with Pre  => Is_Valid_Module (M),
           Post => Is_Valid_Module (M);

   --  Sort declarations by kind, then name
   procedure Sort_Declarations (M : in out Semantic_Module)
      with Pre  => Is_Valid_Module (M),
           Post => Is_Valid_Module (M);

end Semantic_IR.Modules;