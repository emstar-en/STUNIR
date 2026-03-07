-------------------------------------------------------------------------------
--  STUNIR Semantic IR Normalizer Package Specification
--  DO-178C Level A Compliant
--  SPARK 2014 Mode
--
--  This package provides normalization passes for Semantic IR to ensure
--  confluence guarantees. All Semantic IR must be in normal form before
--  emission.
--
--  Normal form rules (from tools/spark/schema/stunir_ir_v1.dcbor.json):
--  1. Field ordering: lexicographic by field name
--  2. Array ordering: types first, then functions, alphabetically within
--  3. Alpha renaming: _t0, _t1, _t2, ... for temporaries
--  4. Literal normalization: consistent representation
--  5. Type normalization: canonical type references
--  6. Control flow normalization: explicit edges in CFG
--
--  Copyright (c) 2026 STUNIR Project
--  License: MIT
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

with Semantic_IR.Modules; use Semantic_IR.Modules;
with Semantic_IR.Types; use Semantic_IR.Types;
with STUNIR_Types; use STUNIR_Types;

package Semantic_IR.Normalizer is

   --  =========================================================================
   --  Normalization Configuration
   --  =========================================================================

   type Normalization_Pass is (
      Pass_Field_Ordering,       --  Sort fields lexicographically
      Pass_Array_Ordering,       --  Sort arrays by kind, then name
      Pass_Alpha_Renaming,       --  Rename temporaries to _t0, _t1, ...
      Pass_Literal_Normalize,    --  Normalize literal representations
      Pass_Type_Canonicalize,    --  Canonicalize type references
      Pass_CFG_Normalize,        --  Normalize control flow graph edges
      Pass_Binding_Resolve,      --  Resolve all type bindings
      Pass_Annotation_Clean,     --  Clean up semantic annotations
      Pass_Hash_Compute          --  Compute content hashes
   );

   type Pass_Enabled is array (Normalization_Pass) of Boolean;

   type Normalizer_Config is record
      Enabled_Passes : Pass_Enabled := (others => True);
      Max_Temps      : Natural := 64;  --  Max temp variables to generate
      Verbose        : Boolean := False;
      Enforce_Confluence : Boolean := True;  --  Require confluence guarantee
   end record;

   --  =========================================================================
   --  Normalization Results
   --  =========================================================================

   type Normalization_Stats is record
      Fields_Ordered      : Natural;
      Arrays_Ordered      : Natural;
      Temps_Renamed       : Natural;
      Literals_Normalized : Natural;
      Types_Canonicalized : Natural;
      CFG_Edges_Normalized : Natural;
      Bindings_Resolved   : Natural;
      Annotations_Cleaned : Natural;
      Hashes_Computed     : Natural;
      --  Validation
      Validation_Errors   : Natural;
      Validation_Warnings : Natural;
   end record;

   type Normalization_Result is record
      Success      : Boolean;
      Stats        : Normalization_Stats;
      Message      : Error_String;
      Is_Normal_Form : Boolean;  --  True if module is in normal form
   end record;

   --  =========================================================================
   --  Core Normalization Procedures
   --  =========================================================================

   --  Normalize a Semantic IR module in place
   procedure Normalize_Module
     (Module : in out Semantic_Module;
      Config : in     Normalizer_Config;
      Result :    out Normalization_Result)
   with
      Pre  => Is_Valid_Module (Module),
      Post => (if Result.Success then Is_Valid_Module (Module));

   --  Check if a module is already in normal form
   function Is_In_Normal_Form (M : Semantic_Module) return Boolean
      with Post => (if Is_In_Normal_Form'Result then Is_Valid_Module (M));

   --  Validate normal form and return detailed diagnostics
   procedure Validate_Normal_Form
     (Module    : in     Semantic_Module;
      Errors    :    out Error_String;
      Warnings  :    out Error_String;
      Is_Valid  :    out Boolean);

   --  =========================================================================
   --  Individual Pass Procedures
   --  =========================================================================

   --  Pass 1: Sort fields lexicographically
   procedure Pass_Field_Ordering
     (Module : in out Semantic_Module;
      Stats  : in out Normalization_Stats);

   --  Pass 2: Sort arrays by kind, then name
   procedure Pass_Array_Ordering
     (Module : in out Semantic_Module;
      Stats  : in out Normalization_Stats);

   --  Pass 3: Rename temporaries to _t0, _t1, ...
   procedure Pass_Alpha_Renaming
     (Module : in out Semantic_Module;
      Stats  : in out Normalization_Stats);

   --  Pass 4: Normalize literal representations
   procedure Pass_Literal_Normalize
     (Module : in out Semantic_Module;
      Stats  : in out Normalization_Stats);

   --  Pass 5: Canonicalize type references
   procedure Pass_Type_Canonicalize
     (Module : in out Semantic_Module;
      Stats  : in out Normalization_Stats);

   --  Pass 6: Normalize control flow graph edges
   procedure Pass_CFG_Normalize
     (Module : in out Semantic_Module;
      Stats  : in out Normalization_Stats);

   --  Pass 7: Resolve all type bindings
   procedure Pass_Binding_Resolve
     (Module : in out Semantic_Module;
      Stats  : in out Normalization_Stats);

   --  Pass 8: Clean up semantic annotations
   procedure Pass_Annotation_Clean
     (Module : in out Semantic_Module;
      Stats  : in out Normalization_Stats);

   --  Pass 9: Compute content hashes
   procedure Pass_Hash_Compute
     (Module : in out Semantic_Module;
      Stats  : in out Normalization_Stats);

   --  =========================================================================
   --  Utility Functions
   --  =========================================================================

   --  Generate a unique temporary name
   function Generate_Temp_Name (Index : Natural) return IR_Name
      with Post => Name_Strings.Length (Generate_Temp_Name'Result) > 0;

   --  Compute content hash for a node
   function Compute_Node_Hash (N : Semantic_Node) return IR_Hash
      with post => Hash_Strings.Length (Compute_Node_Hash'Result) = 71;

   --  Compare two modules for semantic equivalence
   function Are_Semantically_Equivalent (M1, M2 : Semantic_Module) return Boolean;

end Semantic_IR.Normalizer;