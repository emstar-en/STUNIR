-------------------------------------------------------------------------------
--  STUNIR Semantic IR Root Package
--  DO-178C Level A Compliant
--  SPARK 2014 Mode
--
--  This is the root package for the STUNIR Semantic Intermediate Representation
--  (Semantic IR) hierarchy. Semantic IR provides a higher-level, semantically
--  enriched representation compared to the flat IR, with explicit type bindings,
--  control flow graphs, and semantic annotations.
--
--  Key differences from flat IR:
--  - Explicit type bindings (not just type names)
--  - Control flow graph representation
--  - Semantic annotations (safety levels, target categories)
--  - Normalized form with confluence guarantees
--
--  Copyright (c) 2026 STUNIR Project
--  License: MIT
--
--  Safety: This package is marked as Pure and operates in SPARK_Mode,
--          ensuring no side effects and formal verification compatibility.
--
--  Normal Form: All Semantic IR must conform to the normal_form rules
--               defined in tools/spark/schema/stunir_ir_v1.dcbor.json
--               under the semantic_ir section.
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

package Semantic_IR with
   Pure,
   SPARK_Mode => On
is
   --  Semantic IR version constants
   --  These define the current version of the Semantic IR schema

   --  Major version number (breaking changes)
   Semantic_IR_Version_Major : constant := 1;

   --  Minor version number (new features, backward compatible)
   Semantic_IR_Version_Minor : constant := 0;

   --  Patch version number (bug fixes)
   Semantic_IR_Version_Patch : constant := 0;

   --  Full schema version string in semver format
   Schema_Version : constant String := "1.0.0";

   --  Confluence guarantee: Semantic IR is always in normal form
   --  after passing through the normalizer
   Confluence_Guaranteed : constant Boolean := True;

end Semantic_IR;