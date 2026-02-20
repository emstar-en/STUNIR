-------------------------------------------------------------------------------
--  STUNIR Root Package — CANONICAL LOCATION: src/core/stunir.ads
--
--  PRIMARY IMPLEMENTATION (Ada SPARK is the default language for STUNIR tools)
--
--  This is the SOLE root package for all STUNIR Ada/SPARK implementations.
--  It serves as the parent for all child packages:
--    STUNIR.Emitters, STUNIR.Emitters.*, STUNIR.Semantic_IR, etc.
--
--  GOVERNANCE: Do NOT create another stunir.ads anywhere in the source tree.
--  The GPR Source_Dirs ordering ensures this file is found first; any duplicate
--  will cause a "duplicate unit" compile error. See stunir_tools.gpr.
--
--  REGEX_IR_REF: tools/spark/schema/stunir_regex_ir_v1.dcbor.json
--
--  Copyright (c) 2026 STUNIR Project
--  License: MIT
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

package STUNIR is
   pragma Pure;

   --  Semantic version — keep in sync with stunir_tools.gpr header comment
   Major_Version : constant := 1;
   Minor_Version : constant := 0;
   Patch_Version : constant := 0;

end STUNIR;
