-------------------------------------------------------------------------------
--  STUNIR Parent Package
--  PRIMARY IMPLEMENTATION (Ada SPARK is the default language for STUNIR tools)
--
--  This is the root package for all STUNIR Ada/SPARK implementations.
--  It serves as the parent for child packages like STUNIR.Semantic_IR.
--
--  Copyright (c) 2026 STUNIR Project
--  License: MIT
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

package STUNIR is
   pragma Pure;
   
   --  Version information
   Major_Version : constant := 1;
   Minor_Version : constant := 0;
   Patch_Version : constant := 0;

end STUNIR;
