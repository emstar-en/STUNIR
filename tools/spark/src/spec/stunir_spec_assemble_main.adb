-------------------------------------------------------------------------------
--  STUNIR Spec Assemble Main Program
--  PRIMARY IMPLEMENTATION (Ada SPARK is the default language for STUNIR tools)
--
--  Entry point for the spec_assemble command-line tool.
--
--  Copyright (c) 2026 STUNIR Project
--  License: MIT
-------------------------------------------------------------------------------

with STUNIR_Spec_Assemble;

procedure STUNIR_Spec_Assemble_Main is
begin
   STUNIR_Spec_Assemble.Run_Spec_Assemble;
end STUNIR_Spec_Assemble_Main;
