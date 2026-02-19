-------------------------------------------------------------------------------
--  STUNIR Spec to IR Main Program
--  PRIMARY IMPLEMENTATION (Ada SPARK is the default language for STUNIR tools)
--
--  Entry point for the spec_to_ir command-line tool.
--
--  Copyright (c) 2026 STUNIR Project
--  License: MIT
-------------------------------------------------------------------------------

with STUNIR_Spec_To_IR;

procedure STUNIR_Spec_To_IR_Main is
begin
   STUNIR_Spec_To_IR.Run_Spec_To_IR;
end STUNIR_Spec_To_IR_Main;
