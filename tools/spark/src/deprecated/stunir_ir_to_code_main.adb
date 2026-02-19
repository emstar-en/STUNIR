-------------------------------------------------------------------------------
--  STUNIR IR to Code Main Program
--  PRIMARY IMPLEMENTATION (Ada SPARK is the default language for STUNIR tools)
--
--  Entry point for the ir_to_code command-line tool.
--
--  Copyright (c) 2026 STUNIR Project
--  License: MIT
-------------------------------------------------------------------------------

with STUNIR_IR_To_Code;

procedure STUNIR_IR_To_Code_Main is
begin
   STUNIR_IR_To_Code.Run_IR_To_Code;
end STUNIR_IR_To_Code_Main;
