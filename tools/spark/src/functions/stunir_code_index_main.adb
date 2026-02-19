-------------------------------------------------------------------------------
--  STUNIR Code Index Main Program
--  PRIMARY IMPLEMENTATION (Ada SPARK is the default language for STUNIR tools)
--
--  Entry point for the code_index command-line tool.
--
--  Copyright (c) 2026 STUNIR Project
--  License: MIT
-------------------------------------------------------------------------------

with STUNIR_Code_Index;

procedure STUNIR_Code_Index_Main is
begin
   STUNIR_Code_Index.Run_Code_Index;
end STUNIR_Code_Index_Main;
