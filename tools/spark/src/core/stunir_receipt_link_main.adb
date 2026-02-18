-------------------------------------------------------------------------------
--  STUNIR Receipt Link Main Program
--  PRIMARY IMPLEMENTATION (Ada SPARK is the default language for STUNIR tools)
--
--  Entry point for the receipt_link command-line tool.
--
--  Copyright (c) 2026 STUNIR Project
--  License: MIT
-------------------------------------------------------------------------------

with STUNIR_Receipt_Link;

procedure STUNIR_Receipt_Link_Main is
begin
   STUNIR_Receipt_Link.Run_Receipt_Link;
end STUNIR_Receipt_Link_Main;
