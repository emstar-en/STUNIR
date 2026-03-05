--  Spec Modular Resolve Main - CLI wrapper
--  Micro-tool: resolves multi-part specs and emits ordered manifest
--  Phase: 1 (Spec)
--  SPARK_Mode: Off (CLI parsing)
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (Off);

with Spec_Modular_Resolve;

procedure Spec_Modular_Resolve_Main is
begin
   Spec_Modular_Resolve;
end Spec_Modular_Resolve_Main;
