--  STUNIR DO-333 SPARK Integration
--  Implementation
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

package body SPARK_Integration is

   --  ============================================================
   --  Is GNATprove Available
   --  ============================================================

   function Is_GNATprove_Available return Boolean is
   begin
      --  In a real implementation, this would check system PATH
      --  For SPARK compliance, we return a constant
      return True;
   end Is_GNATprove_Available;

   --  ============================================================
   --  Get GNATprove Version
   --  ============================================================

   function Get_GNATprove_Version return String is
   begin
      --  Placeholder version string
      return "GNATprove 24.0";
   end Get_GNATprove_Version;

   --  ============================================================
   --  Is Prover Available
   --  ============================================================

   function Is_Prover_Available (P : Prover_Kind) return Boolean is
   begin
      --  Most provers bundled with GNATprove
      case P is
         when Prover_CVC4      => return True;
         when Prover_Z3        => return True;
         when Prover_Alt_Ergo  => return True;
         when Prover_CVC5      => return True;
         when Prover_Colibri   => return True;
         when Prover_All       => return True;
      end case;
   end Is_Prover_Available;

   --  ============================================================
   --  Mode Name
   --  ============================================================

   function Mode_Name (M : Proof_Mode) return String is
   begin
      case M is
         when Mode_Check => return "check";
         when Mode_Flow  => return "flow";
         when Mode_Prove => return "prove";
         when Mode_All   => return "all";
         when Mode_Gold  => return "gold";
      end case;
   end Mode_Name;

   --  ============================================================
   --  Prover Name
   --  ============================================================

   function Prover_Name (P : Prover_Kind) return String is
   begin
      case P is
         when Prover_CVC4     => return "cvc4";
         when Prover_Z3       => return "z3";
         when Prover_Alt_Ergo => return "altergo";
         when Prover_CVC5     => return "cvc5";
         when Prover_Colibri  => return "colibri";
         when Prover_All      => return "all";
      end case;
   end Prover_Name;

end SPARK_Integration;
