--  STUNIR DO-333 Integration Types Implementation
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

package body DO333_Types is

   function Status_Message (Status : DO333_Status) return String is
   begin
      case Status is
         when Success            => return "Verification successful";
         when Prover_Not_Found   => return "Prover not found";
         when Parse_Error        => return "Parse error in source";
         when Proof_Failed       => return "Some proofs failed";
         when Timeout_Exceeded   => return "Proof timeout exceeded";
         when Resource_Exhausted => return "Prover resources exhausted";
         when Configuration_Error=> return "Configuration error";
         when IO_Error           => return "I/O error occurred";
      end case;
   end Status_Message;

   function Proof_Status_Name (Status : Proof_Status) return String is
   begin
      case Status is
         when Proven        => return "proven";
         when Unproven      => return "unproven";
         when Timeout       => return "timeout";
         when Error         => return "error";
         when Skipped       => return "skipped";
         when Not_Attempted => return "not_attempted";
      end case;
   end Proof_Status_Name;

   function VC_Kind_Name (Kind : VC_Kind) return String is
   begin
      case Kind is
         when Precondition       => return "precondition";
         when Postcondition      => return "postcondition";
         when Assert             => return "assertion";
         when Loop_Invariant     => return "loop_invariant";
         when Loop_Variant       => return "loop_variant";
         when Range_Check        => return "range_check";
         when Overflow_Check     => return "overflow_check";
         when Division_Check     => return "division_check";
         when Index_Check        => return "index_check";
         when Discriminant_Check => return "discriminant_check";
         when Type_Invariant     => return "type_invariant";
         when Contract_Case      => return "contract_case";
         when Subprogram_Variant => return "subprogram_variant";
      end case;
   end VC_Kind_Name;

end DO333_Types;
