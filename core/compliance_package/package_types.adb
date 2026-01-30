--  STUNIR Compliance Package Types Implementation
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

package body Package_Types is

   function Status_Message (Status : Package_Status) return String is
   begin
      case Status is
         when Success           => return "Package generation successful";
         when Invalid_Config    => return "Invalid package configuration";
         when Artifact_Missing  => return "Required artifact missing";
         when Hash_Mismatch     => return "Artifact hash mismatch";
         when Trace_Incomplete  => return "Traceability incomplete";
         when Generation_Failed => return "Package generation failed";
         when Validation_Failed => return "Package validation failed";
         when IO_Error          => return "I/O error occurred";
      end case;
   end Status_Message;

   function TQL_Name (Level : TQL_Level) return String is
   begin
      case Level is
         when TQL_1 => return "TQL-1";
         when TQL_2 => return "TQL-2";
         when TQL_3 => return "TQL-3";
         when TQL_4 => return "TQL-4";
         when TQL_5 => return "TQL-5";
      end case;
   end TQL_Name;

   function DAL_Name (Level : DAL_Level) return String is
   begin
      case Level is
         when DAL_A => return "DAL-A";
         when DAL_B => return "DAL-B";
         when DAL_C => return "DAL-C";
         when DAL_D => return "DAL-D";
         when DAL_E => return "DAL-E";
      end case;
   end DAL_Name;

   function Artifact_Kind_Name (Kind : Artifact_Kind) return String is
   begin
      case Kind is
         when Source_Code   => return "source";
         when Object_Code   => return "object";
         when Executable    => return "executable";
         when Test_Case     => return "test_case";
         when Test_Result   => return "test_result";
         when Coverage_Data => return "coverage";
         when Proof_Result  => return "proof";
         when Document      => return "document";
         when Configuration => return "config";
         when Receipt       => return "receipt";
         when Manifest      => return "manifest";
      end case;
   end Artifact_Kind_Name;

   function Trace_Kind_Name (Kind : Trace_Kind) return String is
   begin
      case Kind is
         when Req_To_Design   => return "req_to_design";
         when Design_To_Code  => return "design_to_code";
         when Code_To_Test    => return "code_to_test";
         when Test_To_Result  => return "test_to_result";
         when Req_To_Test     => return "req_to_test";
         when Code_To_Proof   => return "code_to_proof";
      end case;
   end Trace_Kind_Name;

end Package_Types;
