-------------------------------------------------------------------------------
--  STUNIR Receipt Types - Ada SPARK Implementation
--  Part of Phase 2 SPARK Migration
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

package body Receipt_Types is

   --  Status to string
   function Status_To_String (S : Receipt_Status) return String is
   begin
      case S is
         when Receipt_Created             => return "CREATED";
         when Receipt_Skipped_No_Compiler => return "SKIPPED_NO_COMPILER";
         when Receipt_Skipped_No_Source   => return "SKIPPED_NO_SOURCE";
         when Receipt_Binary_Emitted      => return "BINARY_EMITTED";
         when Receipt_Compilation_Failed  => return "COMPILATION_FAILED";
         when Receipt_Verification_Passed => return "VERIFICATION_PASSED";
         when Receipt_Verification_Failed => return "VERIFICATION_FAILED";
      end case;
   end Status_To_String;

   --  Kind to string
   function Kind_To_String (K : Receipt_Kind) return String is
   begin
      case K is
         when Kind_Build        => return "build";
         when Kind_Verification => return "verification";
         when Kind_Provenance   => return "provenance";
         when Kind_Manifest     => return "manifest";
      end case;
   end Kind_To_String;

end Receipt_Types;
