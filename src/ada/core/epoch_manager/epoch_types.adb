-------------------------------------------------------------------------------
--  STUNIR Epoch Types - Ada SPARK Implementation
--  Part of Phase 2 SPARK Migration
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

package body Epoch_Types is

   --  Convert source to string representation
   function Source_To_String (S : Epoch_Source) return String is
   begin
      case S is
         when Source_Unknown =>
            return "UNKNOWN";
         when Source_Env_Build_Epoch =>
            return "STUNIR_BUILD_EPOCH";
         when Source_Env_Source_Date =>
            return "SOURCE_DATE_EPOCH";
         when Source_Derived_Spec_Digest =>
            return "DERIVED_SPEC_DIGEST_V1";
         when Source_Git_Commit =>
            return "GIT_COMMIT_EPOCH";
         when Source_Zero =>
            return "ZERO";
         when Source_Current_Time =>
            return "CURRENT_TIME";
      end case;
   end Source_To_String;

   --  Convert string to source (parsing)
   function String_To_Source (S : String) return Epoch_Source is
   begin
      if S = "STUNIR_BUILD_EPOCH" then
         return Source_Env_Build_Epoch;
      elsif S = "SOURCE_DATE_EPOCH" then
         return Source_Env_Source_Date;
      elsif S = "DERIVED_SPEC_DIGEST_V1" then
         return Source_Derived_Spec_Digest;
      elsif S = "GIT_COMMIT_EPOCH" then
         return Source_Git_Commit;
      elsif S = "ZERO" then
         return Source_Zero;
      elsif S = "CURRENT_TIME" then
         return Source_Current_Time;
      else
         return Source_Unknown;
      end if;
   end String_To_Source;

end Epoch_Types;
