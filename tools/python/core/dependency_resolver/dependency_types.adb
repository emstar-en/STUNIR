-------------------------------------------------------------------------------
--  STUNIR Dependency Types - Ada SPARK Implementation
--  Part of Phase 2 SPARK Migration
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

package body Dependency_Types is

   --  Check if all required dependencies are accepted
   function All_Required_Accepted (Reg : Dependency_Registry) return Boolean is
   begin
      for I in 1 .. Reg.Count loop
         if not Reg.Entries (I).Is_Optional and 
            not Reg.Entries (I).Is_Accepted 
         then
            return False;
         end if;
      end loop;
      return True;
   end All_Required_Accepted;

   --  Status to string
   function Status_To_String (S : Dependency_Status) return String is
   begin
      case S is
         when Dep_Unknown          => return "unknown";
         when Dep_Accepted         => return "accepted";
         when Dep_Rejected         => return "rejected";
         when Dep_Not_Found        => return "not_found";
         when Dep_Version_Mismatch => return "version_mismatch";
         when Dep_Hash_Mismatch    => return "hash_mismatch";
      end case;
   end Status_To_String;

end Dependency_Types;
