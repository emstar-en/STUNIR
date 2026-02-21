-------------------------------------------------------------------------------
--  STUNIR Toolchain Types - Ada SPARK Implementation
--  Part of Phase 2 SPARK Migration
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

package body Toolchain_Types is

   --  Get registry entry by index
   function Get_Entry (
      Reg   : Tool_Registry;
      Index : Positive) return Tool_Entry
   is
   begin
      return Reg.Entries (Index);
   end Get_Entry;

   --  Find tool by logical name (returns 0 if not found)
   function Find_Tool (
      Reg  : Tool_Registry;
      Name : Short_String) return Natural
   is
   begin
      for I in 1 .. Reg.Count loop
         if Reg.Entries (I).Logical_Name.Length = Name.Length then
            declare
               Match : Boolean := True;
            begin
               for J in 1 .. Name.Length loop
                  if Reg.Entries (I).Logical_Name.Data (J) /= Name.Data (J) then
                     Match := False;
                     exit;
                  end if;
               end loop;
               if Match then
                  return I;
               end if;
            end;
         end if;
      end loop;
      return 0;  --  Not found
   end Find_Tool;

end Toolchain_Types;
