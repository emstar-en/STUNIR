-------------------------------------------------------------------------------
--  STUNIR Dependency Resolver - Ada SPARK Implementation
--  Part of Phase 2 SPARK Migration
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

package body Dependency_Resolver is

   --  Initialize dependency registry
   procedure Initialize_Registry (Reg : out Dependency_Registry) is
   begin
      Reg := Empty_Dep_Registry;
   end Initialize_Registry;

   --  Add dependency to registry
   procedure Add_Dependency (
      Reg       : in out Dependency_Registry;
      Name      : Short_String;
      Kind      : Dependency_Kind;
      Optional  : Boolean := False;
      Success   : out Boolean)
   is
   begin
      if Reg.Count >= Max_Dependencies then
         Success := False;
         return;
      end if;
      
      Reg.Count := Reg.Count + 1;
      Reg.Entries (Reg.Count).Name := Name;
      Reg.Entries (Reg.Count).Kind := Kind;
      Reg.Entries (Reg.Count).Is_Optional := Optional;
      Reg.Entries (Reg.Count).Status := Dep_Unknown;
      Success := True;
   end Add_Dependency;

   --  Resolve a single dependency (stub - would use OS calls)
   procedure Resolve_Dependency (
      Dep     : in out Dependency_Entry;
      Success : out Boolean)
   is
   begin
      --  SPARK-safe stub implementation
      --  Real implementation would:
      --  1. Find tool in PATH or check file existence
      --  2. Compute hash
      --  3. Get version
      --  4. Compare with expectations
      
      --  For now, mark as not found (conservative)
      Dep.Status := Dep_Not_Found;
      Dep.Is_Accepted := False;
      Success := Dep.Is_Optional;
   end Resolve_Dependency;

   --  Resolve all dependencies in registry
   procedure Resolve_All (
      Reg     : in out Dependency_Registry;
      Success : out Boolean)
   is
      Dep_Success : Boolean;
   begin
      Success := True;
      Reg.Accepted_Count := 0;
      Reg.Rejected_Count := 0;
      
      for I in 1 .. Reg.Count loop
         Resolve_Dependency (Reg.Entries (I), Dep_Success);
         
         if Reg.Entries (I).Is_Accepted then
            Reg.Accepted_Count := Reg.Accepted_Count + 1;
         elsif not Reg.Entries (I).Is_Optional then
            Reg.Rejected_Count := Reg.Rejected_Count + 1;
            Success := False;
         end if;
      end loop;
   end Resolve_All;

   --  Parse acceptance receipt from JSON path (stub)
   procedure Parse_Acceptance_Receipt (
      Receipt_Path : Path_String;
      Receipt      : out Acceptance_Receipt;
      Success      : out Boolean)
   is
      pragma Unreferenced (Receipt_Path);
   begin
      --  SPARK-safe stub: would parse JSON file
      Receipt := (
         Receipt_Type  => Make_Short ("dependency_acceptance"),
         Accepted      => False,
         Tool_Name     => Empty_Short,
         Resolved_Path => Empty_Path,
         Resolved_Abs  => Empty_Path
      );
      Success := False;
   end Parse_Acceptance_Receipt;

   --  Check if receipt indicates acceptance
   function Is_Receipt_Accepted (Receipt : Acceptance_Receipt) return Boolean is
   begin
      return Receipt.Accepted;
   end Is_Receipt_Accepted;

   --  Get tool path from receipt
   function Get_Receipt_Tool_Path (
      Receipt : Acceptance_Receipt) return Path_String
   is
   begin
      --  Prefer absolute path
      if Receipt.Resolved_Abs.Length > 0 then
         return Receipt.Resolved_Abs;
      else
         return Receipt.Resolved_Path;
      end if;
   end Get_Receipt_Tool_Path;

   --  Verify dependency hash
   procedure Verify_Hash (
      Dep     : in out Dependency_Entry;
      Matches : out Boolean)
   is
   begin
      --  Compare expected vs actual hash
      Matches := Hashes_Equal (Dep.Expected_Hash, Dep.Actual_Hash);
      if not Matches and Dep.Expected_Hash /= Zero_Hash then
         Dep.Status := Dep_Hash_Mismatch;
      end if;
   end Verify_Hash;

   --  Verify dependency version (stub)
   procedure Verify_Version (
      Dep     : in out Dependency_Entry;
      Matches : out Boolean)
   is
   begin
      --  Simple check: if version spec is empty, accept any version
      if Dep.Version_Spec.Length = 0 then
         Matches := True;
         return;
      end if;
      
      --  Otherwise, require exact match (simplified)
      Matches := Dep.Version_Spec.Length = Dep.Actual_Version.Length;
      if Matches then
         for I in 1 .. Dep.Version_Spec.Length loop
            if Dep.Version_Spec.Data (I) /= Dep.Actual_Version.Data (I) then
               Matches := False;
               exit;
            end if;
         end loop;
      end if;
      
      if not Matches then
         Dep.Status := Dep_Version_Mismatch;
      end if;
   end Verify_Version;

   --  Find dependency by name
   function Find_Dependency (
      Reg  : Dependency_Registry;
      Name : Short_String) return Natural
   is
   begin
      for I in 1 .. Reg.Count loop
         if Reg.Entries (I).Name.Length = Name.Length then
            declare
               Match : Boolean := True;
            begin
               for J in 1 .. Name.Length loop
                  if Reg.Entries (I).Name.Data (J) /= Name.Data (J) then
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
      return 0;
   end Find_Dependency;

   --  Get dependency by index
   function Get_Dependency (
      Reg   : Dependency_Registry;
      Index : Positive) return Dependency_Entry
   is
   begin
      return Reg.Entries (Index);
   end Get_Dependency;

end Dependency_Resolver;
