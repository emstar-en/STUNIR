-------------------------------------------------------------------------------
--  STUNIR Semantic IR Modules Package Body
--  DO-178C Level A Compliant
--  SPARK 2014 Mode
--
--  Implementation of module operations for Semantic IR.
--
--  Copyright (c) 2026 STUNIR Project
--  License: MIT
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

package body Semantic_IR.Modules is

   --  Check if a module is valid
   function Is_Valid_Module (M : Semantic_Module) return Boolean is
   begin
      --  Base node must be valid
      if not Is_Valid_Semantic_Node (M.Base) then
         return False;
      end if;

      --  Module name must be non-empty
      if Name_Strings.Length (M.Module_Name) = 0 then
         return False;
      end if;

      --  Import count must be within bounds
      if M.Import_Count > Max_Imports then
         return False;
      end if;

      --  Export count must be within bounds
      if M.Export_Count > Max_Exports then
         return False;
      end if;

      --  Declaration count must be within bounds
      if M.Decl_Count > Max_Declarations then
         return False;
      end if;

      --  Target count must be within bounds
      if M.Metadata.Target_Count > Max_Target_Categories then
         return False;
      end if;

      return True;
   end Is_Valid_Module;

   --  Check if a module is in normal form
   function Is_Normal_Form (M : Semantic_Module) return Boolean is
   begin
      --  Must be valid first
      if not Is_Valid_Module (M) then
         return False;
      end if;

      --  Imports must be sorted lexicographically
      for I in 2 .. M.Import_Count loop
         declare
            Prev_Name : constant String := Name_Strings.To_String (M.Imports (I - 1).Module_Name);
            Curr_Name : constant String := Name_Strings.To_String (M.Imports (I).Module_Name);
         begin
            if Curr_Name < Prev_Name then
               return False;
            end if;
         end;
      end loop;

      --  Exports must be sorted lexicographically
      for I in 2 .. M.Export_Count loop
         declare
            Prev_Name : constant String := Name_Strings.To_String (M.Exports (I - 1));
            Curr_Name : constant String := Name_Strings.To_String (M.Exports (I));
         begin
            if Curr_Name < Prev_Name then
               return False;
            end if;
         end;
      end loop;

      return True;
   end Is_Normal_Form;

   --  Add an import to the module
   procedure Add_Import (
      M       : in out Semantic_Module;
      Import  : in     Import_Statement;
      Success :    out Boolean
   ) is
   begin
      Success := False;

      --  Check if we have room
      if M.Import_Count >= Max_Imports then
         return;
      end if;

      --  Add the import
      M.Import_Count := M.Import_Count + 1;
      M.Imports (M.Import_Count) := Import;
      Success := True;
   end Add_Import;

   --  Add an export to the module
   procedure Add_Export (
      M       : in out Semantic_Module;
      Name    : in     IR_Name;
      Success :    out Boolean
   ) is
   begin
      Success := False;

      --  Check if we have room
      if M.Export_Count >= Max_Exports then
         return;
      end if;

      --  Add the export
      M.Export_Count := M.Export_Count + 1;
      M.Exports (M.Export_Count) := Name;
      Success := True;
   end Add_Export;

   --  Add a declaration to the module
   procedure Add_Declaration (
      M        : in out Semantic_Module;
      Decl_ID  : in     Node_ID;
      Success  :    out Boolean
   ) is
   begin
      Success := False;

      --  Check if we have room
      if M.Decl_Count >= Max_Declarations then
         return;
      end if;

      --  Add the declaration
      M.Decl_Count := M.Decl_Count + 1;
      M.Declarations (M.Decl_Count) := Decl_ID;
      Success := True;
   end Add_Declaration;

   --  Sort imports lexicographically (simple bubble sort for SPARK)
   procedure Sort_Imports (M : in out Semantic_Module) is
      Temp : Import_Statement;
   begin
      --  Bubble sort imports by module name
      for I in 1 .. M.Import_Count - 1 loop
         for J in I + 1 .. M.Import_Count loop
            declare
               Name_I : constant String := Name_Strings.To_String (M.Imports (I).Module_Name);
               Name_J : constant String := Name_Strings.To_String (M.Imports (J).Module_Name);
            begin
               if Name_J < Name_I then
                  --  Swap
                  Temp := M.Imports (I);
                  M.Imports (I) := M.Imports (J);
                  M.Imports (J) := Temp;
               end if;
            end;
         end loop;
      end loop;
   end Sort_Imports;

   --  Sort exports lexicographically (simple bubble sort for SPARK)
   procedure Sort_Exports (M : in out Semantic_Module) is
      Temp : IR_Name;
   begin
      --  Bubble sort exports
      for I in 1 .. M.Export_Count - 1 loop
         for J in I + 1 .. M.Export_Count loop
            declare
               Name_I : constant String := Name_Strings.To_String (M.Exports (I));
               Name_J : constant String := Name_Strings.To_String (M.Exports (J));
            begin
               if Name_J < Name_I then
                  --  Swap
                  Temp := M.Exports (I);
                  M.Exports (I) := M.Exports (J);
                  M.Exports (J) := Temp;
               end if;
            end;
         end loop;
      end loop;
   end Sort_Exports;

   --  Sort declarations by kind, then name
   procedure Sort_Declarations (M : in out Semantic_Module) is
      --  Note: Full sorting requires access to declaration nodes
      --  This is a placeholder for the sorting logic
      --  In practice, this would need to look up declaration names
      pragma Unreferenced (M);
   begin
      --  Placeholder: Declaration sorting requires node lookup
      --  This would be implemented with a symbol table
      null;
   end Sort_Declarations;

end Semantic_IR.Modules;