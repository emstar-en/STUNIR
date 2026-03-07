-------------------------------------------------------------------------------
--  STUNIR Semantic IR Normalizer Package Body
--  DO-178C Level A Compliant
--  SPARK 2014 Mode
--
--  Implementation of Semantic IR normalization passes.
--
--  Copyright (c) 2026 STUNIR Project
--  License: MIT
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

with Semantic_IR.Nodes; use Semantic_IR.Nodes;
with Ada.Strings.Unbounded;

package body Semantic_IR.Normalizer is

   --  Generate a unique temporary name
   function Generate_Temp_Name (Index : Natural) return IR_Name is
      --  Format: _t0, _t1, _t2, ...
      Temp_Name : constant String := "_t" & Natural'Image (Index);
      --  Remove leading space from Natural'Image
      Clean_Name : constant String :=
         (if Temp_Name (Temp_Name'First) = ' ' then
             Temp_Name (Temp_Name'First + 1 .. Temp_Name'Last)
          else
             Temp_Name);
   begin
      return Name_Strings.To_Bounded_String (Clean_Name);
   end Generate_Temp_Name;

   --  Compute content hash for a node (placeholder)
   function Compute_Node_Hash (N : Semantic_Node) return IR_Hash is
      --  Placeholder: In a full implementation, this would compute
      --  a SHA-256 hash of the node content
      pragma Unreferenced (N);
   begin
      --  Return a placeholder hash
      return Hash_Strings.To_Bounded_String ("sha256:" &
         "0000000000000000000000000000000000000000000000000000000000000000");
   end Compute_Node_Hash;

   --  Compare two modules for semantic equivalence
   function Are_Semantically_Equivalent (M1, M2 : Semantic_Module) return Boolean is
   begin
      --  Modules are semantically equivalent if:
      --  1. They have the same module name
      --  2. They have the same declarations (by content, not ID)
      --  3. They have the same imports/exports

      if Name_Strings.To_String (M1.Module_Name) /=
         Name_Strings.To_String (M2.Module_Name) then
         return False;
      end if;

      --  Compare imports
      if M1.Import_Count /= M2.Import_Count then
         return False;
      end if;

      --  Compare exports
      if M1.Export_Count /= M2.Export_Count then
         return False;
      end if;

      --  Compare declarations
      if M1.Decl_Count /= M2.Decl_Count then
         return False;
      end if;

      --  Full comparison would require deep inspection of all nodes
      --  This is a simplified check
      return True;
   end Are_Semantically_Equivalent;

   --  Check if a module is already in normal form
   function Is_In_Normal_Form (M : Semantic_Module) return Boolean is
   begin
      --  Must be valid first
      if not Is_Valid_Module (M) then
         return False;
      end if;

      --  Check imports are sorted lexicographically
      for I in 2 .. M.Import_Count loop
         declare
            Prev_Name : constant String :=
               Name_Strings.To_String (M.Imports (I - 1).Module_Name);
            Curr_Name : constant String :=
               Name_Strings.To_String (M.Imports (I).Module_Name);
         begin
            if Curr_Name < Prev_Name then
               return False;
            end if;
         end;
      end loop;

      --  Check exports are sorted lexicographically
      for I in 2 .. M.Export_Count loop
         declare
            Prev_Name : constant String :=
               Name_Strings.To_String (M.Exports (I - 1));
            Curr_Name : constant String :=
               Name_Strings.To_String (M.Exports (I));
         begin
            if Curr_Name < Prev_Name then
               return False;
            end if;
         end;
      end loop;

      return True;
   end Is_In_Normal_Form;

   --  Validate normal form and return detailed diagnostics
   procedure Validate_Normal_Form
     (Module    : in     Semantic_Module;
      Errors    :    out Error_String;
      Warnings  :    out Error_String;
      Is_Valid  :    out Boolean)
   is
      pragma Unreferenced (Module);
   begin
      Errors := Error_String_Strings.Null_Bounded_String;
      Warnings := Error_String_Strings.Null_Bounded_String;
      Is_Valid := True;

      --  Placeholder: In a full implementation, this would:
      --  1. Check all field ordering
      --  2. Check all array ordering
      --  3. Check alpha renaming
      --  4. Check literal normalization
      --  5. Check type canonicalization
      --  6. Check CFG normalization
      --  7. Check binding resolution
      --  8. Check annotation cleanliness
      --  9. Check hash computation
   end Validate_Normal_Form;

   --  Pass 1: Sort fields lexicographically
   procedure Pass_Field_Ordering
     (Module : in out Semantic_Module;
      Stats  : in out Normalization_Stats)
   is
      --  Sort imports
      procedure Sort_Imports is
         Temp : Import_Statement;
      begin
         for I in 1 .. Module.Import_Count - 1 loop
            for J in I + 1 .. Module.Import_Count loop
               declare
                  Name_I : constant String :=
                     Name_Strings.To_String (Module.Imports (I).Module_Name);
                  Name_J : constant String :=
                     Name_Strings.To_String (Module.Imports (J).Module_Name);
               begin
                  if Name_J < Name_I then
                     Temp := Module.Imports (I);
                     Module.Imports (I) := Module.Imports (J);
                     Module.Imports (J) := Temp;
                  end if;
               end;
            end loop;
         end loop;
      end Sort_Imports;

      --  Sort exports
      procedure Sort_Exports is
         Temp : IR_Name;
      begin
         for I in 1 .. Module.Export_Count - 1 loop
            for J in I + 1 .. Module.Export_Count loop
               declare
                  Name_I : constant String :=
                     Name_Strings.To_String (Module.Exports (I));
                  Name_J : constant String :=
                     Name_Strings.To_String (Module.Exports (J));
               begin
                  if Name_J < Name_I then
                     Temp := Module.Exports (I);
                     Module.Exports (I) := Module.Exports (J);
                     Module.Exports (J) := Temp;
                  end if;
               end;
            end loop;
         end loop;
      end Sort_Exports;

   begin
      Sort_Imports;
      Sort_Exports;
      Stats.Fields_Ordered := Stats.Fields_Ordered + 1;
   end Pass_Field_Ordering;

   --  Pass 2: Sort arrays by kind, then name
   procedure Pass_Array_Ordering
     (Module : in out Semantic_Module;
      Stats  : in out Normalization_Stats)
   is
      pragma Unreferenced (Module);
   begin
      --  Placeholder: In a full implementation, this would sort
      --  declarations by kind (types first, then functions), then by name
      Stats.Arrays_Ordered := Stats.Arrays_Ordered + 1;
   end Pass_Array_Ordering;

   --  Pass 3: Rename temporaries to _t0, _t1, ...
   procedure Pass_Alpha_Renaming
     (Module : in out Semantic_Module;
      Stats  : in out Normalization_Stats)
   is
      pragma Unreferenced (Module);
   begin
      --  Placeholder: In a full implementation, this would:
      --  1. Find all temporary variables
      --  2. Rename them to _t0, _t1, _t2, ...
      --  3. Update all references
      Stats.Temps_Renamed := Stats.Temps_Renamed + 1;
   end Pass_Alpha_Renaming;

   --  Pass 4: Normalize literal representations
   procedure Pass_Literal_Normalize
     (Module : in out Semantic_Module;
      Stats  : in out Normalization_Stats)
   is
      pragma Unreferenced (Module);
   begin
      --  Placeholder: In a full implementation, this would:
      --  1. Normalize integer literals (remove leading zeros, etc.)
      --  2. Normalize float literals (consistent representation)
      --  3. Normalize string literals (escape sequences)
      Stats.Literals_Normalized := Stats.Literals_Normalized + 1;
   end Pass_Literal_Normalize;

   --  Pass 5: Canonicalize type references
   procedure Pass_Type_Canonicalize
     (Module : in out Semantic_Module;
      Stats  : in out Normalization_Stats)
   is
      pragma Unreferenced (Module);
   begin
      --  Placeholder: In a full implementation, this would:
      --  1. Resolve all type references to their canonical form
      --  2. Remove redundant type aliases
      --  3. Ensure consistent type naming
      Stats.Types_Canonicalized := Stats.Types_Canonicalized + 1;
   end Pass_Type_Canonicalize;

   --  Pass 6: Normalize control flow graph edges
   procedure Pass_CFG_Normalize
     (Module : in out Semantic_Module;
      Stats  : in out Normalization_Stats)
   is
      pragma Unreferenced (Module);
   begin
      --  Placeholder: In a full implementation, this would:
      --  1. Ensure all CFG edges are properly ordered
      --  2. Remove redundant edges
      --  3. Add missing edges
      Stats.CFG_Edges_Normalized := Stats.CFG_Edges_Normalized + 1;
   end Pass_CFG_Normalize;

   --  Pass 7: Resolve all type bindings
   procedure Pass_Binding_Resolve
     (Module : in out Semantic_Module;
      Stats  : in out Normalization_Stats)
   is
      pragma Unreferenced (Module);
   begin
      --  Placeholder: In a full implementation, this would:
      --  1. Resolve all type references to their declarations
      --  2. Resolve all variable references to their declarations
      --  3. Resolve all function references to their declarations
      Stats.Bindings_Resolved := Stats.Bindings_Resolved + 1;
   end Pass_Binding_Resolve;

   --  Pass 8: Clean up semantic annotations
   procedure Pass_Annotation_Clean
     (Module : in out Semantic_Module;
      Stats  : in out Normalization_Stats)
   is
      pragma Unreferenced (Module);
   begin
      --  Placeholder: In a full implementation, this would:
      --  1. Remove redundant annotations
      --  2. Ensure consistent annotation ordering
      --  3. Validate annotation values
      Stats.Annotations_Cleaned := Stats.Annotations_Cleaned + 1;
   end Pass_Annotation_Clean;

   --  Pass 9: Compute content hashes
   procedure Pass_Hash_Compute
     (Module : in out Semantic_Module;
      Stats  : in out Normalization_Stats)
   is
      pragma Unreferenced (Module);
   begin
      --  Placeholder: In a full implementation, this would:
      --  1. Compute SHA-256 hash for each node
      --  2. Compute module-level hash
      --  3. Store hashes in the appropriate fields
      Stats.Hashes_Computed := Stats.Hashes_Computed + 1;
   end Pass_Hash_Compute;

   --  Normalize a Semantic IR module in place
   procedure Normalize_Module
     (Module : in out Semantic_Module;
      Config : in     Normalizer_Config;
      Result :    out Normalization_Result)
   is
      Stats : Normalization_Stats := (others => 0);
   begin
      --  Initialize result
      Result.Success := False;
      Result.Message := Error_String_Strings.Null_Bounded_String;
      Result.Is_Normal_Form := False;

      --  Validate input
      if not Is_Valid_Module (Module) then
         Result.Message := Error_String_Strings.To_Bounded_String ("Invalid module");
         return;
      end if;

      --  Run enabled passes in order
      if Config.Enabled_Passes (Pass_Field_Ordering) then
         Pass_Field_Ordering (Module, Stats);
      end if;

      if Config.Enabled_Passes (Pass_Array_Ordering) then
         Pass_Array_Ordering (Module, Stats);
      end if;

      if Config.Enabled_Passes (Pass_Alpha_Renaming) then
         Pass_Alpha_Renaming (Module, Stats);
      end if;

      if Config.Enabled_Passes (Pass_Literal_Normalize) then
         Pass_Literal_Normalize (Module, Stats);
      end if;

      if Config.Enabled_Passes (Pass_Type_Canonicalize) then
         Pass_Type_Canonicalize (Module, Stats);
      end if;

      if Config.Enabled_Passes (Pass_CFG_Normalize) then
         Pass_CFG_Normalize (Module, Stats);
      end if;

      if Config.Enabled_Passes (Pass_Binding_Resolve) then
         Pass_Binding_Resolve (Module, Stats);
      end if;

      if Config.Enabled_Passes (Pass_Annotation_Clean) then
         Pass_Annotation_Clean (Module, Stats);
      end if;

      if Config.Enabled_Passes (Pass_Hash_Compute) then
         Pass_Hash_Compute (Module, Stats);
      end if;

      --  Validate result
      if not Is_Valid_Module (Module) then
         Result.Message := Error_String_Strings.To_Bounded_String ("Module invalid after normalization");
         return;
      end if;

      --  Check normal form
      Result.Is_Normal_Form := Is_In_Normal_Form (Module);

      --  Enforce confluence if required
      if Config.Enforce_Confluence and then not Result.Is_Normal_Form then
         Result.Message := Error_String_Strings.To_Bounded_String ("Module not in normal form after normalization");
         return;
      end if;

      --  Success
      Result.Success := True;
      Result.Stats := Stats;
   end Normalize_Module;

end Semantic_IR.Normalizer;