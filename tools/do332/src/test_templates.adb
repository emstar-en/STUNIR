--  STUNIR DO-332 Test Templates Implementation
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

package body Test_Templates is

   --  ============================================================
   --  Helper to create template strings
   --  ============================================================

   function Make_Name (S : String) return Name_String is
      Result : Name_String := (others => ' ');
   begin
      for I in S'Range loop
         if I - S'First + 1 <= Max_Name_Length then
            Result (I - S'First + 1) := S (I);
         end if;
      end loop;
      return Result;
   end Make_Name;

   function Make_Template (S : String) return Template_String is
      Result : Template_String := (others => ' ');
   begin
      for I in S'Range loop
         if I - S'First + 1 <= Max_Template_Length then
            Result (I - S'First + 1) := S (I);
         end if;
      end loop;
      return Result;
   end Make_Template;

   --  ============================================================
   --  Get_Inheritance_Template Implementation
   --  ============================================================

   function Get_Inheritance_Template return Test_Template is
      T : Test_Template;
   begin
      T.Kind := Inheritance_Template;
      T.Name := Make_Name ("Inheritance Chain Test");
      T.Name_Length := 21;
      
      T.Description := Make_Template ("Verify inheritance chain for {CLASS_NAME}");
      T.Desc_Length := 41;
      
      T.Setup_Template := Make_Template (
         "-- Setup: Create instance of {CLASS_NAME}" & ASCII.LF &
         "{CLASS_NAME}_Instance : {CLASS_NAME};");
      T.Setup_Length := 80;
      
      T.Action_Template := Make_Template (
         "-- Action: Verify inheritance" & ASCII.LF &
         "Depth := Get_Inheritance_Depth ({CLASS_NAME}_Instance);");
      T.Action_Length := 85;
      
      T.Assert_Template := Make_Template (
         "-- Assert: Check depth and parent" & ASCII.LF &
         "pragma Assert (Depth = {EXPECTED_DEPTH});" & ASCII.LF &
         "pragma Assert (Has_Parent ({CLASS_NAME}_Instance, {PARENT_NAME}));");
      T.Assert_Length := 140;
      
      return T;
   end Get_Inheritance_Template;

   --  ============================================================
   --  Get_Override_Template Implementation
   --  ============================================================

   function Get_Override_Template return Test_Template is
      T : Test_Template;
   begin
      T.Kind := Override_Template;
      T.Name := Make_Name ("Override Verification Test");
      T.Name_Length := 26;
      
      T.Description := Make_Template ("Verify method override for {METHOD_NAME} in {CLASS_NAME}");
      T.Desc_Length := 56;
      
      T.Setup_Template := Make_Template (
         "-- Setup: Create instances" & ASCII.LF &
         "Parent_Obj : {PARENT_CLASS};" & ASCII.LF &
         "Child_Obj : {CLASS_NAME};");
      T.Setup_Length := 80;
      
      T.Action_Template := Make_Template (
         "-- Action: Call method on both" & ASCII.LF &
         "Parent_Result := {PARENT_CLASS}.{METHOD_NAME} (Parent_Obj);" & ASCII.LF &
         "Child_Result := {CLASS_NAME}.{METHOD_NAME} (Child_Obj);");
      T.Action_Length := 150;
      
      T.Assert_Template := Make_Template (
         "-- Assert: Verify override behavior" & ASCII.LF &
         "pragma Assert (Is_Valid (Child_Result));");
      T.Assert_Length := 75;
      
      return T;
   end Get_Override_Template;

   --  ============================================================
   --  Get_Polymorphism_Template Implementation
   --  ============================================================

   function Get_Polymorphism_Template return Test_Template is
      T : Test_Template;
   begin
      T.Kind := Polymorphism_Template;
      T.Name := Make_Name ("Polymorphism Test");
      T.Name_Length := 17;
      
      T.Description := Make_Template ("Verify polymorphic behavior for {CLASS_NAME}");
      T.Desc_Length := 44;
      
      T.Setup_Template := Make_Template (
         "-- Setup: Create subclass instance, assign to parent reference" & ASCII.LF &
         "Subclass_Obj : {SUBCLASS_NAME};" & ASCII.LF &
         "Parent_Ref : access {CLASS_NAME}'Class := Subclass_Obj'Access;");
      T.Setup_Length := 160;
      
      T.Action_Template := Make_Template (
         "-- Action: Call virtual method through parent reference" & ASCII.LF &
         "Result := Parent_Ref.{METHOD_NAME};");
      T.Action_Length := 90;
      
      T.Assert_Template := Make_Template (
         "-- Assert: Subclass implementation was called" & ASCII.LF &
         "pragma Assert (Result = {EXPECTED_RESULT});");
      T.Assert_Length := 90;
      
      return T;
   end Get_Polymorphism_Template;

   --  ============================================================
   --  Get_Dispatch_Template Implementation
   --  ============================================================

   function Get_Dispatch_Template return Test_Template is
      T : Test_Template;
   begin
      T.Kind := Dispatch_Template;
      T.Name := Make_Name ("Dynamic Dispatch Test");
      T.Name_Length := 21;
      
      T.Description := Make_Template (
         "Exercise dispatch target {TARGET_CLASS}.{METHOD_NAME} at site {SITE_ID}");
      T.Desc_Length := 70;
      
      T.Setup_Template := Make_Template (
         "-- Setup: Create target class instance" & ASCII.LF &
         "Target_Obj : {TARGET_CLASS};" & ASCII.LF &
         "Base_Ref : access {BASE_CLASS}'Class := Target_Obj'Access;");
      T.Setup_Length := 130;
      
      T.Action_Template := Make_Template (
         "-- Action: Invoke through base reference to trigger dispatch" & ASCII.LF &
         "Dispatch_Result := Base_Ref.{METHOD_NAME};");
      T.Action_Length := 100;
      
      T.Assert_Template := Make_Template (
         "-- Assert: Correct target was invoked" & ASCII.LF &
         "pragma Assert ({TARGET_CLASS}_Called);" & ASCII.LF &
         "pragma Assert (Dispatch_Result = {EXPECTED_VALUE});");
      T.Assert_Length := 130;
      
      return T;
   end Get_Dispatch_Template;

   --  ============================================================
   --  Get_Coupling_Template Implementation
   --  ============================================================

   function Get_Coupling_Template return Test_Template is
      T : Test_Template;
   begin
      T.Kind := Coupling_Template;
      T.Name := Make_Name ("Coupling Metrics Test");
      T.Name_Length := 21;
      
      T.Description := Make_Template ("Verify coupling metrics for {CLASS_NAME}");
      T.Desc_Length := 40;
      
      T.Setup_Template := Make_Template (
         "-- Setup: Analyze class dependencies" & ASCII.LF &
         "Metrics := Analyze_Coupling ({CLASS_NAME});");
      T.Setup_Length := 80;
      
      T.Action_Template := Make_Template (
         "-- Action: Extract metrics" & ASCII.LF &
         "CBO := Metrics.CBO;" & ASCII.LF &
         "RFC := Metrics.RFC;");
      T.Action_Length := 65;
      
      T.Assert_Template := Make_Template (
         "-- Assert: Metrics within thresholds" & ASCII.LF &
         "pragma Assert (CBO <= CBO_Threshold);" & ASCII.LF &
         "pragma Assert (RFC <= RFC_Threshold);");
      T.Assert_Length := 110;
      
      return T;
   end Get_Coupling_Template;

   --  ============================================================
   --  Get_Lifecycle_Template Implementation
   --  ============================================================

   function Get_Lifecycle_Template return Test_Template is
      T : Test_Template;
   begin
      T.Kind := Lifecycle_Template;
      T.Name := Make_Name ("Object Lifecycle Test");
      T.Name_Length := 21;
      
      T.Description := Make_Template ("Verify constructor/destructor for {CLASS_NAME}");
      T.Desc_Length := 46;
      
      T.Setup_Template := Make_Template (
         "-- Setup: Track construction order" & ASCII.LF &
         "Construction_Log : Log_Type;");
      T.Setup_Length := 60;
      
      T.Action_Template := Make_Template (
         "-- Action: Create and destroy object" & ASCII.LF &
         "declare" & ASCII.LF &
         "   Obj : {CLASS_NAME};" & ASCII.LF &
         "begin" & ASCII.LF &
         "   null;" & ASCII.LF &
         "end;");
      T.Action_Length := 90;
      
      T.Assert_Template := Make_Template (
         "-- Assert: Constructors called in order" & ASCII.LF &
         "pragma Assert (Construction_Log (1) = "{PARENT_CLASS}");" & ASCII.LF &
         "pragma Assert (Construction_Log (2) = "{CLASS_NAME}");");
      T.Assert_Length := 160;
      
      return T;
   end Get_Lifecycle_Template;

   --  ============================================================
   --  Get_Template Implementation
   --  ============================================================

   function Get_Template (Kind : Template_Kind) return Test_Template is
   begin
      case Kind is
         when Inheritance_Template => return Get_Inheritance_Template;
         when Override_Template    => return Get_Override_Template;
         when Polymorphism_Template => return Get_Polymorphism_Template;
         when Dispatch_Template    => return Get_Dispatch_Template;
         when Coupling_Template    => return Get_Coupling_Template;
         when Lifecycle_Template   => return Get_Lifecycle_Template;
      end case;
   end Get_Template;

   --  ============================================================
   --  Apply_Substitutions Implementation
   --  ============================================================

   function Apply_Substitutions (
      Template      : String;
      Substitutions : Substitution_Array
   ) return String is
   begin
      --  Simplified: return template as-is
      --  Full implementation would replace {PLACEHOLDER} with values
      return Template;
   end Apply_Substitutions;

end Test_Templates;
