--  STUNIR DO-331 Transformer Tests
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

with Ada.Text_IO; use Ada.Text_IO;
with Model_IR; use Model_IR;
with IR_To_Model; use IR_To_Model;

procedure Test_Transformer is
   Container : Model_Container;
   Storage   : Element_Storage;
   Root_ID   : Element_ID;
   Action_ID : Element_ID;
   Type_ID   : Element_ID;
   Test_Pass : Natural := 0;
   Test_Fail : Natural := 0;
   
   procedure Assert (Condition : Boolean; Test_Name : String) is
   begin
      if Condition then
         Put_Line ("  [PASS] " & Test_Name);
         Test_Pass := Test_Pass + 1;
      else
         Put_Line ("  [FAIL] " & Test_Name);
         Test_Fail := Test_Fail + 1;
      end if;
   end Assert;
   
begin
   Put_Line ("Transformer Tests");
   Put_Line ("=================");
   
   Reset_ID_Generator;
   Initialize_Storage (Storage);
   Container := Create_Container;
   
   --  Test module transformation
   Transform_Module (
      Module_Name => "flight_controller",
      IR_Hash     => "sha256:abc123",
      Options     => Default_Options,
      Container   => Container,
      Storage     => Storage,
      Root_ID     => Root_ID
   );
   
   Assert (Root_ID /= Null_Element_ID, "Module transformed to package");
   Assert (Storage.Count = 1, "Storage has one element");
   
   --  Test function transformation
   Transform_Function (
      Func_Name    => "check_altitude",
      Parent_ID    => Root_ID,
      Has_Inputs   => True,
      Has_Outputs  => True,
      Input_Count  => 2,
      Output_Count => 1,
      Storage      => Storage,
      Action_ID    => Action_ID
   );
   
   Assert (Action_ID /= Null_Element_ID, "Function transformed to action");
   Assert (Storage.Count = 2, "Storage has two elements");
   
   --  Test type transformation
   Transform_Type (
      Type_Name => "altitude_type",
      Parent_ID => Root_ID,
      Storage   => Storage,
      Attr_ID   => Type_ID
   );
   
   Assert (Type_ID /= Null_Element_ID, "Type transformed to attribute");
   Assert (Storage.Count = 3, "Storage has three elements");
   
   --  Test children retrieval
   declare
      Children : constant Element_ID_Array := Get_Children (Storage, Root_ID);
   begin
      Assert (Children'Length >= 1, "Children retrieved");
   end;
   
   --  Test rule information
   Assert (Get_Rule_Name (Rule_Function_To_Action) = "Function to Action Definition",
           "Rule name correct");
   Assert (Get_DO331_Objective (Rule_Function_To_Action) = "MB.3",
           "DO-331 objective correct");
   
   --  Summary
   Put_Line ("");
   Put_Line ("Results: " & Natural'Image (Test_Pass) & " passed," &
             Natural'Image (Test_Fail) & " failed");
end Test_Transformer;
