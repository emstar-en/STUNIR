--  STUNIR DO-331 Model IR Tests
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

with Ada.Text_IO; use Ada.Text_IO;
with Model_IR; use Model_IR;

procedure Test_Model_IR is
   Element   : Model_Element;
   Container : Model_Container;
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
   Put_Line ("Model IR Tests");
   Put_Line ("==============");
   
   Reset_ID_Generator;
   
   --  Test element creation
   Element := Create_Element (Package_Element, "TestPackage");
   Assert (Element.ID /= Null_Element_ID, "Element ID generated");
   Assert (Element.Kind = Package_Element, "Element kind is Package");
   Assert (Element.Name_Length = 11, "Element name length correct");
   Assert (Get_Name (Element) = "TestPackage", "Element name correct");
   Assert (Is_Valid (Element), "Element is valid");
   Assert (Is_Structural (Package_Element), "Package is structural");
   Assert (Is_Behavioral (Action_Element), "Action is behavioral");
   
   --  Test container
   Container := Create_Container;
   Assert (Container.Element_Count = 0, "Container initially empty");
   
   Set_IR_Hash (Container, "abc123def456");
   Assert (Container.Hash_Length = 12, "IR hash set correctly");
   
   Set_Module_Name (Container, "TestModule");
   Assert (Container.Module_Name_Len = 10, "Module name set");
   
   --  Test DAL requirements
   declare
      Req_A : constant DAL_Coverage_Requirement := Get_DAL_Requirements (DAL_A);
      Req_C : constant DAL_Coverage_Requirement := Get_DAL_Requirements (DAL_C);
   begin
      Assert (Req_A.Requires_MCDC, "DAL A requires MC/DC");
      Assert (not Req_C.Requires_MCDC, "DAL C does not require MC/DC");
      Assert (Req_A.Requires_Decision_Coverage, "DAL A requires decision coverage");
   end;
   
   --  Summary
   Put_Line ("");
   Put_Line ("Results: " & Natural'Image (Test_Pass) & " passed," &
             Natural'Image (Test_Fail) & " failed");
end Test_Model_IR;
