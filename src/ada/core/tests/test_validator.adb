-------------------------------------------------------------------------------
--  STUNIR IR Validator Tests - Ada
--  Part of Phase 1 SPARK Migration
-------------------------------------------------------------------------------

with Ada.Text_IO; use Ada.Text_IO;
with IR_Parser; use IR_Parser;
with IR_Validator; use IR_Validator;

procedure Test_Validator is
   Parse_Res : IR_Parser.Parse_Result;
   Valid_Res : IR_Validator.Validation_Result;
   
   Test_Count : Natural := 0;
   Pass_Count : Natural := 0;
   
   procedure Test (Name : String; Passed : Boolean) is
   begin
      Test_Count := Test_Count + 1;
      if Passed then
         Pass_Count := Pass_Count + 1;
         Put_Line ("  ✓ " & Name);
      else
         Put_Line ("  ✗ " & Name);
      end if;
   end Test;
   
begin
   Put_Line ("STUNIR IR Validator Tests");
   Put_Line ("=========================");
   Put_Line ("");
   
   -- Test Parser basics
   Put_Line ("Parser Tests:");
   
   IR_Parser.Initialize (Parse_Res);
   Test ("Initialize parse result", Parse_Res.Is_Valid);
   Test ("No initial errors", Total_Errors (Parse_Res) = 0);
   Test ("No initial warnings", Total_Warnings (Parse_Res) = 0);
   
   -- Test name creation
   declare
      Name : constant Bounded_Name := Make_Name ("test_module");
   begin
      Test ("Make name works", Name.Length = 11);
      Test ("Names equal self", Names_Equal (Name, Name));
      Test ("Names not equal different", not Names_Equal (Name, Make_Name ("other")));
   end;
   
   -- Test schema identification
   Put_Line ("");
   Put_Line ("Schema Tests:");
   
   declare
      S1 : constant Schema_Name := Make_Schema ("stunir.ir.v1");
      S2 : constant Schema_Name := Make_Schema ("invalid");
   begin
      Test ("Identify stunir.ir.v1", Identify_Schema (S1) = IR_V1_Schema);
      Test ("Invalid schema identified", Identify_Schema (S2) = Unknown_Schema);
      Test ("Valid schema check", Is_Valid_Schema (S1));
      Test ("Invalid schema check", not Is_Valid_Schema (S2));
   end;
   
   -- Test module name validation
   Put_Line ("");
   Put_Line ("Module Name Validation Tests:");
   
   Test ("Valid module name", Is_Valid_Module_Name (Make_Name ("my_module")));
   Test ("Valid module with dots", Is_Valid_Module_Name (Make_Name ("my.module.name")));
   Test ("Empty name invalid", not Is_Valid_Module_Name (Empty_Name));
   
   -- Test error/warning handling
   Put_Line ("");
   Put_Line ("Error Handling Tests:");
   
   IR_Parser.Initialize (Parse_Res);
   Add_Error (Parse_Res, Make_Name ("Test error"));
   Test ("Add error increases count", Total_Errors (Parse_Res) = 1);
   Test ("Error invalidates result", not Parse_Res.Is_Valid);
   
   IR_Parser.Initialize (Parse_Res);
   Add_Warning (Parse_Res, Make_Name ("Test warning"));
   Test ("Add warning increases count", Total_Warnings (Parse_Res) = 1);
   Test ("Warning keeps result valid", Parse_Res.Is_Valid);
   
   -- Test function registration
   Put_Line ("");
   Put_Line ("Function Registration Tests:");
   
   IR_Parser.Initialize (Parse_Res);
   Add_Function (Parse_Res, Make_Name ("main"), True, 2);
   Test ("Add function", Parse_Res.Function_Count = 1);
   Test ("Function has name", Parse_Res.Functions (1).Name.Length > 0);
   Test ("Function has body", Parse_Res.Functions (1).Has_Body);
   Test ("Function has params", Parse_Res.Functions (1).Param_Count = 2);
   
   -- Test validator
   Put_Line ("");
   Put_Line ("Validator Tests:");
   
   IR_Validator.Initialize (Valid_Res, Strict => False);
   Test ("Initialize validator", IR_Validator.Is_Valid (Valid_Res));
   
   Set_Schema (Valid_Res, Make_Schema ("stunir.ir.v1"));
   Test ("Set schema", Get_Schema (Valid_Res).Length > 0);
   
   Set_Module (Valid_Res, Make_Name ("test_module"));
   Test ("Set module", Get_Module (Valid_Res).Length > 0);
   
   Set_Epoch (Valid_Res, 12345);
   Test ("Set epoch", Valid_Res.Parse_Result.Has_Epoch);
   
   Add_Validated_Function (Valid_Res, Make_Name ("main"), True, 0);
   Test ("Add validated function", Get_Function_Count (Valid_Res) = 1);
   
   Finalize_Validation (Valid_Res);
   Test ("Validation passes", IR_Validator.Is_Valid (Valid_Res));
   
   -- Test content hash
   Put_Line ("");
   Put_Line ("Hash Tests:");
   
   declare
      H1 : constant Hash_String := Make_Hash ("a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2");
      H2 : constant Hash_String := Make_Hash ("a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2");
      H3 : constant Hash_String := Make_Hash ("0000000000000000000000000000000000000000000000000000000000000000");
   begin
      Test ("Hash equality same", Hashes_Equal (H1, H2));
      Test ("Hash inequality diff", not Hashes_Equal (H1, H3));
   end;
   
   -- Summary
   Put_Line ("");
   Put_Line ("===================");
   Put_Line ("Results:" & Natural'Image (Pass_Count) & " /" & Natural'Image (Test_Count) & " passed");
   
   if Pass_Count = Test_Count then
      Put_Line ("All tests PASSED!");
   else
      Put_Line ("Some tests FAILED.");
   end if;
   
end Test_Validator;
