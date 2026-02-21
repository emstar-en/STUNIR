--  STUNIR DO-331 SysML Emitter Tests
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

with Ada.Text_IO; use Ada.Text_IO;
with Model_IR; use Model_IR;
with IR_To_Model; use IR_To_Model;
with SysML_Emitter; use SysML_Emitter;

procedure Test_SysML_Emitter is
   Container : Model_Container;
   Storage   : Element_Storage;
   Root_ID   : Element_ID;
   Action_ID : Element_ID;
   Buffer    : Output_Buffer;
   Result    : Emitter_Result;
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
   Put_Line ("SysML Emitter Tests");
   Put_Line ("===================");
   
   Reset_ID_Generator;
   Initialize_Storage (Storage);
   Container := Create_Container;
   
   --  Create test model
   Transform_Module (
      Module_Name => "test_module",
      IR_Hash     => "sha256:test",
      Options     => Default_Options,
      Container   => Container,
      Storage     => Storage,
      Root_ID     => Root_ID
   );
   
   Transform_Function (
      Func_Name    => "process_data",
      Parent_ID    => Root_ID,
      Has_Inputs   => True,
      Has_Outputs  => True,
      Input_Count  => 1,
      Output_Count => 1,
      Storage      => Storage,
      Action_ID    => Action_ID
   );
   
   --  Test buffer operations
   Initialize_Buffer (Buffer);
   Assert (Buffer.Length = 0, "Buffer initialized empty");
   
   Append (Buffer, "test");
   Assert (Buffer.Length = 4, "Append works");
   
   Append_Line (Buffer, "line");
   Assert (Buffer.Length > 4, "Append_Line works");
   
   --  Test model emission
   Emit_Model (
      Container => Container,
      Storage   => Storage,
      Options   => Default_Emitter_Options,
      Buffer    => Buffer,
      Result    => Result
   );
   
   Assert (Result.Status = Emit_Success, "Emission successful");
   Assert (Result.Output_Length > 0, "Output generated");
   Assert (Result.Line_Count > 0, "Lines generated");
   
   --  Check content
   declare
      Content : constant String := Get_Content (Buffer);
   begin
      Assert (Content'Length > 0, "Content not empty");
      --  Check for expected keywords
      --  Note: Simple substring check would need implementation
   end;
   
   --  Summary
   Put_Line ("");
   Put_Line ("Results: " & Natural'Image (Test_Pass) & " passed," &
             Natural'Image (Test_Fail) & " failed");
end Test_SysML_Emitter;
