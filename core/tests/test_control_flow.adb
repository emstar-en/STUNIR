-------------------------------------------------------------------------------
--  STUNIR Control Flow Tests - Ada
--  Part of Phase 1 SPARK Migration
-------------------------------------------------------------------------------

with Ada.Text_IO; use Ada.Text_IO;
with IR_Basic_Blocks; use IR_Basic_Blocks;
with IR_Control_Flow; use IR_Control_Flow;

procedure Test_Control_Flow is
   CFG : Control_Flow_Graph;
   Entry_Id, Exit_Id, Block1, Block2, Block3 : Block_Id;
   
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
   Put_Line ("STUNIR Control Flow Tests");
   Put_Line ("=========================");
   Put_Line ("");
   
   -- Test CFG creation
   Put_Line ("CFG Creation Tests:");
   
   Create_Entry (CFG, Entry_Id);
   Test ("Create entry block", Entry_Id /= No_Block);
   Test ("Entry block exists", Block_Exists (CFG, Entry_Id));
   Test ("Entry block is entry type", Is_Entry (Get_Block (CFG, Entry_Id)));
   
   Create_Exit (CFG, Exit_Id);
   Test ("Create exit block", Exit_Id /= No_Block);
   Test ("Exit block exists", Block_Exists (CFG, Exit_Id));
   
   Create_Block (CFG, Normal_Block, Block1);
   Test ("Create normal block", Block1 /= No_Block);
   
   Create_Block (CFG, Conditional_Block, Block2);
   Test ("Create conditional block", Block2 /= No_Block);
   Test ("Block is conditional", Is_Conditional (Get_Block (CFG, Block2)));
   
   Create_Block (CFG, Loop_Header_Block, Block3);
   Test ("Create loop header", Block3 /= No_Block);
   Test ("Block is loop related", Is_Loop_Related (Get_Block (CFG, Block3)));
   
   -- Test edge creation
   Put_Line ("");
   Put_Line ("Edge Tests:");
   
   Add_Edge (CFG, Entry_Id, Block1);
   Test ("Add edge entry->block1", Has_Edge (CFG, Entry_Id, Block1));
   
   Add_Edge (CFG, Block1, Block2);
   Test ("Add edge block1->block2", Has_Edge (CFG, Block1, Block2));
   
   Add_Edge (CFG, Block2, Block3);
   Add_Edge (CFG, Block2, Exit_Id);
   Test ("Block2 has two successors", Get_Block (CFG, Block2).Successor_Count >= 2);
   
   Add_Edge (CFG, Block3, Block2);  -- Back edge for loop
   Add_Edge (CFG, Block3, Exit_Id);
   
   -- Test dominator computation
   Put_Line ("");
   Put_Line ("Dominator Tests:");
   
   Compute_Dominators (CFG);
   
   Test ("Entry dominates entry", Dominates (CFG, Entry_Id, Entry_Id));
   Test ("Entry dominates block1", Dominates (CFG, Entry_Id, Block1));
   Test ("Entry dominates block2", Dominates (CFG, Entry_Id, Block2));
   Test ("Entry dominates exit", Dominates (CFG, Entry_Id, Exit_Id));
   
   -- Test loop detection
   Put_Line ("");
   Put_Line ("Loop Detection Tests:");
   
   Detect_Loops (CFG);
   
   Test ("Loop count > 0", Get_Loop_Count (CFG) > 0);
   
   if Get_Loop_Count (CFG) > 0 then
      declare
         L : constant Loop_Info := Get_Loop (CFG, 1);
      begin
         Test ("Loop has header", L.Header_Id /= No_Block);
         Test ("Loop has back edge source", L.Back_Edge_Source /= No_Block);
      end;
   end if;
   
   -- Test branch detection
   Put_Line ("");
   Put_Line ("Branch Detection Tests:");
   
   Detect_Branches (CFG);
   
   Test ("Branch count > 0", Get_Branch_Count (CFG) > 0);
   
   -- Summary
   Put_Line ("");
   Put_Line ("===================");
   Put_Line ("Results:" & Natural'Image (Pass_Count) & " /" & Natural'Image (Test_Count) & " passed");
   
   if Pass_Count = Test_Count then
      Put_Line ("All tests PASSED!");
   else
      Put_Line ("Some tests FAILED.");
   end if;
   
end Test_Control_Flow;
