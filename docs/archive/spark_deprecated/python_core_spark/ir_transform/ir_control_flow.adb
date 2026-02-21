-------------------------------------------------------------------------------
--  STUNIR IR Control Flow - Ada SPARK Body
--  Part of Phase 1 SPARK Migration
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

package body IR_Control_Flow is

   -------------------------------------------------------------------------
   --  Has_Edge: Check if an edge exists between two blocks
   -------------------------------------------------------------------------
   function Has_Edge (CFG : Control_Flow_Graph; From_Id, To_Id : Block_Id) return Boolean
   is
   begin
      return Has_Successor (CFG.Blocks (From_Id), To_Id);
   end Has_Edge;

   -------------------------------------------------------------------------
   --  Create_Entry: Create the entry block
   -------------------------------------------------------------------------
   procedure Create_Entry (CFG : in out Control_Flow_Graph; Id : out Block_Id) is
   begin
      Id := CFG.Next_Block_Id;
      CFG.Blocks (Id) := (
         Id              => Id,
         Block_Kind      => Entry_Block,
         Successors      => (others => No_Block),
         Successor_Count => 0,
         Predecessors    => (others => No_Block),
         Predecessor_Count => 0,
         Dominators      => (others => False),
         Immediate_Dom   => No_Block,
         Loop_Depth      => 0,
         Is_Loop_Header  => False,
         Statement_Count => 0
      );
      CFG.Entry_Block := Id;
      CFG.Block_Count := CFG.Block_Count + 1;
      CFG.Next_Block_Id := CFG.Next_Block_Id + 1;
   end Create_Entry;

   -------------------------------------------------------------------------
   --  Create_Exit: Create the exit block
   -------------------------------------------------------------------------
   procedure Create_Exit (CFG : in out Control_Flow_Graph; Id : out Block_Id) is
   begin
      Id := CFG.Next_Block_Id;
      CFG.Blocks (Id) := (
         Id              => Id,
         Block_Kind      => Exit_Block,
         Successors      => (others => No_Block),
         Successor_Count => 0,
         Predecessors    => (others => No_Block),
         Predecessor_Count => 0,
         Dominators      => (others => False),
         Immediate_Dom   => No_Block,
         Loop_Depth      => 0,
         Is_Loop_Header  => False,
         Statement_Count => 0
      );
      CFG.Exit_Block := Id;
      CFG.Block_Count := CFG.Block_Count + 1;
      CFG.Next_Block_Id := CFG.Next_Block_Id + 1;
   end Create_Exit;

   -------------------------------------------------------------------------
   --  Create_Block: Create a new basic block
   -------------------------------------------------------------------------
   procedure Create_Block (
      CFG        : in out Control_Flow_Graph;
      Kind       : Block_Type := Normal_Block;
      Id         : out Block_Id)
   is
   begin
      Id := CFG.Next_Block_Id;
      CFG.Blocks (Id) := (
         Id              => Id,
         Block_Kind      => Kind,
         Successors      => (others => No_Block),
         Successor_Count => 0,
         Predecessors    => (others => No_Block),
         Predecessor_Count => 0,
         Dominators      => (others => False),
         Immediate_Dom   => No_Block,
         Loop_Depth      => 0,
         Is_Loop_Header  => (Kind = Loop_Header_Block),
         Statement_Count => 0
      );
      CFG.Block_Count := CFG.Block_Count + 1;
      CFG.Next_Block_Id := CFG.Next_Block_Id + 1;
   end Create_Block;

   -------------------------------------------------------------------------
   --  Add_Edge: Add an edge between two blocks
   -------------------------------------------------------------------------
   procedure Add_Edge (
      CFG     : in out Control_Flow_Graph;
      From_Id : Block_Id;
      To_Id   : Block_Id)
   is
   begin
      Add_Successor (CFG.Blocks (From_Id), To_Id);
      Add_Predecessor (CFG.Blocks (To_Id), From_Id);
   end Add_Edge;

   -------------------------------------------------------------------------
   --  Compute_Dominators: Iterative dominator computation
   -------------------------------------------------------------------------
   procedure Compute_Dominators (CFG : in out Control_Flow_Graph) is
      Changed : Boolean;
      Iteration : Natural := 0;
      Max_Iterations : constant Natural := Natural (CFG.Block_Count) * Natural (CFG.Block_Count);
   begin
      --  Initialize: entry dominates only itself, others dominated by all
      for Id in 1 .. CFG.Next_Block_Id - 1 loop
         if CFG.Blocks (Id).Id /= No_Block then
            Clear_Dominators (CFG.Blocks (Id));
            if Id = CFG.Entry_Block then
               --  Entry dominates only itself
               Set_Dominator (CFG.Blocks (Id), Id, True);
            else
               --  All others initially dominated by all blocks
               for D in 1 .. CFG.Next_Block_Id - 1 loop
                  if CFG.Blocks (D).Id /= No_Block then
                     Set_Dominator (CFG.Blocks (Id), D, True);
                  end if;
               end loop;
            end if;
         end if;
      end loop;

      --  Iterate until fixed point
      Changed := True;
      while Changed and Iteration < Max_Iterations loop
         Changed := False;
         Iteration := Iteration + 1;

         for Id in 1 .. CFG.Next_Block_Id - 1 loop
            if CFG.Blocks (Id).Id /= No_Block and then Id /= CFG.Entry_Block then
               declare
                  B : Basic_Block renames CFG.Blocks (Id);
                  New_Doms : Dominator_Bits := (others => True);
                  Has_Pred : Boolean := False;
               begin
                  --  Compute intersection of predecessor dominators
                  for P in 1 .. B.Predecessor_Count loop
                     declare
                        Pred_Id : constant Block_Id := B.Predecessors (P);
                     begin
                        if Pred_Id /= No_Block and then CFG.Blocks (Pred_Id).Id /= No_Block then
                           Has_Pred := True;
                           for D in New_Doms'Range loop
                              New_Doms (D) := New_Doms (D) and CFG.Blocks (Pred_Id).Dominators (D);
                           end loop;
                        end if;
                     end;
                  end loop;

                  --  Add self to dominator set
                  New_Doms (Id) := True;

                  --  Check if changed
                  if Has_Pred and then New_Doms /= B.Dominators then
                     CFG.Blocks (Id).Dominators := New_Doms;
                     Changed := True;
                  end if;
               end;
            end if;
         end loop;
      end loop;

      --  Compute immediate dominators
      for Id in 1 .. CFG.Next_Block_Id - 1 loop
         if CFG.Blocks (Id).Id /= No_Block and then Id /= CFG.Entry_Block then
            declare
               B : Basic_Block renames CFG.Blocks (Id);
            begin
               CFG.Blocks (Id).Immediate_Dom := No_Block;

               --  Find the unique dominator that doesn't dominate any other strict dominator
               for Candidate in 1 .. CFG.Next_Block_Id - 1 loop
                  if B.Dominators (Candidate) and Candidate /= Id then
                     declare
                        Is_Idom : Boolean := True;
                     begin
                        --  Check if candidate dominates all other strict dominators
                        for Other in 1 .. CFG.Next_Block_Id - 1 loop
                           if B.Dominators (Other) and Other /= Id and Other /= Candidate then
                              if CFG.Blocks (Other).Dominators (Candidate) then
                                 Is_Idom := False;
                                 exit;
                              end if;
                           end if;
                        end loop;

                        if Is_Idom then
                           CFG.Blocks (Id).Immediate_Dom := Candidate;
                           exit;
                        end if;
                     end;
                  end if;
               end loop;
            end;
         end if;
      end loop;
   end Compute_Dominators;

   -------------------------------------------------------------------------
   --  Dominates: Check if A dominates B
   -------------------------------------------------------------------------
   function Dominates (
      CFG : Control_Flow_Graph;
      A   : Block_Id;
      B   : Block_Id) return Boolean
   is
   begin
      return Is_Dominated_By (CFG.Blocks (B), A);
   end Dominates;

   -------------------------------------------------------------------------
   --  Detect_Loops: Find natural loops in the CFG
   -------------------------------------------------------------------------
   procedure Detect_Loops (CFG : in out Control_Flow_Graph) is
   begin
      --  Ensure dominators are computed
      Compute_Dominators (CFG);

      CFG.Loop_Count := 0;

      --  Find back edges: edge (n -> d) where d dominates n
      for Id in 1 .. CFG.Next_Block_Id - 1 loop
         if CFG.Blocks (Id).Id /= No_Block then
            for S in 1 .. CFG.Blocks (Id).Successor_Count loop
               declare
                  Succ_Id : constant Block_Id := CFG.Blocks (Id).Successors (S);
               begin
                  --  Check if successor dominates current block (back edge)
                  if Succ_Id /= No_Block and then Dominates (CFG, Succ_Id, Id) then
                     --  Found a back edge from Id to Succ_Id
                     if CFG.Loop_Count < Max_Loops then
                        CFG.Loop_Count := CFG.Loop_Count + 1;
                        CFG.Loops (CFG.Loop_Count) := (
                           Header_Id        => Succ_Id,
                           Back_Edge_Source => Id,
                           Loop_Type        => While_Type,
                           Depth            => 0,
                           Is_Natural       => True,
                           Has_Break        => False,
                           Has_Continue     => False
                        );

                        --  Mark header
                        CFG.Blocks (Succ_Id).Is_Loop_Header := True;
                        CFG.Blocks (Succ_Id).Block_Kind := Loop_Header_Block;
                     end if;
                  end if;
               end;
            end loop;
         end if;
      end loop;
   end Detect_Loops;

   -------------------------------------------------------------------------
   --  Get_Loop: Get loop info by index
   -------------------------------------------------------------------------
   function Get_Loop (CFG : Control_Flow_Graph; Index : Positive) return Loop_Info is
   begin
      return CFG.Loops (Index);
   end Get_Loop;

   -------------------------------------------------------------------------
   --  Detect_Branches: Find conditional branches in the CFG
   -------------------------------------------------------------------------
   procedure Detect_Branches (CFG : in out Control_Flow_Graph) is
   begin
      CFG.Branch_Count := 0;

      for Id in 1 .. CFG.Next_Block_Id - 1 loop
         if CFG.Blocks (Id).Id /= No_Block then
            declare
               B : Basic_Block renames CFG.Blocks (Id);
            begin
               if B.Block_Kind = Conditional_Block or else B.Successor_Count > 1 then
                  if CFG.Branch_Count < Max_Branches then
                     CFG.Branch_Count := CFG.Branch_Count + 1;
                     CFG.Branches (CFG.Branch_Count) := (
                        Condition_Block => Id,
                        True_Target     => (if B.Successor_Count >= 1 then B.Successors (1) else No_Block),
                        False_Target    => (if B.Successor_Count >= 2 then B.Successors (2) else No_Block),
                        Merge_Point     => No_Block,  --  Computed separately
                        Is_Multiway     => B.Successor_Count > 2
                     );
                  end if;
               end if;
            end;
         end if;
      end loop;
   end Detect_Branches;

   -------------------------------------------------------------------------
   --  Get_Block: Get block by ID
   -------------------------------------------------------------------------
   function Get_Block (CFG : Control_Flow_Graph; Id : Block_Id) return Basic_Block is
   begin
      return CFG.Blocks (Id);
   end Get_Block;

   -------------------------------------------------------------------------
   --  Compute_Loop_Depths: Calculate nesting depth for all blocks
   -------------------------------------------------------------------------
   procedure Compute_Loop_Depths (CFG : in out Control_Flow_Graph) is
   begin
      --  Reset all depths
      for Id in 1 .. CFG.Next_Block_Id - 1 loop
         if CFG.Blocks (Id).Id /= No_Block then
            CFG.Blocks (Id).Loop_Depth := 0;
         end if;
      end loop;

      --  For each loop, increment depth of all blocks in it
      --  Note: This is a simplified version; full implementation would
      --  need to track which blocks are in each loop body
      for L in 1 .. CFG.Loop_Count loop
         declare
            Header_Id : constant Block_Id := CFG.Loops (L).Header_Id;
         begin
            if Header_Id /= No_Block then
               CFG.Blocks (Header_Id).Loop_Depth :=
                 CFG.Blocks (Header_Id).Loop_Depth + 1;
            end if;
         end;
      end loop;
   end Compute_Loop_Depths;

end IR_Control_Flow;
