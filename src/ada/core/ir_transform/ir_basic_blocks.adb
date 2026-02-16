-------------------------------------------------------------------------------
--  STUNIR IR Basic Blocks - Ada SPARK Body
--  Part of Phase 1 SPARK Migration
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

package body IR_Basic_Blocks is

   -------------------------------------------------------------------------
   --  Add_Successor: Add a successor block
   -------------------------------------------------------------------------
   procedure Add_Successor (
      B       : in out Basic_Block;
      Succ_Id : Block_Id)
   is
   begin
      --  Check if already present
      for I in 1 .. B.Successor_Count loop
         if B.Successors (I) = Succ_Id then
            return;  --  Already exists
         end if;
      end loop;

      B.Successor_Count := B.Successor_Count + 1;
      B.Successors (B.Successor_Count) := Succ_Id;
   end Add_Successor;

   -------------------------------------------------------------------------
   --  Add_Predecessor: Add a predecessor block
   -------------------------------------------------------------------------
   procedure Add_Predecessor (
      B       : in out Basic_Block;
      Pred_Id : Block_Id)
   is
   begin
      --  Check if already present
      for I in 1 .. B.Predecessor_Count loop
         if B.Predecessors (I) = Pred_Id then
            return;  --  Already exists
         end if;
      end loop;

      B.Predecessor_Count := B.Predecessor_Count + 1;
      B.Predecessors (B.Predecessor_Count) := Pred_Id;
   end Add_Predecessor;

   -------------------------------------------------------------------------
   --  Has_Successor: Check if a block is a successor
   -------------------------------------------------------------------------
   function Has_Successor (
      B       : Basic_Block;
      Succ_Id : Block_Id) return Boolean
   is
   begin
      for I in 1 .. B.Successor_Count loop
         if B.Successors (I) = Succ_Id then
            return True;
         end if;
      end loop;
      return False;
   end Has_Successor;

   -------------------------------------------------------------------------
   --  Has_Predecessor: Check if a block is a predecessor
   -------------------------------------------------------------------------
   function Has_Predecessor (
      B       : Basic_Block;
      Pred_Id : Block_Id) return Boolean
   is
   begin
      for I in 1 .. B.Predecessor_Count loop
         if B.Predecessors (I) = Pred_Id then
            return True;
         end if;
      end loop;
      return False;
   end Has_Predecessor;

   -------------------------------------------------------------------------
   --  Set_Dominator: Set or clear a dominator bit
   -------------------------------------------------------------------------
   procedure Set_Dominator (
      B       : in out Basic_Block;
      Dom_Id  : Block_Id;
      Value   : Boolean := True)
   is
   begin
      B.Dominators (Dom_Id) := Value;
   end Set_Dominator;

   -------------------------------------------------------------------------
   --  Is_Dominated_By: Check if block is dominated by another
   -------------------------------------------------------------------------
   function Is_Dominated_By (
      B      : Basic_Block;
      Dom_Id : Block_Id) return Boolean
   is
   begin
      if Dom_Id = No_Block then
         return False;
      end if;
      return B.Dominators (Dom_Id);
   end Is_Dominated_By;

   -------------------------------------------------------------------------
   --  Clear_Dominators: Clear all dominator bits
   -------------------------------------------------------------------------
   procedure Clear_Dominators (B : in out Basic_Block) is
   begin
      B.Dominators := (others => False);
   end Clear_Dominators;

   -------------------------------------------------------------------------
   --  Copy_Dominators: Copy dominator set from source to target
   -------------------------------------------------------------------------
   procedure Copy_Dominators (
      Source : Basic_Block;
      Target : in out Basic_Block)
   is
   begin
      Target.Dominators := Source.Dominators;
   end Copy_Dominators;

   -------------------------------------------------------------------------
   --  Intersect_Dominators: Compute intersection of dominator sets
   -------------------------------------------------------------------------
   procedure Intersect_Dominators (
      B     : in out Basic_Block;
      Other : Basic_Block)
   is
   begin
      for I in B.Dominators'Range loop
         B.Dominators (I) := B.Dominators (I) and Other.Dominators (I);
      end loop;
   end Intersect_Dominators;

end IR_Basic_Blocks;
