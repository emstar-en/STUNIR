-------------------------------------------------------------------------------
--  STUNIR IR Basic Blocks - Ada SPARK Specification
--  Part of Phase 1 SPARK Migration
--
--  This package provides basic block representation for control flow analysis.
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

package IR_Basic_Blocks is

   --  Maximum bounds for bounded containers
   Max_Blocks      : constant := 10_000;
   Max_Successors  : constant := 64;
   Max_Predecessors: constant := 64;
   Max_Statements  : constant := 1_000;
   Max_Labels      : constant := 16;

   --  Block identifier type
   type Block_Id is range 0 .. Max_Blocks - 1;
   No_Block : constant Block_Id := 0;

   --  Block types enumeration
   type Block_Type is (
      Entry_Block,
      Exit_Block,
      Normal_Block,
      Conditional_Block,
      Loop_Header_Block,
      Loop_Body_Block,
      Loop_Exit_Block,
      Switch_Block,
      Case_Block,
      Try_Block,
      Catch_Block,
      Finally_Block
   );

   --  Successor/Predecessor counts
   subtype Successor_Count is Natural range 0 .. Max_Successors;
   subtype Predecessor_Count is Natural range 0 .. Max_Predecessors;

   --  Arrays for block connections
   type Block_Id_Array is array (Positive range <>) of Block_Id;
   subtype Successor_Array is Block_Id_Array (1 .. Max_Successors);
   subtype Predecessor_Array is Block_Id_Array (1 .. Max_Predecessors);

   --  Dominator set represented as a bit vector
   type Dominator_Bits is array (Block_Id range 1 .. Max_Blocks - 1) of Boolean;

   --  Basic Block Record
   type Basic_Block is record
      Id              : Block_Id := No_Block;
      Block_Kind      : Block_Type := Normal_Block;
      Successors      : Successor_Array := (others => No_Block);
      Successor_Count : Natural := 0;
      Predecessors    : Predecessor_Array := (others => No_Block);
      Predecessor_Count : Natural := 0;
      Dominators      : Dominator_Bits := (others => False);
      Immediate_Dom   : Block_Id := No_Block;
      Loop_Depth      : Natural := 0;
      Is_Loop_Header  : Boolean := False;
      Statement_Count : Natural := 0;
   end record;

   --  Default empty block
   Empty_Block : constant Basic_Block := (
      Id              => No_Block,
      Block_Kind      => Normal_Block,
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

   --  Block predicates
   function Is_Valid (B : Basic_Block) return Boolean is
     (B.Id /= No_Block);

   function Is_Entry (B : Basic_Block) return Boolean is
     (B.Block_Kind = Entry_Block);

   function Is_Exit (B : Basic_Block) return Boolean is
     (B.Block_Kind = Exit_Block);

   function Is_Conditional (B : Basic_Block) return Boolean is
     (B.Block_Kind = Conditional_Block);

   function Is_Loop_Related (B : Basic_Block) return Boolean is
     (B.Block_Kind in Loop_Header_Block | Loop_Body_Block | Loop_Exit_Block);

   --  Block modification operations
   procedure Add_Successor (
      B       : in out Basic_Block;
      Succ_Id : Block_Id)
     with
       Pre  => B.Successor_Count < Max_Successors and Succ_Id /= No_Block,
       Post => B.Successor_Count = B.Successor_Count'Old + 1;

   procedure Add_Predecessor (
      B       : in out Basic_Block;
      Pred_Id : Block_Id)
     with
       Pre  => B.Predecessor_Count < Max_Predecessors and Pred_Id /= No_Block,
       Post => B.Predecessor_Count = B.Predecessor_Count'Old + 1;

   --  Block query operations
   function Has_Successor (
      B       : Basic_Block;
      Succ_Id : Block_Id) return Boolean;

   function Has_Predecessor (
      B       : Basic_Block;
      Pred_Id : Block_Id) return Boolean;

   --  Dominator operations
   procedure Set_Dominator (
      B       : in out Basic_Block;
      Dom_Id  : Block_Id;
      Value   : Boolean := True)
     with
       Pre => Dom_Id /= No_Block;

   function Is_Dominated_By (
      B      : Basic_Block;
      Dom_Id : Block_Id) return Boolean;

   procedure Clear_Dominators (B : in out Basic_Block);

   procedure Copy_Dominators (
      Source : Basic_Block;
      Target : in out Basic_Block);

   procedure Intersect_Dominators (
      B     : in out Basic_Block;
      Other : Basic_Block);

end IR_Basic_Blocks;
