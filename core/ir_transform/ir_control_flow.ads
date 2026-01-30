-------------------------------------------------------------------------------
--  STUNIR IR Control Flow - Ada SPARK Specification
--  Part of Phase 1 SPARK Migration
--
--  This package provides control flow graph construction and analysis.
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

with IR_Basic_Blocks; use IR_Basic_Blocks;

package IR_Control_Flow is

   --  Control Flow Type enumeration
   type Control_Flow_Type is (
      If_Type,
      If_Else_Type,
      If_Elif_Else_Type,
      While_Type,
      For_Type,
      Do_While_Type,
      Loop_Type,
      Switch_Type,
      Match_Type,
      Try_Catch_Type,
      Try_Catch_Finally_Type,
      Goto_Type,
      Break_Type,
      Continue_Type,
      Return_Type,
      Recursion_Type
   );

   --  Maximum number of loops/branches
   Max_Loops    : constant := 1_000;
   Max_Branches : constant := 5_000;

   --  Loop Information Record
   type Loop_Info is record
      Header_Id       : Block_Id := No_Block;
      Back_Edge_Source: Block_Id := No_Block;
      Loop_Type       : Control_Flow_Type := While_Type;
      Depth           : Natural := 0;
      Is_Natural      : Boolean := True;
      Has_Break       : Boolean := False;
      Has_Continue    : Boolean := False;
   end record;

   --  Branch Information Record
   type Branch_Info is record
      Condition_Block : Block_Id := No_Block;
      True_Target     : Block_Id := No_Block;
      False_Target    : Block_Id := No_Block;
      Merge_Point     : Block_Id := No_Block;
      Is_Multiway     : Boolean := False;
   end record;

   --  Arrays for loops and branches
   type Loop_Array is array (Positive range <>) of Loop_Info;
   subtype Loop_Vector is Loop_Array (1 .. Max_Loops);

   type Branch_Array is array (Positive range <>) of Branch_Info;
   subtype Branch_Vector is Branch_Array (1 .. Max_Branches);

   --  Block array for the CFG
   type Block_Array is array (Block_Id range 1 .. Max_Blocks - 1) of Basic_Block;

   --  Control Flow Graph Record
   type Control_Flow_Graph is record
      Blocks       : Block_Array;
      Block_Count  : Natural := 0;
      Entry_Block  : Block_Id := No_Block;
      Exit_Block   : Block_Id := No_Block;
      Loops        : Loop_Vector := (others => (others => <>));
      Loop_Count   : Natural := 0;
      Branches     : Branch_Vector := (others => (others => <>));
      Branch_Count : Natural := 0;
      Next_Block_Id: Block_Id := 1;
   end record;

   --  CFG predicates
   function Is_Valid (CFG : Control_Flow_Graph) return Boolean is
     (CFG.Block_Count > 0 and CFG.Entry_Block /= No_Block);

   function Block_Exists (CFG : Control_Flow_Graph; Id : Block_Id) return Boolean is
     (Id /= No_Block and then Id < CFG.Next_Block_Id and then
      CFG.Blocks (Id).Id = Id);

   function Has_Edge (CFG : Control_Flow_Graph; From_Id, To_Id : Block_Id) return Boolean
     with
       Pre => Block_Exists (CFG, From_Id) and Block_Exists (CFG, To_Id);

   --  CFG construction operations
   procedure Create_Entry (CFG : in Out Control_Flow_Graph; Id : out Block_Id)
     with
       Pre  => CFG.Block_Count < Max_Blocks - 1,
       Post => Id /= No_Block and CFG.Entry_Block = Id;

   procedure Create_Exit (CFG : in Out Control_Flow_Graph; Id : out Block_Id)
     with
       Pre  => CFG.Block_Count < Max_Blocks - 1,
       Post => Id /= No_Block and CFG.Exit_Block = Id;

   procedure Create_Block (
      CFG        : in Out Control_Flow_Graph;
      Kind       : Block_Type := Normal_Block;
      Id         : out Block_Id)
     with
       Pre  => CFG.Block_Count < Max_Blocks - 1,
       Post => Id /= No_Block and Block_Exists (CFG, Id);

   procedure Add_Edge (
      CFG     : in Out Control_Flow_Graph;
      From_Id : Block_Id;
      To_Id   : Block_Id)
     with
       Pre  => From_Id /= No_Block and To_Id /= No_Block and
               Block_Exists (CFG, From_Id) and Block_Exists (CFG, To_Id),
       Post => Has_Edge (CFG, From_Id, To_Id);

   --  Dominator analysis
   procedure Compute_Dominators (CFG : in Out Control_Flow_Graph)
     with
       Pre => Is_Valid (CFG);

   function Dominates (
      CFG : Control_Flow_Graph;
      A   : Block_Id;
      B   : Block_Id) return Boolean
     with
       Pre => Block_Exists (CFG, A) and Block_Exists (CFG, B);

   --  Loop detection
   procedure Detect_Loops (CFG : in Out Control_Flow_Graph)
     with
       Pre => Is_Valid (CFG);

   function Get_Loop_Count (CFG : Control_Flow_Graph) return Natural is
     (CFG.Loop_Count);

   function Get_Loop (CFG : Control_Flow_Graph; Index : Positive) return Loop_Info
     with
       Pre => Index <= CFG.Loop_Count;

   --  Branch detection
   procedure Detect_Branches (CFG : in Out Control_Flow_Graph)
     with
       Pre => Is_Valid (CFG);

   function Get_Branch_Count (CFG : Control_Flow_Graph) return Natural is
     (CFG.Branch_Count);

   --  Utility functions
   function Get_Block (CFG : Control_Flow_Graph; Id : Block_Id) return Basic_Block
     with
       Pre => Block_Exists (CFG, Id);

   function Get_Block_Count (CFG : Control_Flow_Graph) return Natural is
     (CFG.Block_Count);

   --  Loop depth calculation
   procedure Compute_Loop_Depths (CFG : in Out Control_Flow_Graph)
     with
       Pre => Is_Valid (CFG);

end IR_Control_Flow;
