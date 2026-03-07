-------------------------------------------------------------------------------
--  STUNIR Semantic IR Statements Package Body
--  DO-178C Level A Compliant
--  SPARK 2014 Mode
--
--  Implementation of validation functions for Semantic IR statements.
--
--  Copyright (c) 2026 STUNIR Project
--  License: MIT
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

package body Semantic_IR.Statements is

   --  Check if a block statement is valid
   function Is_Valid_Block_Stmt (B : Block_Stmt) return Boolean is
   begin
      --  Base node must be valid
      if not Is_Valid_Semantic_Node (B.Base) then
         return False;
      end if;

      --  Statement count must be within bounds
      if B.Stmt_Count > Max_Block_Stmts then
         return False;
      end if;

      --  Validate all statement references
      for I in 1 .. B.Stmt_Count loop
         if not Is_Valid_Node_ID (B.Statements (I)) then
            return False;
         end if;
      end loop;

      return True;
   end Is_Valid_Block_Stmt;

   --  Check if an if statement is valid
   function Is_Valid_If_Stmt (I : If_Stmt) return Boolean is
   begin
      --  Base node must be valid
      if not Is_Valid_Semantic_Node (I.Base) then
         return False;
      end if;

      --  Condition must be valid
      if not Is_Valid_Node_ID (I.Condition) then
         return False;
      end if;

      --  Then branch must be valid
      if not Is_Valid_Node_ID (I.Then_Branch) then
         return False;
      end if;

      --  Else branch is optional (empty Node_ID is allowed)

      return True;
   end Is_Valid_If_Stmt;

   --  Check if a while statement is valid
   function Is_Valid_While_Stmt (W : While_Stmt) return Boolean is
   begin
      --  Base node must be valid
      if not Is_Valid_Semantic_Node (W.Base) then
         return False;
      end if;

      --  Condition must be valid
      if not Is_Valid_Node_ID (W.Condition) then
         return False;
      end if;

      --  Body must be valid
      if not Is_Valid_Node_ID (W.Body_Node) then
         return False;
      end if;

      return True;
   end Is_Valid_While_Stmt;

   --  Check if a for statement is valid
   function Is_Valid_For_Stmt (F : For_Stmt) return Boolean is
   begin
      --  Base node must be valid
      if not Is_Valid_Semantic_Node (F.Base) then
         return False;
      end if;

      --  Loop variable must be valid
      if not Is_Valid_Node_ID (F.Loop_Var) then
         return False;
      end if;

      --  Start expression must be valid
      if not Is_Valid_Node_ID (F.Start_Expr) then
         return False;
      end if;

      --  End expression must be valid
      if not Is_Valid_Node_ID (F.End_Expr) then
         return False;
      end if;

      --  Body must be valid
      if not Is_Valid_Node_ID (F.Body_Node) then
         return False;
      end if;

      --  Step expression is optional (empty Node_ID is allowed)

      return True;
   end Is_Valid_For_Stmt;

end Semantic_IR.Statements;