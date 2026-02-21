--  STUNIR DO-333 Formal Specification Framework
--  Implementation
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

package body Formal_Spec is

   --  ============================================================
   --  Add Precondition
   --  ============================================================

   procedure Add_Precondition
     (C       : in Out Contract_Spec;
      Expr    : String;
      Line    : Natural;
      Col     : Natural;
      Success : out Boolean)
   is
      New_Expr : Formal_Expression;
   begin
      if C.Pre_Count >= Max_Conditions then
         Success := False;
         return;
      end if;

      Make_Expression (Expr, Precondition, Line, Col, New_Expr);
      C.Pre_Count := C.Pre_Count + 1;
      C.Preconditions (C.Pre_Count) := New_Expr;
      Success := True;
   end Add_Precondition;

   --  ============================================================
   --  Add Postcondition
   --  ============================================================

   procedure Add_Postcondition
     (C       : in Out Contract_Spec;
      Expr    : String;
      Line    : Natural;
      Col     : Natural;
      Success : out Boolean)
   is
      New_Expr : Formal_Expression;
   begin
      if C.Post_Count >= Max_Conditions then
         Success := False;
         return;
      end if;

      Make_Expression (Expr, Postcondition, Line, Col, New_Expr);
      C.Post_Count := C.Post_Count + 1;
      C.Postconditions (C.Post_Count) := New_Expr;
      Success := True;
   end Add_Postcondition;

   --  ============================================================
   --  Add Invariant
   --  ============================================================

   procedure Add_Invariant
     (C       : in Out Contract_Spec;
      Expr    : String;
      Kind    : Spec_Kind;
      Line    : Natural;
      Col     : Natural;
      Success : out Boolean)
   is
      New_Expr : Formal_Expression;
   begin
      if C.Inv_Count >= Max_Conditions then
         Success := False;
         return;
      end if;

      Make_Expression (Expr, Kind, Line, Col, New_Expr);
      C.Inv_Count := C.Inv_Count + 1;
      C.Invariants (C.Inv_Count) := New_Expr;
      Success := True;
   end Add_Invariant;

   --  ============================================================
   --  Make Expression
   --  ============================================================

   procedure Make_Expression
     (Expr   : String;
      Kind   : Spec_Kind;
      Line   : Natural;
      Col    : Natural;
      Result : out Formal_Expression)
   is
      Content : Expr_String := (others => ' ');
   begin
      --  Copy expression content
      for I in Expr'Range loop
         Content (I - Expr'First + 1) := Expr (I);
      end loop;

      Result := (
         Kind      => Kind,
         Content   => Content,
         Length    => Expr'Length,
         Line_Num  => Line,
         Column    => Col,
         Verified  => False,
         Traceable => True
      );
   end Make_Expression;

   --  ============================================================
   --  Mark Verified
   --  ============================================================

   procedure Mark_Verified
     (E : in Out Formal_Expression)
   is
   begin
      E.Verified := True;
   end Mark_Verified;

end Formal_Spec;
