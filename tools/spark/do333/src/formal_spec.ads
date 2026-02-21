--  STUNIR DO-333 Formal Specification Framework
--  Specification Types and Operations
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0
--
--  This package provides formal specification types supporting:
--  - Pre/postconditions
--  - Invariants (loop, type, data)
--  - Type contracts
--  - Ghost code
--  - Proof functions
--
--  DO-333 Objective: FM.1 (Formal Specification)

pragma SPARK_Mode (On);

package Formal_Spec is

   --  ============================================================
   --  Constants
   --  ============================================================

   Max_Name_Length        : constant := 256;
   Max_Expression_Length  : constant := 4096;
   Max_Conditions         : constant := 64;
   Max_Variables          : constant := 128;
   Max_Functions          : constant := 64;

   --  ============================================================
   --  Specification Kind
   --  ============================================================

   type Spec_Kind is (
      Precondition,
      Postcondition,
      Invariant,
      Type_Invariant,
      Loop_Invariant,
      Assertion,
      Assumption,
      Ghost_Code,
      Contract_Cases,
      Default_Initial_Condition
   );

   --  ============================================================
   --  Bounded Strings
   --  ============================================================

   subtype Name_Index is Positive range 1 .. Max_Name_Length;
   subtype Name_Length is Natural range 0 .. Max_Name_Length;
   subtype Name_String is String (Name_Index);

   subtype Expr_Index is Positive range 1 .. Max_Expression_Length;
   subtype Expr_Length is Natural range 0 .. Max_Expression_Length;
   subtype Expr_String is String (Expr_Index);

   --  ============================================================
   --  Formal Expression
   --  ============================================================

   type Formal_Expression is record
      Kind       : Spec_Kind;
      Content    : Expr_String;
      Length     : Expr_Length;
      Line_Num   : Natural;
      Column     : Natural;
      Verified   : Boolean;
      Traceable  : Boolean;
   end record;

   --  Expression validity predicate
   function Is_Valid_Expression (E : Formal_Expression) return Boolean is
     (E.Length > 0 and then E.Length <= Max_Expression_Length);

   --  Empty expression constant
   Empty_Expression : constant Formal_Expression := (
      Kind      => Assertion,
      Content   => (others => ' '),
      Length    => 0,
      Line_Num  => 0,
      Column    => 0,
      Verified  => False,
      Traceable => False
   );

   --  ============================================================
   --  Condition Array Types
   --  ============================================================

   subtype Condition_Index is Positive range 1 .. Max_Conditions;
   subtype Condition_Count is Natural range 0 .. Max_Conditions;

   type Condition_Array is array (Condition_Index) of Formal_Expression;

   --  ============================================================
   --  Contract Specification
   --  ============================================================

   type Contract_Spec is record
      Preconditions  : Condition_Array;
      Pre_Count      : Condition_Count;
      Postconditions : Condition_Array;
      Post_Count     : Condition_Count;
      Invariants     : Condition_Array;
      Inv_Count      : Condition_Count;
   end record;

   --  Contract validity predicate
   function Is_Valid_Contract (C : Contract_Spec) return Boolean is
     (C.Pre_Count <= Max_Conditions and then
      C.Post_Count <= Max_Conditions and then
      C.Inv_Count <= Max_Conditions);

   --  Empty contract constant
   Empty_Contract : constant Contract_Spec := (
      Preconditions  => (others => Empty_Expression),
      Pre_Count      => 0,
      Postconditions => (others => Empty_Expression),
      Post_Count     => 0,
      Invariants     => (others => Empty_Expression),
      Inv_Count      => 0
   );

   --  ============================================================
   --  Ghost Variable
   --  ============================================================

   type Ghost_Variable is record
      Name      : Name_String;
      Name_Len  : Name_Length;
      Type_Name : Name_String;
      Type_Len  : Name_Length;
      Init_Expr : Expr_String;
      Init_Len  : Expr_Length;
      Is_Global : Boolean;
   end record;

   --  Ghost variable validity predicate
   function Is_Valid_Ghost (G : Ghost_Variable) return Boolean is
     (G.Name_Len > 0 and then G.Type_Len > 0);

   --  Empty ghost constant
   Empty_Ghost : constant Ghost_Variable := (
      Name      => (others => ' '),
      Name_Len  => 0,
      Type_Name => (others => ' '),
      Type_Len  => 0,
      Init_Expr => (others => ' '),
      Init_Len  => 0,
      Is_Global => False
   );

   --  ============================================================
   --  Proof Function
   --  ============================================================

   type Proof_Function is record
      Name        : Name_String;
      Name_Len    : Name_Length;
      Parameters  : Expr_String;
      Param_Len   : Expr_Length;
      Return_Type : Name_String;
      Ret_Type_Len : Name_Length;
      Return_Expr : Expr_String;
      Ret_Len     : Expr_Length;
      Is_Ghost    : Boolean;
      Is_Pure     : Boolean;
   end record;

   --  Proof function validity predicate
   function Is_Valid_Proof_Function (P : Proof_Function) return Boolean is
     (P.Name_Len > 0 and then P.Ret_Type_Len > 0);

   --  Empty proof function constant
   Empty_Proof_Function : constant Proof_Function := (
      Name        => (others => ' '),
      Name_Len    => 0,
      Parameters  => (others => ' '),
      Param_Len   => 0,
      Return_Type => (others => ' '),
      Ret_Type_Len => 0,
      Return_Expr => (others => ' '),
      Ret_Len     => 0,
      Is_Ghost    => False,
      Is_Pure     => False
   );

   --  ============================================================
   --  Type Contract
   --  ============================================================

   type Type_Contract is record
      Type_Name    : Name_String;
      Name_Len     : Name_Length;
      Predicates   : Condition_Array;
      Pred_Count   : Condition_Count;
      Invariants   : Condition_Array;
      Inv_Count    : Condition_Count;
      Has_DIC      : Boolean;  -- Default Initial Condition
   end record;

   --  Type contract validity predicate
   function Is_Valid_Type_Contract (T : Type_Contract) return Boolean is
     (T.Name_Len > 0);

   --  Empty type contract constant
   Empty_Type_Contract : constant Type_Contract := (
      Type_Name  => (others => ' '),
      Name_Len   => 0,
      Predicates => (others => Empty_Expression),
      Pred_Count => 0,
      Invariants => (others => Empty_Expression),
      Inv_Count  => 0,
      Has_DIC    => False
   );

   --  ============================================================
   --  Operations
   --  ============================================================

   --  Add a precondition to a contract
   procedure Add_Precondition
     (C       : in out Contract_Spec;
      Expr    : String;
      Line    : Natural;
      Col     : Natural;
      Success : out Boolean)
   with
      Pre  => Expr'Length > 0 and then Expr'Length <= Max_Expression_Length,
      Post => (if Success then C.Pre_Count = C.Pre_Count'Old + 1);

   --  Add a postcondition to a contract
   procedure Add_Postcondition
     (C       : in Out Contract_Spec;
      Expr    : String;
      Line    : Natural;
      Col     : Natural;
      Success : out Boolean)
   with
      Pre  => Expr'Length > 0 and then Expr'Length <= Max_Expression_Length,
      Post => (if Success then C.Post_Count = C.Post_Count'Old + 1);

   --  Add an invariant to a contract
   procedure Add_Invariant
     (C       : in Out Contract_Spec;
      Expr    : String;
      Kind    : Spec_Kind;
      Line    : Natural;
      Col     : Natural;
      Success : out Boolean)
   with
      Pre  => Expr'Length > 0 and then
              Expr'Length <= Max_Expression_Length and then
              Kind in Invariant | Type_Invariant | Loop_Invariant,
      Post => (if Success then C.Inv_Count = C.Inv_Count'Old + 1);

   --  Get count of all conditions
   function Total_Conditions (C : Contract_Spec) return Natural is
     (C.Pre_Count + C.Post_Count + C.Inv_Count);

   --  Check if contract is empty
   function Is_Empty (C : Contract_Spec) return Boolean is
     (Total_Conditions (C) = 0);

   --  Create expression from string
   procedure Make_Expression
     (Expr   : String;
      Kind   : Spec_Kind;
      Line   : Natural;
      Col    : Natural;
      Result : out Formal_Expression)
   with
      Pre => Expr'Length > 0 and then Expr'Length <= Max_Expression_Length;

   --  Mark expression as verified
   procedure Mark_Verified
     (E : in out Formal_Expression)
   with
      Post => E.Verified;

end Formal_Spec;
