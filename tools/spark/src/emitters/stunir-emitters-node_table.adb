-- STUNIR Emitter Node Table (Semantic IR AST storage)
-- DO-178C Level A
-- Phase 3a: Core Category Emitters

pragma SPARK_Mode (On);

package body STUNIR.Emitters.Node_Table is

   procedure Initialize (Table : out Node_Table) is
   begin
      Table.Count := 0;
      for I in Table.Entries'Range loop
         Table.Entries (I).Used := False;
      end loop;
   end Initialize;

   function Lookup (Table : Node_Table; ID : Node_ID) return Node_Index is
   begin
      for I in 1 .. Table.Count loop
         if Table.Entries (I).Used and then Name_Strings.To_String (Table.Entries (I).ID) = Name_Strings.To_String (ID) then
            return I;
         end if;
      end loop;
      return 0;
   end Lookup;

   function Get_Declaration (Table : Node_Table; Index : Node_Index) return Declaration_Record is
   begin
      return Table.Entries (Index).Data.Decl;
   end Get_Declaration;

   function Get_Statement (Table : Node_Table; Index : Node_Index) return Statement_Record is
   begin
      return Table.Entries (Index).Data.Stmt;
   end Get_Statement;

   function Get_Expression (Table : Node_Table; Index : Node_Index) return Expression_Record is
   begin
      return Table.Entries (Index).Data.Expr;
   end Get_Expression;

   procedure Add_Node
     (Table : in out Node_Table;
      Node  : IR_Node;
      Success : out Boolean)
   is
   begin
      if Table.Count < Max_Nodes then
         Table.Count := Table.Count + 1;
         Table.Entries (Table.Count) := (
            ID    => Node.ID,
            Kind  => Node.Kind,
            Group => Group_Node,
            Data  => (Group => Group_Node, Node => Node),
            Used  => True);
         Success := True;
      else
         Success := False;
      end if;
   end Add_Node;

   procedure Add_Declaration
     (Table   : in out Node_Table;
      Decl    : Declaration_Record;
      Success : out Boolean)
   is
   begin
      if Table.Count < Max_Nodes then
         Table.Count := Table.Count + 1;
         Table.Entries (Table.Count) := (
            ID    => Decl.Base.Base.ID,
            Kind  => Decl.Base.Base.Kind,
            Group => Group_Declaration,
            Data  => (Group => Group_Declaration, Decl => Decl),
            Used  => True);
         Success := True;
      else
         Success := False;
      end if;
   end Add_Declaration;

   procedure Add_Function_Declaration
     (Table   : in out Node_Table;
      Decl    : Function_Declaration;
      Success : out Boolean)
   is
      Wrapped : constant Declaration_Record := (Kind => Kind_Function_Decl, Func => Decl);
   begin
      Add_Declaration (Table, Wrapped, Success);
   end Add_Function_Declaration;

   procedure Add_Type_Declaration
     (Table   : in out Node_Table;
      Decl    : Type_Declaration;
      Success : out Boolean)
   is
      Wrapped : constant Declaration_Record := (Kind => Kind_Type_Decl, Typ => Decl);
   begin
      Add_Declaration (Table, Wrapped, Success);
   end Add_Type_Declaration;

   procedure Add_Const_Declaration
     (Table   : in out Node_Table;
      Decl    : Const_Declaration;
      Success : out Boolean)
   is
      Wrapped : constant Declaration_Record := (Kind => Kind_Const_Decl, Cst => Decl);
   begin
      Add_Declaration (Table, Wrapped, Success);
   end Add_Const_Declaration;

   procedure Add_Variable_Declaration
     (Table   : in out Node_Table;
      Decl    : Variable_Declaration;
      Success : out Boolean)
   is
      Wrapped : constant Declaration_Record := (Kind => Kind_Var_Decl, Var => Decl);
   begin
      Add_Declaration (Table, Wrapped, Success);
   end Add_Variable_Declaration;

   procedure Add_Statement
     (Table   : in out Node_Table;
      Stmt    : Statement_Record;
      Success : out Boolean)
   is
   begin
      if Table.Count < Max_Nodes then
         Table.Count := Table.Count + 1;
         Table.Entries (Table.Count) := (
            ID    => Stmt.Base.Base.ID,
            Kind  => Stmt.Base.Base.Kind,
            Group => Group_Statement,
            Data  => (Group => Group_Statement, Stmt => Stmt),
            Used  => True);
         Success := True;
      else
         Success := False;
      end if;
   end Add_Statement;

   procedure Add_Expression
     (Table   : in out Node_Table;
      Expr    : Expression_Record;
      Success : out Boolean)
   is
   begin
      if Table.Count < Max_Nodes then
         Table.Count := Table.Count + 1;
         Table.Entries (Table.Count) := (
            ID    => Expr.Base.Base.ID,
            Kind  => Expr.Base.Base.Kind,
            Group => Group_Expression,
            Data  => (Group => Group_Expression, Expr => Expr),
            Used  => True);
         Success := True;
      else
         Success := False;
      end if;
   end Add_Expression;

   procedure Add_Expression_Node
     (Table   : in out Node_Table;
      Expr    : Expression_Node;
      Success : out Boolean)
   is
      Wrapped : Expression_Record := (Kind => Kind_Integer_Literal,
                       Base => (Kind => Kind_Integer_Literal,
                             Base => (Kind     => Kind_Integer_Literal,
                                ID       => Expr.Base.ID,
                                Location => Expr.Base.Location,
                                Hash     => Expr.Base.Hash,
                                Int_Value => 0,
                                Int_Radix => 10),
                             Expr_Type => Expr.Expr_Type));
   begin
      case Expr.Kind is
         when Kind_Binary_Expr =>
            Wrapped := (Kind => Kind_Binary_Expr, Bin => (Base => Expr, Operator => Op_Add, Left_ID => Name_Strings.Null_Bounded_String, Right_ID => Name_Strings.Null_Bounded_String));
         when Kind_Unary_Expr =>
            Wrapped := (Kind => Kind_Unary_Expr, Un => (Base => Expr, Operator => Op_Not, Operand_ID => Name_Strings.Null_Bounded_String));
         when Kind_Function_Call =>
            Wrapped := (Kind => Kind_Function_Call, Call => (Base => Expr, Function_Name => Name_Strings.Null_Bounded_String, Func_Binding => Name_Strings.Null_Bounded_String, Arg_Count => 0, Arguments => [others => Name_Strings.Null_Bounded_String]));
         when Kind_Member_Expr =>
            Wrapped := (Kind => Kind_Member_Expr, Mem => (Base => Expr, Object_ID => Name_Strings.Null_Bounded_String, Member_Name => Name_Strings.Null_Bounded_String, Is_Arrow => False));
         when Kind_Array_Access =>
            Wrapped := (Kind => Kind_Array_Access, Arr => (Base => Expr, Array_ID => Name_Strings.Null_Bounded_String, Index_ID => Name_Strings.Null_Bounded_String));
         when Kind_Cast_Expr =>
            Wrapped := (Kind => Kind_Cast_Expr, Cast => (Base => Expr, Operand_ID => Name_Strings.Null_Bounded_String, Target_Type => (Kind => TK_Primitive, Prim_Type => Type_Void)));
         when Kind_Ternary_Expr =>
            Wrapped := (Kind => Kind_Ternary_Expr, Ter => (Base => Expr, Condition_ID => Name_Strings.Null_Bounded_String, Then_ID => Name_Strings.Null_Bounded_String, Else_ID => Name_Strings.Null_Bounded_String));
         when Kind_Array_Init =>
            Wrapped := (Kind => Kind_Array_Init, Arr_Init => (Base => Expr, Elem_Count => 0, Elements => [others => Name_Strings.Null_Bounded_String]));
         when Kind_Struct_Init =>
            Wrapped := (Kind => Kind_Struct_Init, Struct_Init => (Base => Expr, Struct_Name => Name_Strings.Null_Bounded_String, Field_Count => 0, Fields => [others => (Field_Name => Name_Strings.Null_Bounded_String, Value_ID => Name_Strings.Null_Bounded_String)]));
         when others =>
            Wrapped := (Kind => Kind_Var_Ref,
                        Base => (Kind => Kind_Var_Ref,
                                 Base => (Kind     => Kind_Var_Ref,
                                          ID       => Expr.Base.ID,
                                          Location => Expr.Base.Location,
                                          Hash     => Expr.Base.Hash,
                                          Var_Name => Name_Strings.Null_Bounded_String,
                                          Var_Binding => Name_Strings.Null_Bounded_String),
                                 Expr_Type => Expr.Expr_Type));
      end case;
      Add_Expression (Table, Wrapped, Success);
   end Add_Expression_Node;

end STUNIR.Emitters.Node_Table;
