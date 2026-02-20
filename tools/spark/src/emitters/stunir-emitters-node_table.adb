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
         if Table.Entries (I).Used and then Table.Entries (I).ID = ID then
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
            ID    => Node.Node_ID,
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
            ID    => Decl.Base.Node_ID,
            Kind  => Decl.Base.Kind,
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
      Wrapped : Declaration_Record := (Kind => Kind_Function_Decl, Func => Decl);
   begin
      Add_Declaration (Table, Wrapped, Success);
   end Add_Function_Declaration;

   procedure Add_Type_Declaration
     (Table   : in out Node_Table;
      Decl    : Type_Declaration;
      Success : out Boolean)
   is
      Wrapped : Declaration_Record := (Kind => Kind_Type_Decl, Typ => Decl);
   begin
      Add_Declaration (Table, Wrapped, Success);
   end Add_Type_Declaration;

   procedure Add_Const_Declaration
     (Table   : in out Node_Table;
      Decl    : Const_Declaration;
      Success : out Boolean)
   is
      Wrapped : Declaration_Record := (Kind => Kind_Const_Decl, Cst => Decl);
   begin
      Add_Declaration (Table, Wrapped, Success);
   end Add_Const_Declaration;

   procedure Add_Variable_Declaration
     (Table   : in out Node_Table;
      Decl    : Variable_Declaration;
      Success : out Boolean)
   is
      Wrapped : Declaration_Record := (Kind => Kind_Var_Decl, Var => Decl);
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
            ID    => Stmt.Base.Node_ID,
            Kind  => Stmt.Base.Kind,
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
            ID    => Expr.Base.Node_ID,
            Kind  => Expr.Base.Kind,
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
      Wrapped : Expression_Record := (Kind => Expr.Kind, Base => Expr);
   begin
      Add_Expression (Table, Wrapped, Success);
   end Add_Expression_Node;

end STUNIR.Emitters.Node_Table;
