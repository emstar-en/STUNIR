-------------------------------------------------------------------------------
--  STUNIR Semantic IR JSON Parser (Implementation)
--  DO-178C Level A Compliant
--  SPARK 2014 Mode
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

with STUNIR_Types;
with STUNIR_JSON_Parser;
with Semantic_IR.Types;
with Semantic_IR.Nodes;
with Semantic_IR.Modules;
with Semantic_IR.Declarations;
with Semantic_IR.Statements;
with Semantic_IR.Expressions;
with STUNIR.Emitters.Node_Table;

package body Semantic_IR.JSON is

   use STUNIR_JSON_Parser;
   use Semantic_IR.Types;
   use Semantic_IR.Nodes;
   use Semantic_IR.Modules;
   use Semantic_IR.Declarations;
   use Semantic_IR.Statements;
   use Semantic_IR.Expressions;
   use STUNIR.Emitters.Node_Table;
   use type Name_Strings.Bounded_String;

   function To_Name (S : String) return IR_Name is
   begin
      if S'Length = 0 then
         return Name_Strings.Null_Bounded_String;
      elsif S'Length > Max_Name_Length then
         return Name_Strings.To_Bounded_String (S (S'First .. S'First + Max_Name_Length - 1));
      else
         return Name_Strings.To_Bounded_String (S);
      end if;
   end To_Name;

   function To_Node_ID (S : String) return Node_ID is
   begin
      if S'Length = 0 then
         return Name_Strings.Null_Bounded_String;
      elsif S'Length > Max_Name_Length then
         return Name_Strings.To_Bounded_String (S (S'First .. S'First + Max_Name_Length - 1));
      else
         return Name_Strings.To_Bounded_String (S);
      end if;
   end To_Node_ID;

   function To_Path (S : String) return IR_Path is
   begin
      if S'Length = 0 then
         return Semantic_IR.Types.Path_Strings.Null_Bounded_String;
      elsif S'Length > Semantic_IR.Types.Max_Path_Length then
         return Semantic_IR.Types.Path_Strings.To_Bounded_String (S (S'First .. S'First + Semantic_IR.Types.Max_Path_Length - 1));
      else
         return Semantic_IR.Types.Path_Strings.To_Bounded_String (S);
      end if;
   end To_Path;

   function To_Hash (S : String) return IR_Hash is
   begin
      if S'Length = 0 then
         return Hash_Strings.Null_Bounded_String;
      elsif S'Length > Max_Hash_Length then
         return Hash_Strings.To_Bounded_String (S (S'First .. S'First + Max_Hash_Length - 1));
      else
         return Hash_Strings.To_Bounded_String (S);
      end if;
   end To_Hash;

   function To_Int (S : String; Default : Natural := 0) return Natural is
   begin
      if S'Length = 0 then
         return Default;
      end if;
      return Natural'Value (S);
   exception
      when others =>
         return Default;
   end To_Int;

   procedure Next (State : in out Parser_State; Status : out Status_Code) is
   begin
      Next_Token (State, Status);
   end Next;

   procedure Skip (State : in out Parser_State; Status : out Status_Code) is
   begin
      Skip_Value (State, Status);
   end Skip;

   procedure Read_Member
     (State : in out Parser_State;
      Name  :    out Identifier_String;
      Status:    out Status_Code)
   is
   begin
      Name := Identifier_Strings.Null_Bounded_String;
      if State.Current_Token /= Token_String then
         Status := Error_Parse;
         return;
      end if;

      Name := Identifier_Strings.To_Bounded_String (JSON_Strings.To_String (State.Token_Value));
      Next (State, Status);
      if Status /= Success or else State.Current_Token /= Token_Colon then
         Status := Error_Parse;
         return;
      end if;
      Next (State, Status);
   end Read_Member;

   procedure Parse_Source_Location
     (State  : in out Parser_State;
      Loc    :    out Semantic_IR.Nodes.Source_Location;
      Status :    out Status_Code)
   is
      Member_Name : Identifier_String;
   begin
      Loc := (File => Semantic_IR.Types.Path_Strings.Null_Bounded_String, Line => 1, Column => 1, Length => 0);
      if State.Current_Token /= Token_Object_Start then
         Status := Error_Parse;
         return;
      end if;

      Next (State, Status);
      while Status = Success and then State.Current_Token /= Token_Object_End loop
         Read_Member (State, Member_Name, Status);
         exit when Status /= Success;
         declare
            Key : constant String := Identifier_Strings.To_String (Member_Name);
         begin
            if Key = "file" and then State.Current_Token = Token_String then
               Loc.File := To_Path (JSON_Strings.To_String (State.Token_Value));
            elsif Key = "line" and then State.Current_Token = Token_Number then
               Loc.Line := To_Int (JSON_Strings.To_String (State.Token_Value), 1);
            elsif Key = "column" and then State.Current_Token = Token_Number then
               Loc.Column := To_Int (JSON_Strings.To_String (State.Token_Value), 1);
            elsif Key = "length" and then State.Current_Token = Token_Number then
               Loc.Length := To_Int (JSON_Strings.To_String (State.Token_Value), 0);
            else
               Skip (State, Status);
            end if;
         end;

         if State.Current_Token = Token_Comma then
            Next (State, Status);
         end if;
      end loop;

      if State.Current_Token = Token_Object_End then
         Next (State, Status);
      end if;
   end Parse_Source_Location;

   procedure Parse_Type_Reference
     (State  : in out Parser_State;
      T      :    out Type_Reference;
      Status :    out Status_Code)
   is
      Member_Name : Identifier_String;
      Kind_Str    : String := "";
      Prim_Str    : String := "";
      Name_Str    : String := "";
   begin
      T := (Kind => TK_Primitive, Prim_Type => Type_Void);
      if State.Current_Token /= Token_Object_Start then
         Status := Error_Parse;
         return;
      end if;

      Next (State, Status);
      while Status = Success and then State.Current_Token /= Token_Object_End loop
         Read_Member (State, Member_Name, Status);
         exit when Status /= Success;
         declare
            Key : constant String := Identifier_Strings.To_String (Member_Name);
         begin
            if Key = "kind" and then State.Current_Token = Token_String then
               Kind_Str := JSON_Strings.To_String (State.Token_Value);
            elsif Key = "primitive" and then State.Current_Token = Token_String then
               Prim_Str := JSON_Strings.To_String (State.Token_Value);
            elsif Key = "name" and then State.Current_Token = Token_String then
               Name_Str := JSON_Strings.To_String (State.Token_Value);
            else
               Skip (State, Status);
            end if;
         end;

         if State.Current_Token = Token_Comma then
            Next (State, Status);
         end if;
      end loop;

      if Kind_Str = "primitive_type" then
         T := (Kind => TK_Primitive, Prim_Type => Parse_Primitive_Type (Prim_Str));
      elsif Kind_Str = "type_ref" then
         T := (Kind => TK_Ref, Type_Name => To_Name (Name_Str), Type_Binding => Name_Strings.Null_Bounded_String);
      end if;

      if State.Current_Token = Token_Object_End then
         Next (State, Status);
      end if;
   end Parse_Type_Reference;

   procedure Parse_Expression
     (State  : in out Parser_State;
      Nodes  : in out Node_Table;
      Expr_ID:    out Node_ID;
      Status :    out Status_Code)
   is
      Member_Name : Identifier_String;
      Kind_Str    : String := "";
      NodeId_Str  : String := "";
      Hash_Str    : String := "";
      Name_Str    : String := "";
      Bind_Str    : String := "";
      Int_Str     : String := "";
      Str_Str     : String := "";
      Bool_Str    : String := "";
      Location    : Semantic_IR.Nodes.Source_Location := (File => Semantic_IR.Types.Path_Strings.Null_Bounded_String, Line => 1, Column => 1, Length => 0);
      Expr_Type   : Type_Reference := (Kind => TK_Primitive, Prim_Type => Type_Void);
      Left_ID     : Node_ID := Name_Strings.Null_Bounded_String;
      Right_ID    : Node_ID := Name_Strings.Null_Bounded_String;
      Cond_ID     : Node_ID := Name_Strings.Null_Bounded_String;
      Then_ID     : Node_ID := Name_Strings.Null_Bounded_String;
      Else_ID     : Node_ID := Name_Strings.Null_Bounded_String;
      Op_Bin      : Binary_Operator := Op_Add;
      Added       : Boolean;
   begin
      Expr_ID := Name_Strings.Null_Bounded_String;

      if State.Current_Token = Token_String then
         Expr_ID := To_Node_ID (JSON_Strings.To_String (State.Token_Value));
         Next (State, Status);
         return;
      end if;

      if State.Current_Token /= Token_Object_Start then
         Status := Error_Parse;
         return;
      end if;

      Next (State, Status);
      while Status = Success and then State.Current_Token /= Token_Object_End loop
         Read_Member (State, Member_Name, Status);
         exit when Status /= Success;
         declare
            Key : constant String := Identifier_Strings.To_String (Member_Name);
         begin
            if Key = "kind" and then State.Current_Token = Token_String then
               Kind_Str := JSON_Strings.To_String (State.Token_Value);
            elsif Key = "node_id" and then State.Current_Token = Token_String then
               NodeId_Str := JSON_Strings.To_String (State.Token_Value);
            elsif Key = "hash" and then State.Current_Token = Token_String then
               Hash_Str := JSON_Strings.To_String (State.Token_Value);
            elsif Key = "name" and then State.Current_Token = Token_String then
               Name_Str := JSON_Strings.To_String (State.Token_Value);
            elsif Key = "value" and then State.Current_Token = Token_Number then
               Int_Str := JSON_Strings.To_String (State.Token_Value);
               Str_Str := JSON_Strings.To_String (State.Token_Value);
            elsif Key = "value" and then State.Current_Token = Token_String then
               Str_Str := JSON_Strings.To_String (State.Token_Value);
            elsif Key = "value" and then (State.Current_Token = Token_True or else State.Current_Token = Token_False) then
               Bool_Str := JSON_Strings.To_String (State.Token_Value);
            elsif Key = "binding" and then State.Current_Token = Token_String then
               Bind_Str := JSON_Strings.To_String (State.Token_Value);
            elsif Key = "op" and then State.Current_Token = Token_String then
               declare
                  Op_Str : constant String := JSON_Strings.To_String (State.Token_Value);
               begin
                  Op_Bin := Parse_Binary_Operator (Op_Str);
               end;
            elsif Key = "type" and then State.Current_Token = Token_Object_Start then
               Parse_Type_Reference (State, Expr_Type, Status);
            elsif Key = "location" and then State.Current_Token = Token_Object_Start then
               Parse_Source_Location (State, Location, Status);
            elsif Key = "left" then
               Parse_Expression (State, Nodes, Left_ID, Status);
            elsif Key = "right" then
               Parse_Expression (State, Nodes, Right_ID, Status);
            elsif Key = "condition" then
               Parse_Expression (State, Nodes, Cond_ID, Status);
            elsif Key = "then_expr" then
               Parse_Expression (State, Nodes, Then_ID, Status);
            elsif Key = "else_expr" then
               Parse_Expression (State, Nodes, Else_ID, Status);
            else
               Skip (State, Status);
            end if;
         end;

         if State.Current_Token = Token_Comma then
            Next (State, Status);
         end if;
      end loop;

      Expr_ID := To_Node_ID (NodeId_Str);
      declare
         Kind : constant IR_Node_Kind := Parse_Node_Kind (Kind_Str);
      begin
         case Kind is
            when Kind_Binary_Expr =>
               declare
                  Expr_Node : constant Expression_Node (Kind_Binary_Expr) :=
                    (Kind      => Kind_Binary_Expr,
                     Base      => (Kind     => Kind_Binary_Expr,
                                  ID       => Expr_ID,
                                  Location => Location,
                                  Hash     => To_Hash (Hash_Str)),
                     Expr_Type => Expr_Type);
               begin
                  Add_Expression (Nodes,
                    (Kind => Kind_Binary_Expr,
                     Bin  => (Base => Expr_Node, Operator => Op_Bin, Left_ID => Left_ID, Right_ID => Right_ID)),
                    Added);
               end;
            when Kind_Ternary_Expr =>
               declare
                  Expr_Node : constant Expression_Node (Kind_Ternary_Expr) :=
                    (Kind      => Kind_Ternary_Expr,
                     Base      => (Kind     => Kind_Ternary_Expr,
                                  ID       => Expr_ID,
                                  Location => Location,
                                  Hash     => To_Hash (Hash_Str)),
                     Expr_Type => Expr_Type);
               begin
                  Add_Expression (Nodes,
                    (Kind => Kind_Ternary_Expr,
                     Ter  => (Base => Expr_Node, Condition_ID => Cond_ID, Then_ID => Then_ID, Else_ID => Else_ID)),
                    Added);
               end;
            when Kind_Var_Ref =>
               declare
                  Expr_Node : Expression_Node (Kind_Var_Ref) :=
                    (Kind      => Kind_Var_Ref,
                     Base      => (Kind     => Kind_Var_Ref,
                                  ID       => Expr_ID,
                                  Location => Location,
                                  Hash     => To_Hash (Hash_Str),
                                  Var_Name => Name_Strings.Null_Bounded_String,
                                  Var_Binding => Name_Strings.Null_Bounded_String),
                     Expr_Type => Expr_Type);
               begin
                  Expr_Node.Base.Var_Name := To_Name (Name_Str);
                  Expr_Node.Base.Var_Binding := To_Node_ID (Bind_Str);
                  Add_Expression_Node (Nodes, Expr_Node, Added);
               end;
            when Kind_Integer_Literal =>
               declare
                  Expr_Node : Expression_Node (Kind_Integer_Literal) :=
                    (Kind      => Kind_Integer_Literal,
                     Base      => (Kind     => Kind_Integer_Literal,
                                  ID       => Expr_ID,
                                  Location => Location,
                                  Hash     => To_Hash (Hash_Str),
                                  Int_Value => 0,
                                  Int_Radix => 10),
                     Expr_Type => Expr_Type);
               begin
                  Expr_Node.Base.Int_Value := Long_Long_Integer (To_Int (Int_Str, 0));
                  Expr_Node.Base.Int_Radix := 10;
                  Add_Expression_Node (Nodes, Expr_Node, Added);
               end;
            when Kind_Float_Literal =>
               declare
                  Expr_Node : Expression_Node (Kind_Float_Literal) :=
                    (Kind      => Kind_Float_Literal,
                     Base      => (Kind     => Kind_Float_Literal,
                                  ID       => Expr_ID,
                                  Location => Location,
                                  Hash     => To_Hash (Hash_Str),
                                  Float_Value => 0.0),
                     Expr_Type => Expr_Type);
                  Float_Val : Long_Float := 0.0;
               begin
                  if Str_Str'Length > 0 then
                     begin
                        Float_Val := Long_Float'Value (Str_Str);
                     exception
                        when others =>
                           Float_Val := 0.0;
                     end;
                  end if;
                  Expr_Node.Base.Float_Value := Float_Val;
                  Add_Expression_Node (Nodes, Expr_Node, Added);
               end;
            when Kind_String_Literal =>
               declare
                  Expr_Node : Expression_Node (Kind_String_Literal) :=
                    (Kind      => Kind_String_Literal,
                     Base      => (Kind     => Kind_String_Literal,
                                  ID       => Expr_ID,
                                  Location => Location,
                                  Hash     => To_Hash (Hash_Str),
                                  Str_Value => Name_Strings.Null_Bounded_String),
                     Expr_Type => Expr_Type);
               begin
                  Expr_Node.Base.Str_Value := To_Name (Str_Str);
                  Add_Expression_Node (Nodes, Expr_Node, Added);
               end;
            when Kind_Bool_Literal =>
               declare
                  Expr_Node : Expression_Node (Kind_Bool_Literal) :=
                    (Kind      => Kind_Bool_Literal,
                     Base      => (Kind     => Kind_Bool_Literal,
                                  ID       => Expr_ID,
                                  Location => Location,
                                  Hash     => To_Hash (Hash_Str),
                                  Bool_Value => False),
                     Expr_Type => Expr_Type);
               begin
                  Expr_Node.Base.Bool_Value := (Bool_Str = "true");
                  Add_Expression_Node (Nodes, Expr_Node, Added);
               end;
            when others =>
               declare
                  Expr_Node : constant Expression_Node (Kind_Var_Ref) :=
                    (Kind      => Kind_Var_Ref,
                     Base      => (Kind     => Kind_Var_Ref,
                                  ID       => Expr_ID,
                                  Location => Location,
                                  Hash     => To_Hash (Hash_Str),
                                  Var_Name => Name_Strings.Null_Bounded_String,
                                  Var_Binding => Name_Strings.Null_Bounded_String),
                     Expr_Type => Expr_Type);
               begin
                  Add_Expression_Node (Nodes, Expr_Node, Added);
               end;
         end case;
      end;

      if State.Current_Token = Token_Object_End then
         Next (State, Status);
      end if;
   end Parse_Expression;

   procedure Parse_Statement
     (State  : in out Parser_State;
      Nodes  : in out Node_Table;
      Stmt_ID:    out Node_ID;
      Status :    out Status_Code)
   is
      Member_Name : Identifier_String;
      Kind_Str    : String := "";
      NodeId_Str  : String := "";
      Hash_Str    : String := "";
      Location    : Semantic_IR.Nodes.Source_Location := (File => Semantic_IR.Types.Path_Strings.Null_Bounded_String, Line => 1, Column => 1, Length => 0);
      Value_ID    : Node_ID := Name_Strings.Null_Bounded_String;
         Added       : Boolean;
   begin
      Stmt_ID := Name_Strings.Null_Bounded_String;
      if State.Current_Token /= Token_Object_Start then
         Status := Error_Parse;
         return;
      end if;

      Next (State, Status);
      while Status = Success and then State.Current_Token /= Token_Object_End loop
         Read_Member (State, Member_Name, Status);
         exit when Status /= Success;
         declare
            Key : constant String := Identifier_Strings.To_String (Member_Name);
         begin
            if Key = "kind" and then State.Current_Token = Token_String then
               Kind_Str := JSON_Strings.To_String (State.Token_Value);
            elsif Key = "node_id" and then State.Current_Token = Token_String then
               NodeId_Str := JSON_Strings.To_String (State.Token_Value);
            elsif Key = "hash" and then State.Current_Token = Token_String then
               Hash_Str := JSON_Strings.To_String (State.Token_Value);
            elsif Key = "value" then
               Parse_Expression (State, Nodes, Value_ID, Status);
            elsif Key = "location" and then State.Current_Token = Token_Object_Start then
               Parse_Source_Location (State, Location, Status);
            else
               Skip (State, Status);
            end if;
         end;

         if State.Current_Token = Token_Comma then
            Next (State, Status);
         end if;
      end loop;

      Stmt_ID := To_Node_ID (NodeId_Str);
      declare
         Kind : constant IR_Node_Kind := Parse_Node_Kind (Kind_Str);
      begin
         if Kind = Kind_Return_Stmt then
            declare
               Stmt_Node : constant Statement_Node (Kind_Return_Stmt) :=
                 (Kind     => Kind_Return_Stmt,
                  Base     => (Kind     => Kind_Return_Stmt,
                               ID       => Stmt_ID,
                               Location => Location,
                               Hash     => To_Hash (Hash_Str)));
            begin
               Add_Statement (Nodes,
                 (Kind => Kind_Return_Stmt, Ret => (Base => Stmt_Node, Value_ID => Value_ID)),
                 Added);
            end;
         else
            declare
               Stmt_Node : constant Statement_Node (Kind_Expr_Stmt) :=
                 (Kind     => Kind_Expr_Stmt,
                  Base     => (Kind     => Kind_Expr_Stmt,
                               ID       => Stmt_ID,
                               Location => Location,
                               Hash     => To_Hash (Hash_Str)));
            begin
               Add_Statement (Nodes,
                 (Kind => Kind_Expr_Stmt,
                  Expr => (Base => Stmt_Node, Expr_ID => Name_Strings.Null_Bounded_String)),
                 Added);
            end;
         end if;
      end;

      if State.Current_Token = Token_Object_End then
         Next (State, Status);
      end if;
   end Parse_Statement;


   procedure Parse_Declaration
     (State  : in out Parser_State;
      Nodes  : in out Node_Table;
      Decl_ID:    out Node_ID;
      Status :    out Status_Code)
   is
      Member_Name : Identifier_String;
      Kind_Str    : String := "";
      Decl        : Function_Declaration :=
        (Base => (Kind => Kind_Function_Decl,
                  Base => (Kind => Kind_Function_Decl,
                           ID => Name_Strings.Null_Bounded_String,
                           Location => (File => Semantic_IR.Types.Path_Strings.Null_Bounded_String, Line => 1, Column => 1, Length => 0),
                           Hash => Hash_Strings.Null_Bounded_String),
                  Decl_Name => Name_Strings.Null_Bounded_String,
                  Visibility => Vis_Public),
         Return_Type => (Kind => TK_Primitive, Prim_Type => Type_Void),
         Param_Count => 0,
         Parameters  => (others => Name_Strings.Null_Bounded_String),
         Body_ID     => Name_Strings.Null_Bounded_String,
         Inline      => Inline_None,
         Is_Pure     => False,
         Stack_Usage => 0,
         Priority    => 0,
         Interrupt_Vec => 0,
         Entry_Point => False);
      Decl_Record : Declaration_Record;
      Added       : Boolean;
   begin
      Decl_ID := Name_Strings.Null_Bounded_String;
      if State.Current_Token /= Token_Object_Start then
         Status := Error_Parse;
         return;
      end if;

      Next (State, Status);
      while Status = Success and then State.Current_Token /= Token_Object_End loop
         Read_Member (State, Member_Name, Status);
         exit when Status /= Success;
         declare
            Key : constant String := Identifier_Strings.To_String (Member_Name);
         begin
            if Key = "kind" and then State.Current_Token = Token_String then
               Kind_Str := JSON_Strings.To_String (State.Token_Value);
            elsif Key = "name" and then State.Current_Token = Token_String then
               Decl.Base.Decl_Name := To_Name (JSON_Strings.To_String (State.Token_Value));
            elsif Key = "node_id" and then State.Current_Token = Token_String then
               Decl.Base.Base.ID := To_Node_ID (JSON_Strings.To_String (State.Token_Value));
            elsif Key = "hash" and then State.Current_Token = Token_String then
               Decl.Base.Base.Hash := To_Hash (JSON_Strings.To_String (State.Token_Value));
            elsif Key = "visibility" and then State.Current_Token = Token_String then
               Decl.Base.Visibility := Parse_Visibility (JSON_Strings.To_String (State.Token_Value));
            elsif Key = "inline" and then State.Current_Token = Token_String then
               Decl.Inline := Parse_Inline_Hint (JSON_Strings.To_String (State.Token_Value));
            elsif Key = "function_type" and then State.Current_Token = Token_Object_Start then
               declare
                  Member_Key : Identifier_String;
               begin
                  Next (State, Status);
                  while Status = Success and then State.Current_Token /= Token_Object_End loop
                     Read_Member (State, Member_Key, Status);
                     exit when Status /= Success;
                     if Identifier_Strings.To_String (Member_Key) = "return_type" then
                        Parse_Type_Reference (State, Decl.Return_Type, Status);
                     elsif Identifier_Strings.To_String (Member_Key) = "parameters" and then State.Current_Token = Token_Array_Start then
                        Next (State, Status);
                        while Status = Success and then State.Current_Token /= Token_Array_End loop
                           declare
                              P_Name : String := "";
                              P_Key  : Identifier_String;
                           begin
                              if State.Current_Token = Token_Object_Start then
                                 Next (State, Status);
                                 while Status = Success and then State.Current_Token /= Token_Object_End loop
                                    Read_Member (State, P_Key, Status);
                                    exit when Status /= Success;
                                    if Identifier_Strings.To_String (P_Key) = "name" and then State.Current_Token = Token_String then
                                       P_Name := JSON_Strings.To_String (State.Token_Value);
                                    else
                                       Skip (State, Status);
                                    end if;
                                    if State.Current_Token = Token_Comma then
                                       Next (State, Status);
                                    end if;
                                 end loop;
                                 if State.Current_Token = Token_Object_End then
                                    Next (State, Status);
                                 end if;
                              else
                                 Skip (State, Status);
                              end if;

                              if Decl.Param_Count < Semantic_IR.Declarations.Max_Parameters then
                                 Decl.Param_Count := Decl.Param_Count + 1;
                                 Decl.Parameters (Decl.Param_Count) := To_Node_ID (P_Name);
                              end if;

                              if State.Current_Token = Token_Comma then
                                 Next (State, Status);
                              end if;
                           end;
                        end loop;
                        if State.Current_Token = Token_Array_End then
                           Next (State, Status);
                        end if;
                     else
                        Skip (State, Status);
                     end if;
                     if State.Current_Token = Token_Comma then
                        Next (State, Status);
                     end if;
                  end loop;
                  if State.Current_Token = Token_Object_End then
                     Next (State, Status);
                  end if;
               end;
            elsif Key = "body" then
               Parse_Statement (State, Nodes, Decl.Body_ID, Status);
            else
               Skip (State, Status);
            end if;
         end;

         if State.Current_Token = Token_Comma then
            Next (State, Status);
         end if;
      end loop;

      if Kind_Str = "function_decl" then
         Decl_Record := (Kind => Kind_Function_Decl, Func => Decl);
         Add_Declaration (Nodes, Decl_Record, Added);
         Decl_ID := Decl.Base.Base.ID;
      elsif Kind_Str = "type_decl" then
         declare
                  T : Type_Declaration :=
                     (Base => (Kind => Kind_Type_Decl,
                                    Base => (Kind => Kind_Type_Decl,
                                                 ID => Name_Strings.Null_Bounded_String,
                                                 Location => (File => Semantic_IR.Types.Path_Strings.Null_Bounded_String, Line => 1, Column => 1, Length => 0),
                                                 Hash => Hash_Strings.Null_Bounded_String),
                                    Decl_Name => Name_Strings.Null_Bounded_String,
                                    Visibility => Vis_Public),
                      Type_Def => (Kind => TK_Primitive, Prim_Type => Type_Void));
            Added : Boolean;
         begin
            if State.Current_Token = Token_Object_Start then
               Next (State, Status);
               while Status = Success and then State.Current_Token /= Token_Object_End loop
                  Read_Member (State, Member_Name, Status);
                  exit when Status /= Success;
                  if Identifier_Strings.To_String (Member_Name) = "name" and then State.Current_Token = Token_String then
                     T.Base.Decl_Name := To_Name (JSON_Strings.To_String (State.Token_Value));
                  elsif Identifier_Strings.To_String (Member_Name) = "node_id" and then State.Current_Token = Token_String then
                     T.Base.Base.ID := To_Node_ID (JSON_Strings.To_String (State.Token_Value));
                  elsif Identifier_Strings.To_String (Member_Name) = "hash" and then State.Current_Token = Token_String then
                     T.Base.Base.Hash := To_Hash (JSON_Strings.To_String (State.Token_Value));
                  elsif Identifier_Strings.To_String (Member_Name) = "type_definition" and then State.Current_Token = Token_Object_Start then
                     Parse_Type_Reference (State, T.Type_Def, Status);
                  else
                     Skip (State, Status);
                  end if;
                  if State.Current_Token = Token_Comma then
                     Next (State, Status);
                  end if;
               end loop;
               if State.Current_Token = Token_Object_End then
                  Next (State, Status);
               end if;
            end if;
            Add_Type_Declaration (Nodes, T, Added);
            Decl_ID := T.Base.Base.ID;
         end;
      else
         Status := Error_Not_Implemented;
      end if;
   end Parse_Declaration;

   procedure Parse_Imports
     (State  : in out Parser_State;
      Module : in out IR_Module;
      Status :    out Status_Code)
   is
      Member_Name : Identifier_String;
      Import_Item : Import_Statement;
   begin
      if State.Current_Token /= Token_Array_Start then
         Status := Error_Parse;
         return;
      end if;

      Next (State, Status);
      while Status = Success and then State.Current_Token /= Token_Array_End loop
         Import_Item := (Module_Name => Name_Strings.Null_Bounded_String,
                         Symbol_Count => 0,
                         Symbols => (others => Name_Strings.Null_Bounded_String),
                         Import_All => False,
                         Alias => Name_Strings.Null_Bounded_String);

         if State.Current_Token = Token_Object_Start then
            Next (State, Status);
            while Status = Success and then State.Current_Token /= Token_Object_End loop
               Read_Member (State, Member_Name, Status);
               exit when Status /= Success;
               if Identifier_Strings.To_String (Member_Name) = "module" and then State.Current_Token = Token_String then
                  Import_Item.Module_Name := To_Name (JSON_Strings.To_String (State.Token_Value));
               elsif Identifier_Strings.To_String (Member_Name) = "symbols" and then State.Current_Token = Token_Array_Start then
                  Next (State, Status);
                  while Status = Success and then State.Current_Token /= Token_Array_End loop
                     if State.Current_Token = Token_String then
                        if Import_Item.Symbol_Count < Max_Symbols then
                           Import_Item.Symbol_Count := Import_Item.Symbol_Count + 1;
                           Import_Item.Symbols (Import_Item.Symbol_Count) := To_Name (JSON_Strings.To_String (State.Token_Value));
                        end if;
                        Next (State, Status);
                     elsif State.Current_Token = Token_Comma then
                        Next (State, Status);
                     else
                        Skip (State, Status);
                     end if;
                  end loop;
                  if State.Current_Token = Token_Array_End then
                     Next (State, Status);
                  end if;
               elsif Identifier_Strings.To_String (Member_Name) = "alias" and then State.Current_Token = Token_String then
                  Import_Item.Alias := To_Name (JSON_Strings.To_String (State.Token_Value));
               else
                  Skip (State, Status);
               end if;
               if State.Current_Token = Token_Comma then
                  Next (State, Status);
               end if;
            end loop;
            if State.Current_Token = Token_Object_End then
               Next (State, Status);
            end if;
            declare
               Added : Boolean;
            begin
               Add_Import (Module, Import_Item, Added);
            end;
         else
            Skip (State, Status);
         end if;

         if State.Current_Token = Token_Comma then
            Next (State, Status);
         end if;
      end loop;

      if State.Current_Token = Token_Array_End then
         Next (State, Status);
      end if;
   end Parse_Imports;

   procedure Parse_Module
     (State  : in out Parser_State;
      Module :    out IR_Module;
      Nodes  : in out Node_Table;
      Status :    out Status_Code)
   is
      Member_Name : Identifier_String;
   begin
      Module := (Base => (Kind => Kind_Module,
                 ID => Name_Strings.Null_Bounded_String,
                 Location => (File => Semantic_IR.Types.Path_Strings.Null_Bounded_String, Line => 1, Column => 1, Length => 0),
                 Hash => Hash_Strings.Null_Bounded_String),
                 Module_Name => Name_Strings.Null_Bounded_String,
                 Import_Count => 0,
                 Imports => (others => (Module_Name => Name_Strings.Null_Bounded_String, Symbol_Count => 0, Symbols => (others => Name_Strings.Null_Bounded_String), Import_All => False, Alias => Name_Strings.Null_Bounded_String)),
                 Export_Count => 0,
                 Exports => (others => Name_Strings.Null_Bounded_String),
                 Decl_Count => 0,
                 Declarations => (others => Name_Strings.Null_Bounded_String),
                 Metadata => (Target_Count => 0, Target_Categories => (others => Target_Native), Module_Safety => Level_None, Optimization_Level => 0));

      if State.Current_Token /= Token_Object_Start then
         Status := Error_Parse;
         return;
      end if;

      Next (State, Status);
      while Status = Success and then State.Current_Token /= Token_Object_End loop
         Read_Member (State, Member_Name, Status);
         exit when Status /= Success;
         declare
            Key : constant String := Identifier_Strings.To_String (Member_Name);
         begin
            if Key = "name" and then State.Current_Token = Token_String then
               Module.Module_Name := To_Name (JSON_Strings.To_String (State.Token_Value));
            elsif Key = "node_id" and then State.Current_Token = Token_String then
               Module.Base.ID := To_Node_ID (JSON_Strings.To_String (State.Token_Value));
            elsif Key = "hash" and then State.Current_Token = Token_String then
               Module.Base.Hash := To_Hash (JSON_Strings.To_String (State.Token_Value));
            elsif Key = "location" and then State.Current_Token = Token_Object_Start then
               Parse_Source_Location (State, Module.Base.Location, Status);
            elsif Key = "imports" and then State.Current_Token = Token_Array_Start then
               Parse_Imports (State, Module, Status);
            elsif Key = "exports" and then State.Current_Token = Token_Array_Start then
               Next (State, Status);
               while Status = Success and then State.Current_Token /= Token_Array_End loop
                  if State.Current_Token = Token_String then
                     if Module.Export_Count < Max_Exports then
                        Module.Export_Count := Module.Export_Count + 1;
                        Module.Exports (Module.Export_Count) := To_Name (JSON_Strings.To_String (State.Token_Value));
                     end if;
                     Next (State, Status);
                  elsif State.Current_Token = Token_Comma then
                     Next (State, Status);
                  else
                     Skip (State, Status);
                  end if;
               end loop;
               if State.Current_Token = Token_Array_End then
                  Next (State, Status);
               end if;
            elsif Key = "declarations" and then State.Current_Token = Token_Array_Start then
               Next (State, Status);
               while Status = Success and then State.Current_Token /= Token_Array_End loop
                  declare
                     Decl_ID : Node_ID := Name_Strings.Null_Bounded_String;
                     Added   : Boolean;
                  begin
                     Parse_Declaration (State, Nodes, Decl_ID, Status);
                     if Status /= Success then
                        return;
                     end if;
                     if Decl_ID /= Name_Strings.Null_Bounded_String then
                        Add_Declaration (Module, Decl_ID, Added);
                     end if;
                     if State.Current_Token = Token_Comma then
                        Next (State, Status);
                     end if;
                  end;
               end loop;
               if State.Current_Token = Token_Array_End then
                  Next (State, Status);
               end if;
            else
               Skip (State, Status);
            end if;
         end;

         if State.Current_Token = Token_Comma then
            Next (State, Status);
         end if;
      end loop;

      if State.Current_Token = Token_Object_End then
         Next (State, Status);
      end if;
   end Parse_Module;

   procedure Parse_IR_JSON
     (JSON_Content : in     JSON_String;
      Module       :    out Semantic_IR.Modules.IR_Module;
      Nodes        :    out STUNIR.Emitters.Node_Table.Node_Table;
      Status       :    out Status_Code)
   is
      Parser : Parser_State;
      Member_Name : Identifier_String;
   begin
      Initialize (Nodes);
      Initialize_Parser (Parser, JSON_Content, Status);
      if Status /= Success then
         return;
      end if;

      Next (Parser, Status);
      if Status /= Success or else Parser.Current_Token /= Token_Object_Start then
         Status := Error_Invalid_JSON;
         return;
      end if;

      Next (Parser, Status);
      while Status = Success and then Parser.Current_Token /= Token_Object_End loop
         Read_Member (Parser, Member_Name, Status);
         exit when Status /= Success;
         if Identifier_Strings.To_String (Member_Name) = "root" and then Parser.Current_Token = Token_Object_Start then
            Parse_Module (Parser, Module, Nodes, Status);
         else
            Skip (Parser, Status);
         end if;

         if Parser.Current_Token = Token_Comma then
            Next (Parser, Status);
         end if;
      end loop;
   end Parse_IR_JSON;

end Semantic_IR.JSON;
