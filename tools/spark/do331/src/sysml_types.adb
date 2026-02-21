--  STUNIR DO-331 SysML 2.0 Type Definitions Implementation
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

package body SysML_Types is

   --  ============================================================
   --  SysML 2.0 Keywords
   --  ============================================================

   function Get_Keyword (Kind : Element_Kind) return String is
   begin
      case Kind is
         when Package_Element     => return "package";
         when Part_Element        => return "part def";
         when Attribute_Element   => return "attribute";
         when Connection_Element  => return "connection";
         when Port_Element        => return "port";
         when Interface_Element   => return "interface def";
         when Action_Element      => return "action def";
         when State_Element       => return "state";
         when Transition_Element  => return "transition";
         when Activity_Element    => return "activity";
         when Requirement_Element => return "requirement def";
         when Constraint_Element  => return "constraint def";
         when Satisfy_Element     => return "satisfy";
         when Verify_Element      => return "verify";
         when Derive_Element      => return "derive";
         when Allocate_Element    => return "allocate";
         when Comment_Element     => return "doc";
         when Import_Element      => return "import";
      end case;
   end Get_Keyword;

   --  ============================================================
   --  Type Mapping
   --  ============================================================

   function IR_Type_To_SysML (IR_Type : String) return SysML_Primitive_Type is
   begin
      if IR_Type = "bool" or IR_Type = "boolean" or IR_Type = "Boolean" then
         return Boolean_Type;
      elsif IR_Type = "int" or IR_Type = "i32" or IR_Type = "int32" or
            IR_Type = "i64" or IR_Type = "int64" or IR_Type = "Integer" then
         return Integer_Type;
      elsif IR_Type = "nat" or IR_Type = "u32" or IR_Type = "uint32" or
            IR_Type = "u64" or IR_Type = "uint64" or IR_Type = "Natural" then
         return Natural_Type;
      elsif IR_Type = "float" or IR_Type = "f32" or IR_Type = "float32" or
            IR_Type = "double" or IR_Type = "f64" or IR_Type = "float64" or
            IR_Type = "Real" then
         return Real_Type;
      elsif IR_Type = "string" or IR_Type = "str" or IR_Type = "String" then
         return String_Type;
      else
         return Any_Type;
      end if;
   end IR_Type_To_SysML;

   function Get_Type_Name (T : SysML_Primitive_Type) return String is
   begin
      case T is
         when Boolean_Type  => return "Boolean";
         when Integer_Type  => return "Integer";
         when Natural_Type  => return "Natural";
         when Positive_Type => return "Positive";
         when Real_Type     => return "Real";
         when String_Type   => return "String";
         when Any_Type      => return "Anything";
      end case;
   end Get_Type_Name;

   --  ============================================================
   --  Operator Mapping
   --  ============================================================

   function IR_Op_To_SysML (IR_Op : String) return SysML_Operator is
   begin
      if IR_Op = "+" or IR_Op = "add" or IR_Op = "Add" then
         return Op_Add;
      elsif IR_Op = "-" or IR_Op = "sub" or IR_Op = "Sub" then
         return Op_Subtract;
      elsif IR_Op = "*" or IR_Op = "mul" or IR_Op = "Mul" then
         return Op_Multiply;
      elsif IR_Op = "/" or IR_Op = "div" or IR_Op = "Div" then
         return Op_Divide;
      elsif IR_Op = "%" or IR_Op = "mod" or IR_Op = "Mod" then
         return Op_Modulo;
      elsif IR_Op = "==" or IR_Op = "eq" or IR_Op = "Eq" then
         return Op_Equal;
      elsif IR_Op = "!=" or IR_Op = "neq" or IR_Op = "Neq" or IR_Op = "<>" then
         return Op_Not_Equal;
      elsif IR_Op = "<" or IR_Op = "lt" or IR_Op = "Lt" then
         return Op_Less;
      elsif IR_Op = "<=" or IR_Op = "le" or IR_Op = "Le" then
         return Op_Less_Equal;
      elsif IR_Op = ">" or IR_Op = "gt" or IR_Op = "Gt" then
         return Op_Greater;
      elsif IR_Op = ">=" or IR_Op = "ge" or IR_Op = "Ge" then
         return Op_Greater_Equal;
      elsif IR_Op = "&&" or IR_Op = "and" or IR_Op = "And" then
         return Op_And;
      elsif IR_Op = "||" or IR_Op = "or" or IR_Op = "Or" then
         return Op_Or;
      elsif IR_Op = "!" or IR_Op = "not" or IR_Op = "Not" then
         return Op_Not;
      elsif IR_Op = "=" or IR_Op = ":=" or IR_Op = "assign" then
         return Op_Assign;
      else
         return Op_Unknown;
      end if;
   end IR_Op_To_SysML;

   function Get_Operator_Symbol (Op : SysML_Operator) return String is
   begin
      case Op is
         when Op_Add           => return "+";
         when Op_Subtract      => return "-";
         when Op_Multiply      => return "*";
         when Op_Divide        => return "/";
         when Op_Modulo        => return "%";
         when Op_Equal         => return "==";
         when Op_Not_Equal     => return "!=";
         when Op_Less          => return "<";
         when Op_Less_Equal    => return "<=";
         when Op_Greater       => return ">";
         when Op_Greater_Equal => return ">=";
         when Op_And           => return "and";
         when Op_Or            => return "or";
         when Op_Not           => return "not";
         when Op_Assign        => return ":=";
         when Op_Unknown       => return "?";
      end case;
   end Get_Operator_Symbol;

   --  ============================================================
   --  State Machine Types
   --  ============================================================

   function Get_State_Keyword (Kind : SysML_State_Kind) return String is
   begin
      case Kind is
         when State_Simple       => return "state";
         when State_Composite    => return "state";
         when State_Initial      => return "entry state";
         when State_Final        => return "final state";
         when State_Choice       => return "decide";
         when State_Fork         => return "fork";
         when State_Join         => return "join";
         when State_History      => return "history state";
         when State_Deep_History => return "history state";
      end case;
   end Get_State_Keyword;

   --  ============================================================
   --  Relationship Types
   --  ============================================================

   function Get_Relationship_Keyword (Rel : SysML_Relationship) return String is
   begin
      case Rel is
         when Rel_Satisfy  => return "satisfy";
         when Rel_Verify   => return "verify";
         when Rel_Derive   => return "derive";
         when Rel_Refine   => return "refine";
         when Rel_Trace    => return "trace";
         when Rel_Allocate => return "allocate";
         when Rel_Copy     => return "copy";
         when Rel_Expose   => return "expose";
      end case;
   end Get_Relationship_Keyword;

   --  ============================================================
   --  Standard Library References
   --  ============================================================

   function Get_Standard_Import (T : SysML_Primitive_Type) return String is
   begin
      case T is
         when Boolean_Type  => return "ScalarValues::Boolean";
         when Integer_Type  => return "ScalarValues::Integer";
         when Natural_Type  => return "ScalarValues::Natural";
         when Positive_Type => return "ScalarValues::Positive";
         when Real_Type     => return "ScalarValues::Real";
         when String_Type   => return "ScalarValues::String";
         when Any_Type      => return "Base::Anything";
      end case;
   end Get_Standard_Import;

end SysML_Types;
