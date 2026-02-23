-- STUNIR IR Types Package Body
-- DO-178C Level A Compliant
-- SPARK 2014 Mode

pragma SPARK_Mode (On);

with Ada.Characters.Handling;

package body IR.Types is

   function Parse_Node_Kind (Kind_Str : String) return IR_Node_Kind is
      K : constant String := Ada.Characters.Handling.To_Lower (Kind_Str);
   begin
      if K = "module" then
         return Kind_Module;
      elsif K = "function_decl" then
         return Kind_Function_Decl;
      elsif K = "type_decl" then
         return Kind_Type_Decl;
      elsif K = "const_decl" then
         return Kind_Const_Decl;
      elsif K = "var_decl" then
         return Kind_Var_Decl;
      elsif K = "block_stmt" then
         return Kind_Block_Stmt;
      elsif K = "expr_stmt" then
         return Kind_Expr_Stmt;
      elsif K = "if_stmt" then
         return Kind_If_Stmt;
      elsif K = "while_stmt" then
         return Kind_While_Stmt;
      elsif K = "for_stmt" then
         return Kind_For_Stmt;
      elsif K = "return_stmt" then
         return Kind_Return_Stmt;
      elsif K = "break_stmt" then
         return Kind_Break_Stmt;
      elsif K = "continue_stmt" then
         return Kind_Continue_Stmt;
      elsif K = "var_decl_stmt" then
         return Kind_Var_Decl_Stmt;
      elsif K = "assign_stmt" then
         return Kind_Assign_Stmt;
      elsif K = "integer_literal" then
         return Kind_Integer_Literal;
      elsif K = "float_literal" then
         return Kind_Float_Literal;
      elsif K = "string_literal" then
         return Kind_String_Literal;
      elsif K = "bool_literal" then
         return Kind_Bool_Literal;
      elsif K = "var_ref" then
         return Kind_Var_Ref;
      elsif K = "binary_expr" then
         return Kind_Binary_Expr;
      elsif K = "unary_expr" then
         return Kind_Unary_Expr;
      elsif K = "function_call" then
         return Kind_Function_Call;
      elsif K = "member_expr" then
         return Kind_Member_Expr;
      elsif K = "array_access" then
         return Kind_Array_Access;
      elsif K = "cast_expr" then
         return Kind_Cast_Expr;
      elsif K = "ternary_expr" then
         return Kind_Ternary_Expr;
      elsif K = "array_init" then
         return Kind_Array_Init;
      elsif K = "struct_init" then
         return Kind_Struct_Init;
      else
         return Kind_Module;
      end if;
   end Parse_Node_Kind;

   function Parse_Binary_Operator (Op_Str : String) return Binary_Operator is
      Op : constant String := Op_Str;
   begin
      if Op = "+" then
         return Op_Add;
      elsif Op = "-" then
         return Op_Sub;
      elsif Op = "*" then
         return Op_Mul;
      elsif Op = "/" then
         return Op_Div;
      elsif Op = "%" then
         return Op_Mod;
      elsif Op = "==" then
         return Op_Eq;
      elsif Op = "!=" then
         return Op_Neq;
      elsif Op = "<" then
         return Op_Lt;
      elsif Op = "<=" then
         return Op_Leq;
      elsif Op = ">" then
         return Op_Gt;
      elsif Op = ">=" then
         return Op_Geq;
      elsif Op = "&&" then
         return Op_And;
      elsif Op = "||" then
         return Op_Or;
      elsif Op = "&" then
         return Op_Bit_And;
      elsif Op = "|" then
         return Op_Bit_Or;
      elsif Op = "^" then
         return Op_Bit_Xor;
      elsif Op = "<<" then
         return Op_Shl;
      elsif Op = ">>" then
         return Op_Shr;
      elsif Op = "=" then
         return Op_Assign;
      else
         return Op_Add;
      end if;
   end Parse_Binary_Operator;

   function Parse_Unary_Operator (Op_Str : String) return Unary_Operator is
      Op : constant String := Op_Str;
   begin
      if Op = "-" then
         return Op_Neg;
      elsif Op = "!" then
         return Op_Not;
      elsif Op = "~" then
         return Op_Bit_Not;
      elsif Op = "++" then
         return Op_Pre_Inc;
      elsif Op = "--" then
         return Op_Pre_Dec;
      elsif Op = "*" then
         return Op_Deref;
      elsif Op = "&" then
         return Op_Addr_Of;
      else
         return Op_Not;
      end if;
   end Parse_Unary_Operator;

   function Parse_Visibility (Vis_Str : String) return Visibility_Kind is
      V : constant String := Ada.Characters.Handling.To_Lower (Vis_Str);
   begin
      if V = "public" then
         return Vis_Public;
      elsif V = "private" then
         return Vis_Private;
      elsif V = "protected" then
         return Vis_Protected;
      elsif V = "internal" then
         return Vis_Internal;
      else
         return Vis_Public;
      end if;
   end Parse_Visibility;

   function Parse_Mutability (Mut_Str : String) return Mutability_Kind is
      M : constant String := Ada.Characters.Handling.To_Lower (Mut_Str);
   begin
      if M = "mutable" then
         return Mut_Mutable;
      elsif M = "immutable" then
         return Mut_Immutable;
      elsif M = "const" then
         return Mut_Const;
      else
         return Mut_Mutable;
      end if;
   end Parse_Mutability;

   function Parse_Storage_Class (Storage_Str : String) return Storage_Class is
      S : constant String := Ada.Characters.Handling.To_Lower (Storage_Str);
   begin
      if S = "auto" then
         return Storage_Auto;
      elsif S = "static" then
         return Storage_Static;
      elsif S = "extern" then
         return Storage_Extern;
      elsif S = "register" then
         return Storage_Register;
      elsif S = "stack" then
         return Storage_Stack;
      elsif S = "heap" then
         return Storage_Heap;
      elsif S = "global" then
         return Storage_Global;
      else
         return Storage_Auto;
      end if;
   end Parse_Storage_Class;

   function Parse_Inline_Hint (Inline_Str : String) return Inline_Hint is
      I : constant String := Ada.Characters.Handling.To_Lower (Inline_Str);
   begin
      if I = "always" then
         return Inline_Always;
      elsif I = "never" then
         return Inline_Never;
      elsif I = "hint" then
         return Inline_Hint_Suggest;
      else
         return Inline_None;
      end if;
   end Parse_Inline_Hint;

   function Parse_Primitive_Type (Prim_Str : String) return IR_Primitive_Type is
      P : constant String := Ada.Characters.Handling.To_Lower (Prim_Str);
   begin
      if P = "void" then
         return Type_Void;
      elsif P = "bool" then
         return Type_Bool;
      elsif P = "i8" then
         return Type_I8;
      elsif P = "i16" then
         return Type_I16;
      elsif P = "i32" then
         return Type_I32;
      elsif P = "i64" then
         return Type_I64;
      elsif P = "u8" then
         return Type_U8;
      elsif P = "u16" then
         return Type_U16;
      elsif P = "u32" then
         return Type_U32;
      elsif P = "u64" then
         return Type_U64;
      elsif P = "f32" then
         return Type_F32;
      elsif P = "f64" then
         return Type_F64;
      elsif P = "string" then
         return Type_String;
      elsif P = "char" then
         return Type_Char;
      else
         return Type_Void;
      end if;
   end Parse_Primitive_Type;

   function Parse_Target_Category (Cat_Str : String) return Target_Category is
      C : constant String := Ada.Characters.Handling.To_Lower (Cat_Str);
   begin
      if C = "embedded" then
         return Target_Embedded;
      elsif C = "realtime" then
         return Target_Realtime;
      elsif C = "safety_critical" then
         return Target_Safety_Critical;
      elsif C = "gpu" then
         return Target_GPU;
      elsif C = "wasm" then
         return Target_WASM;
      elsif C = "native" then
         return Target_Native;
      elsif C = "jit" then
         return Target_JIT;
      elsif C = "interpreter" then
         return Target_Interpreter;
      elsif C = "functional" then
         return Target_Functional;
      elsif C = "logic" then
         return Target_Logic;
      elsif C = "constraint" then
         return Target_Constraint;
      elsif C = "dataflow" then
         return Target_Dataflow;
      elsif C = "reactive" then
         return Target_Reactive;
      elsif C = "quantum" then
         return Target_Quantum;
      elsif C = "neuromorphic" then
         return Target_Neuromorphic;
      elsif C = "biocomputing" then
         return Target_Biocomputing;
      elsif C = "molecular" then
         return Target_Molecular;
      elsif C = "optical" then
         return Target_Optical;
      elsif C = "reversible" then
         return Target_Reversible;
      elsif C = "analog" then
         return Target_Analog;
      elsif C = "stochastic" then
         return Target_Stochastic;
      elsif C = "fuzzy" then
         return Target_Fuzzy;
      elsif C = "approximate" then
         return Target_Approximate;
      elsif C = "probabilistic" then
         return Target_Probabilistic;
      else
         return Target_Native;
      end if;
   end Parse_Target_Category;

   function Parse_Safety_Level (Level_Str : String) return Safety_Level is
      L : constant String := Ada.Characters.Handling.To_Lower (Level_Str);
   begin
      if L = "do-178c_level_a" then
         return Level_DO178C_A;
      elsif L = "do-178c_level_b" then
         return Level_DO178C_B;
      elsif L = "do-178c_level_c" then
         return Level_DO178C_C;
      elsif L = "do-178c_level_d" then
         return Level_DO178C_D;
      else
         return Level_None;
      end if;
   end Parse_Safety_Level;

   function Operator_Symbol (Op : Binary_Operator) return String is
   begin
      case Op is
         when Op_Add => return "+";
         when Op_Sub => return "-";
         when Op_Mul => return "*";
         when Op_Div => return "/";
         when Op_Mod => return "%";
         when Op_Eq  => return "==";
         when Op_Neq => return "!=";
         when Op_Lt  => return "<";
         when Op_Leq => return "<=";
         when Op_Gt  => return ">";
         when Op_Geq => return ">=";
         when Op_And => return "&&";
         when Op_Or  => return "||";
         when Op_Bit_And => return "&";
         when Op_Bit_Or  => return "|";
         when Op_Bit_Xor => return "^";
         when Op_Shl => return "<<";
         when Op_Shr => return ">>";
         when Op_Assign => return "=";
      end case;
   end Operator_Symbol;
   
   function Operator_Symbol (Op : Unary_Operator) return String is
   begin
      case Op is
         when Op_Neg => return "-";
         when Op_Not => return "!";
         when Op_Bit_Not => return "~";
         when Op_Pre_Inc => return "++";
         when Op_Pre_Dec => return "--";
         when Op_Post_Inc => return "++";
         when Op_Post_Dec => return "--";
         when Op_Deref => return "*";
         when Op_Addr_Of => return "&";
      end case;
   end Operator_Symbol;
   
   function Primitive_Type_Name (T : IR_Primitive_Type) return String is
   begin
      case T is
         when Type_Void   => return "void";
         when Type_Bool   => return "bool";
         when Type_I8     => return "i8";
         when Type_I16    => return "i16";
         when Type_I32    => return "i32";
         when Type_I64    => return "i64";
         when Type_U8     => return "u8";
         when Type_U16    => return "u16";
         when Type_U32    => return "u32";
         when Type_U64    => return "u64";
         when Type_F32    => return "f32";
         when Type_F64    => return "f64";
         when Type_String => return "string";
         when Type_Char   => return "char";
      end case;
   end Primitive_Type_Name;
   
end IR.Types;
