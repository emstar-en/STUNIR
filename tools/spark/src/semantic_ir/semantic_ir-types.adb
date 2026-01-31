-- STUNIR Semantic IR Types Package Body
-- DO-178C Level A Compliant
-- SPARK 2014 Mode

pragma SPARK_Mode (On);

package body Semantic_IR.Types is

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
   
end Semantic_IR.Types;
