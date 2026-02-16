--  STUNIR Lisp Base - Ada SPARK Implementation

pragma SPARK_Mode (On);

package body Lisp_Base is

   procedure Emit_Atom (
      Value   : in String;
      Content : out Content_String;
      Status  : out Emitter_Status)
   is
   begin
      Content := Content_Strings.Null_Bounded_String;
      if Value'Length > Max_Content_Length then
         Status := Error_Buffer_Overflow;
      else
         Content_Strings.Append (Content, Value);
         Status := Success;
      end if;
   end Emit_Atom;

   procedure Emit_List_Start (
      Content : in Out Content_String;
      Status  : out Emitter_Status)
   is
   begin
      if Content_Strings.Length (Content) + 1 > Max_Content_Length then
         Status := Error_Buffer_Overflow;
      else
         Content_Strings.Append (Content, "(");
         Status := Success;
      end if;
   end Emit_List_Start;

   procedure Emit_List_End (
      Content : in Out Content_String;
      Status  : out Emitter_Status)
   is
   begin
      if Content_Strings.Length (Content) + 1 > Max_Content_Length then
         Status := Error_Buffer_Overflow;
      else
         Content_Strings.Append (Content, ")");
         Status := Success;
      end if;
   end Emit_List_End;

   function Get_Dialect_Comment_Prefix (Dialect : Lisp_Dialect) return String is
   begin
      case Dialect is
         when Common_Lisp | Scheme | Racket | Guile =>
            return ";; ";
         when Clojure =>
            return ";; ";
         when Emacs_Lisp =>
            return ";;; ";
         when Hy =>
            return "; ";
         when Janet =>
            return "# ";
      end case;
   end Get_Dialect_Comment_Prefix;

end Lisp_Base;
