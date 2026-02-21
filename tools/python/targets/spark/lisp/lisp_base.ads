--  STUNIR Lisp Base - Ada SPARK Specification
--  Common types for all Lisp dialect emitters
--  DO-178C Level A compliant

pragma SPARK_Mode (On);

with Emitter_Types; use Emitter_Types;

package Lisp_Base is

   type Lisp_Dialect is (
      Common_Lisp,
      Scheme,
      Clojure,
      Racket,
      Emacs_Lisp,
      Guile,
      Hy,
      Janet
   );

   type Lisp_Config is record
      Dialect        : Lisp_Dialect;
      Indent_Width   : Positive;
      Max_Line_Width : Positive;
   end record;

   Default_Config : constant Lisp_Config := (
      Dialect        => Common_Lisp,
      Indent_Width   => 2,
      Max_Line_Width => 80
   );

   --  S-expression emission
   procedure Emit_Atom (
      Value   : in String;
      Content : out Content_String;
      Status  : out Emitter_Status);

   procedure Emit_List_Start (
      Content : in out Content_String;
      Status  : out Emitter_Status);

   procedure Emit_List_End (
      Content : in Out Content_String;
      Status  : out Emitter_Status);

   function Get_Dialect_Comment_Prefix (Dialect : Lisp_Dialect) return String;

end Lisp_Base;
