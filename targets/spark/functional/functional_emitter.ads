--  STUNIR Functional Language Emitter - Ada SPARK Specification
--  Haskell, OCaml, F# emitters
--  DO-178C Level A compliant

pragma SPARK_Mode (On);

with Emitter_Types; use Emitter_Types;

package Functional_Emitter is

   type Functional_Language is (Haskell, OCaml, FSharp);

   type Functional_Config is record
      Language : Functional_Language;
   end record;

   Default_Config : constant Functional_Config := (Language => Haskell);

   procedure Emit_Module (
      Module_Name : in Identifier_String;
      Content     : out Content_String;
      Config      : in Functional_Config;
      Status      : out Emitter_Status);

   procedure Emit_Function (
      Func_Name : in Identifier_String;
      Signature : in String;
      Body_Code : in String;
      Content   : out Content_String;
      Config    : in Functional_Config;
      Status    : out Emitter_Status);

   procedure Emit_Type (
      Type_Name : in Identifier_String;
      Type_Def  : in String;
      Content   : out Content_String;
      Config    : in Functional_Config;
      Status    : out Emitter_Status);

end Functional_Emitter;
