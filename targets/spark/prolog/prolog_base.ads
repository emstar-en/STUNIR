--  STUNIR Prolog Base - Ada SPARK Specification
--  Common types for all Prolog dialect emitters
--  DO-178C Level A compliant

pragma SPARK_Mode (On);

with Emitter_Types; use Emitter_Types;

package Prolog_Base is

   type Prolog_Dialect is (
      SWI_Prolog,
      GNU_Prolog,
      SICStus,
      YAP,
      XSB,
      ECLiPSe,
      Tau_Prolog,
      Mercury,
      Datalog
   );

   type Prolog_Config is record
      Dialect : Prolog_Dialect;
   end record;

   Default_Config : constant Prolog_Config := (Dialect => SWI_Prolog);

   procedure Emit_Module (
      Module_Name : in Identifier_String;
      Content     : out Content_String;
      Config      : in Prolog_Config;
      Status      : out Emitter_Status);

   procedure Emit_Clause (
      Head : in String;
      Body : in String;
      Content : out Content_String;
      Status : out Emitter_Status);

   procedure Emit_Fact (
      Predicate : in Identifier_String;
      Args      : in String;
      Content   : out Content_String;
      Status    : out Emitter_Status);

end Prolog_Base;
