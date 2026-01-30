--  STUNIR BEAM Emitter - Ada SPARK Specification
--  Erlang/Elixir BEAM VM target emitter
--  DO-178C Level A compliant

pragma SPARK_Mode (On);

with Emitter_Types; use Emitter_Types;

package BEAM_Emitter is

   type BEAM_Language is (Erlang, Elixir);

   type BEAM_Config is record
      Language : BEAM_Language;
   end record;

   Default_Config : constant BEAM_Config := (Language => Erlang);

   procedure Emit_Module (
      Module_Name : in Identifier_String;
      Content     : out Content_String;
      Config      : in BEAM_Config;
      Status      : out Emitter_Status);

   procedure Emit_Function (
      Func_Name : in Identifier_String;
      Params    : in String;
      Body_Code : in String;
      Config    : in BEAM_Config;
      Content   : out Content_String;
      Status    : out Emitter_Status);

end BEAM_Emitter;
