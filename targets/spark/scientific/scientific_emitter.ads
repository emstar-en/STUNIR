--  STUNIR Scientific Emitter - Ada SPARK Specification
--  Fortran and Pascal emitters
--  DO-178C Level A compliant

pragma SPARK_Mode (On);

with Emitter_Types; use Emitter_Types;

package Scientific_Emitter is

   type Scientific_Language is (Fortran_90, Fortran_2008, Pascal, Delphi);

   type Scientific_Config is record
      Language : Scientific_Language;
   end record;

   Default_Config : constant Scientific_Config := (Language => Fortran_90);

   procedure Emit_Module (
      Module_Name : in Identifier_String;
      Content     : out Content_String;
      Config      : in Scientific_Config;
      Status      : out Emitter_Status);

   procedure Emit_Function (
      Func_Name : in Identifier_String;
      Params    : in String;
      Body_Code : in String;
      Content   : out Content_String;
      Config    : in Scientific_Config;
      Status    : out Emitter_Status);

end Scientific_Emitter;
