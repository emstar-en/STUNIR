--  STUNIR Mobile Emitter - Ada SPARK Specification
--  iOS/Android code emitter
--  DO-178C Level A compliant

pragma SPARK_Mode (On);

with Emitter_Types; use Emitter_Types;

package Mobile_Emitter is

   type Mobile_Platform is (iOS_Swift, Android_Kotlin, React_Native);

   type Mobile_Config is record
      Platform : Mobile_Platform;
   end record;

   Default_Config : constant Mobile_Config := (Platform => iOS_Swift);

   procedure Emit_View (
      View_Name : in Identifier_String;
      Content   : out Content_String;
      Config    : in Mobile_Config;
      Status    : out Emitter_Status);

   procedure Emit_Function (
      Func_Name : in Identifier_String;
      Params    : in String;
      Body_Code : in String;
      Content   : out Content_String;
      Config    : in Mobile_Config;
      Status    : out Emitter_Status);

end Mobile_Emitter;
