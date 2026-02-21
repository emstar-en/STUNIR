--  STUNIR C89 Emitter - Ada SPARK Specification
--  Emit strictly ANSI C89 compliant code
--  DO-178C Level A compliant

pragma SPARK_Mode (On);

with Emitter_Types; use Emitter_Types;

package C89_Emitter is

   type C89_Config is record
      Use_KnR_Style  : Boolean;
      Max_Line_Width : Positive;
      Use_Trigraphs  : Boolean;
   end record;

   Default_Config : constant C89_Config := (
      Use_KnR_Style  => False,
      Max_Line_Width => 80,
      Use_Trigraphs  => False
   );

   procedure Emit_Header (
      Module_Name : in Identifier_String;
      Content     : out Content_String;
      Config      : in C89_Config;
      Status      : out Emitter_Status);

   procedure Emit_Source (
      Module_Name : in Identifier_String;
      Content     : out Content_String;
      Config      : in C89_Config;
      Status      : out Emitter_Status);

   function Map_Type_C89 (IR_Type : IR_Data_Type) return Type_Name_String;

end C89_Emitter;
