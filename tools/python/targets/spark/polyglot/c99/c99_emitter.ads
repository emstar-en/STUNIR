--  STUNIR C99 Emitter - Ada SPARK Specification
--  Emit C99 compliant code with modern features
--  DO-178C Level A compliant

pragma SPARK_Mode (On);

with Emitter_Types; use Emitter_Types;

package C99_Emitter is

   type C99_Config is record
      Use_VLA         : Boolean;  --  Variable Length Arrays
      Use_Designated  : Boolean;  --  Designated initializers
      Max_Line_Width  : Positive;
   end record;

   Default_Config : constant C99_Config := (
      Use_VLA        => False,
      Use_Designated => True,
      Max_Line_Width => 100
   );

   procedure Emit_Header (
      Module_Name : in Identifier_String;
      Content     : out Content_String;
      Config      : in C99_Config;
      Status      : out Emitter_Status);

   procedure Emit_Source (
      Module_Name : in Identifier_String;
      Content     : out Content_String;
      Config      : in C99_Config;
      Status      : out Emitter_Status);

end C99_Emitter;
