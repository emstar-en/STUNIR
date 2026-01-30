--  STUNIR Rust Emitter - Ada SPARK Specification
--  Emit Rust code with memory safety guarantees
--  DO-178C Level A compliant

pragma SPARK_Mode (On);

with Emitter_Types; use Emitter_Types;

package Rust_Emitter is

   type Rust_Edition is (Edition_2015, Edition_2018, Edition_2021);

   type Rust_Config is record
      Edition      : Rust_Edition;
      Use_Unsafe   : Boolean;
      No_Std       : Boolean;
   end record;

   Default_Config : constant Rust_Config := (
      Edition    => Edition_2021,
      Use_Unsafe => False,
      No_Std     => False
   );

   procedure Emit_Module (
      Module_Name : in Identifier_String;
      Content     : out Content_String;
      Config      : in Rust_Config;
      Status      : out Emitter_Status);

   function Map_Type_Rust (IR_Type : IR_Data_Type) return Type_Name_String;

end Rust_Emitter;
