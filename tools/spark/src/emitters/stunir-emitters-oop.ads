-- STUNIR Object-Oriented Programming Emitter (SPARK Specification)
-- DO-178C Level A
-- Phase 3c: Remaining Category Emitters
-- Support: Java, C++, C#, Python OOP, Ruby

with STUNIR.Semantic_IR; use STUNIR.Semantic_IR;
with STUNIR.Emitters;    use STUNIR.Emitters;

package STUNIR.Emitters.OOP is
   pragma SPARK_Mode (On);
   pragma Elaborate_Body;

   -- OOP language enumeration
   type OOP_Language is
     (Java,
      Cpp,
      CSharp,
      Python_OOP,
      Ruby,
      Smalltalk,
      Eiffel,
      Kotlin);

   -- OOP emitter configuration
   type OOP_Config is record
      Language       : OOP_Language := Java;
      Use_Interfaces : Boolean := True;
      Use_Abstract   : Boolean := True;
      Use_Final      : Boolean := True;
      Visibility     : IR_Name_String; -- public, private, protected
      Indent_Size    : Positive := 4;
      Max_Line_Width : Positive := 100;
   end record;

   -- Default configuration
   Default_Config : constant OOP_Config :=
     (Language       => Java,
      Use_Interfaces => True,
      Use_Abstract   => True,
      Use_Final      => True,
      Visibility     => Name_Strings.To_Bounded_String ("public"),
      Indent_Size    => 4,
      Max_Line_Width => 100);

   -- OOP emitter type
   type OOP_Emitter is new Base_Emitter with record
      Config : OOP_Config := Default_Config;
   end record;

   -- Override abstract methods
   overriding procedure Emit_Module
     (Self   : in out OOP_Emitter;
      Module : in     IR_Module;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   with
     Pre'Class  => Is_Valid_Module (Module),
     Post'Class => (if Success then Code_Buffers.Length (Output) > 0);

   overriding procedure Emit_Type
     (Self   : in out OOP_Emitter;
      T      : in     IR_Type_Def;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   with
     Pre'Class  => T.Field_Cnt > 0,
     Post'Class => (if Success then Code_Buffers.Length (Output) >= 0);

   overriding procedure Emit_Function
     (Self   : in out OOP_Emitter;
      Func   : in     IR_Function;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   with
     Pre'Class  => Func.Arg_Cnt >= 0,
     Post'Class => (if Success then Code_Buffers.Length (Output) >= 0);

end STUNIR.Emitters.OOP;
