--  STUNIR Embedded Emitter Main Program
--  Ada SPARK implementation for DO-178C compliance
--  Entry point for standalone embedded code generation

with Ada.Text_IO;
with Ada.Command_Line;
with Emitter_Types; use Emitter_Types;
with Embedded_Emitter; use Embedded_Emitter;

procedure Embedded_Emitter_Main is
   Config : Embedded_Config := Default_Config;
   Result : Emitter_Result;

   Test_Module : aliased Embedded_Module;
   Test_Func   : aliased Embedded_Function;

begin
   Ada.Text_IO.Put_Line ("STUNIR Embedded Emitter - Ada SPARK");
   Ada.Text_IO.Put_Line ("DO-178C Level A Compliant");
   Ada.Text_IO.Put_Line ("=====================================");
   
   --  Check command line arguments
   if Ada.Command_Line.Argument_Count < 1 then
      Ada.Text_IO.Put_Line ("Usage: embedded_emitter_main <ir_file> [--arch=arm|avr|mips]");
      Ada.Text_IO.Put_Line ("");
      Ada.Text_IO.Put_Line ("Options:");
      Ada.Text_IO.Put_Line ("  --arch=arm     ARM Cortex-M (default)");
      Ada.Text_IO.Put_Line ("  --arch=avr     AVR microcontrollers");
      Ada.Text_IO.Put_Line ("  --arch=mips    MIPS processors");
      Ada.Text_IO.Put_Line ("  --arch=riscv   RISC-V processors");
      return;
   end if;
   
   --  Parse architecture option if provided
   if Ada.Command_Line.Argument_Count >= 2 then
      declare
         Arg : constant String := Ada.Command_Line.Argument (2);
      begin
         if Arg = "--arch=arm" then
            Config.Architecture := Arch_ARM;
         elsif Arg = "--arch=avr" then
            Config.Architecture := Arch_AVR;
         elsif Arg = "--arch=mips" then
            Config.Architecture := Arch_MIPS;
         elsif Arg = "--arch=riscv" then
            Config.Architecture := Arch_RISCV;
         end if;
      end;
   end if;
   
   Ada.Text_IO.Put_Line ("Architecture: " & Architecture_Type'Image (Config.Architecture));
   Ada.Text_IO.Put_Line ("Stack Size: " & Positive'Image (Config.Stack_Size));
   Ada.Text_IO.Put_Line ("");
   
   --  Initialize test module
   Test_Module.Name := Identifier_Strings.To_Bounded_String ("test_module");
   
   --  Initialize test function
   Test_Func.Name := Identifier_Strings.To_Bounded_String ("main");
   Test_Func.Return_Type := Type_I32;
   
   --  Add statements using vector Append
   Statement_Vectors.Append (Test_Func.Statements, (
      Stmt_Type => Stmt_Var_Decl,
      Data_Type => Type_I32,
      Target    => Identifier_Strings.To_Bounded_String ("result"),
      Value     => Content_Strings.To_Bounded_String ("0"),
      Left_Op   => Identifier_Strings.Null_Bounded_String,
      Right_Op  => Identifier_Strings.Null_Bounded_String
   ));
   
   Statement_Vectors.Append (Test_Func.Statements, (
      Stmt_Type => Stmt_Return,
      Data_Type => Type_I32,
      Target    => Identifier_Strings.Null_Bounded_String,
      Value     => Content_Strings.To_Bounded_String ("result"),
      Left_Op   => Identifier_Strings.Null_Bounded_String,
      Right_Op  => Identifier_Strings.Null_Bounded_String
   ));
   
   --  Add function to module using vector Append
   Function_Vectors.Append (Test_Module.Functions, Test_Func);
   
   --  Emit the module
   Emit_Module (
      Module   => Test_Module,
      Config   => Config,
      Out_Path => Path_Strings.To_Bounded_String ("./output"),
      Result   => Result
   );
   
   --  Report results
   if Result.Status = Success then
      Ada.Text_IO.Put_Line ("Emission successful!");
      Ada.Text_IO.Put_Line ("Files generated: " & Natural'Image (Result.Files_Count));
      Ada.Text_IO.Put_Line ("Total size: " & Natural'Image (Result.Total_Size) & " bytes");
   else
      Ada.Text_IO.Put_Line ("Emission failed: " & Emitter_Status'Image (Result.Status));
   end if;
   
end Embedded_Emitter_Main;
