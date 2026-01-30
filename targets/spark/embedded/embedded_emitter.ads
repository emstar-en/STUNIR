--  STUNIR Embedded Emitter - Ada SPARK Specification
--  Emit bare-metal C for embedded systems (ARM, AVR, MIPS, RISC-V)
--  DO-178C Level A compliant for safety-critical avionics
--
--  SPARK_Mode: On - Enables formal verification for Ardupilot integration

pragma SPARK_Mode (On);

with Emitter_Types; use Emitter_Types;

package Embedded_Emitter is

   --  Embedded target configuration
   type Embedded_Config is record
      Architecture : Architecture_Type;
      Stack_Size   : Positive;
      Heap_Size    : Natural;  --  0 = no heap (bare metal)
      Use_Stdlib   : Boolean;  --  False for bare metal
      Optimize_Size : Boolean;
   end record;

   --  Default configuration for ARM Cortex-M (Ardupilot compatible)
   Default_Config : constant Embedded_Config := (
      Architecture  => Arch_ARM,
      Stack_Size    => 1024,
      Heap_Size     => 0,
      Use_Stdlib    => False,
      Optimize_Size => True
   );

   --  IR Statement representation for embedded code
   type Embedded_Statement is record
      Stmt_Type : IR_Statement_Type;
      Data_Type : IR_Data_Type;
      Target    : Identifier_String;
      Value     : Content_String;
      Left_Op   : Identifier_String;
      Right_Op  : Identifier_String;
   end record;

   --  IR Function representation
   Max_Statements : constant := 1000;
   Max_Parameters : constant := 16;

   type Statement_Array is array (1 .. Max_Statements) of Embedded_Statement;
   type Parameter_Array is array (1 .. Max_Parameters) of Identifier_String;
   type Parameter_Types is array (1 .. Max_Parameters) of IR_Data_Type;

   type Embedded_Function is record
      Name        : Identifier_String;
      Return_Type : IR_Data_Type;
      Params      : Parameter_Array;
      Param_Types : Parameter_Types;
      Param_Count : Natural range 0 .. Max_Parameters;
      Statements  : Statement_Array;
      Stmt_Count  : Natural range 0 .. Max_Statements;
   end record;

   --  IR Module (collection of functions)
   Max_Functions : constant := 100;
   type Function_Array is array (1 .. Max_Functions) of Embedded_Function;

   type Embedded_Module is record
      Name      : Identifier_String;
      Functions : Function_Array;
      Func_Count : Natural range 0 .. Max_Functions;
   end record;

   --  Main emission procedures with SPARK contracts
   procedure Emit_Module (
      Module    : in Embedded_Module;
      Config    : in Embedded_Config;
      Out_Path  : in Path_String;
      Result    : out Emitter_Result)
      with Pre => Identifier_Strings.Length (Module.Name) > 0
                  and then Path_Strings.Length (Out_Path) > 0,
           Post => Result.Status = Success or Result.Files_Count = 0;

   procedure Emit_Function (
      Func      : in Embedded_Function;
      Config    : in Embedded_Config;
      Content   : out Content_String;
      Status    : out Emitter_Status)
      with Pre => Identifier_Strings.Length (Func.Name) > 0,
           Post => (Status = Success) = (Content_Strings.Length (Content) > 0);

   procedure Emit_Statement (
      Stmt      : in Embedded_Statement;
      Config    : in Embedded_Config;
      Indent    : in Natural;
      Content   : out Content_String;
      Status    : out Emitter_Status);

   --  Header file generation
   procedure Generate_Header (
      Module    : in Embedded_Module;
      Config    : in Embedded_Config;
      Content   : out Content_String;
      Status    : out Emitter_Status)
      with Pre => Identifier_Strings.Length (Module.Name) > 0;

   --  Linker script generation for embedded targets
   procedure Generate_Linker_Script (
      Config    : in Embedded_Config;
      Content   : out Content_String;
      Status    : out Emitter_Status);

   --  Startup code generation
   procedure Generate_Startup (
      Config    : in Embedded_Config;
      Content   : out Content_String;
      Status    : out Emitter_Status);

   --  Architecture-specific type mapping
   function Get_C_Type (
      Data_Type : IR_Data_Type;
      Config    : Embedded_Config) return Type_Name_String;

   --  Memory alignment calculation
   function Calculate_Alignment (
      Data_Type : IR_Data_Type;
      Config    : Embedded_Config) return Positive
      with Post => Calculate_Alignment'Result in 1 .. 16;

end Embedded_Emitter;
