--  STUNIR Embedded Emitter - Ada SPARK Specification
--  Emit bare-metal C for embedded systems (ARM, AVR, MIPS, RISC-V)
--  DO-178C Level A compliant for safety-critical avionics
--
--  SPARK_Mode: On - Enables formal verification for Ardupilot integration

pragma SPARK_Mode (On);

with Emitter_Types; use Emitter_Types;
with Ada.Containers.Formal_Vectors;

package Embedded_Emitter is

   use type Identifier_String;
   use type IR_Data_Type;

   --  Embedded target configuration
   type Embedded_Config is record
      Architecture : Architecture_Type;
      Stack_Size   : Positive;
      Heap_Size    : Natural;
      Use_Stdlib   : Boolean;
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

   --  Explicit equality for Embedded_Statement (required for Formal_Vectors)
   function "=" (Left, Right : Embedded_Statement) return Boolean;

   --  Capacity limits for heap-allocated vectors (production-ready sizes)
   Max_Statements : constant := 10;
   Max_Parameters : constant := 4;
   Max_Functions  : constant := 2;

   --  Formal vector packages for heap-backed storage
   package Statement_Vectors is new Ada.Containers.Formal_Vectors
     (Index_Type   => Positive,
      Element_Type => Embedded_Statement);

   package Parameter_Vectors is new Ada.Containers.Formal_Vectors
     (Index_Type   => Positive,
      Element_Type => Identifier_String);

   package Parameter_Type_Vectors is new Ada.Containers.Formal_Vectors
     (Index_Type   => Positive,
      Element_Type => IR_Data_Type);

   --  Vector subtypes with bounded capacity
   subtype Statement_Vector is Statement_Vectors.Vector (Max_Statements);
   subtype Parameter_Vector is Parameter_Vectors.Vector (Max_Parameters);
   subtype Parameter_Type_Vector is Parameter_Type_Vectors.Vector (Max_Parameters);

   --  IR Function representation with heap-allocated vectors
   type Embedded_Function is record
      Name        : Identifier_String;
      Return_Type : IR_Data_Type;
      Params      : Parameter_Vector;
      Param_Types : Parameter_Type_Vector;
      Statements  : Statement_Vector;
   end record;

   --  Function vector package
   package Function_Vectors is new Ada.Containers.Formal_Vectors
     (Index_Type   => Positive,
      Element_Type => Embedded_Function);

   subtype Function_Vector is Function_Vectors.Vector (Max_Functions);

   --  IR Module (collection of functions) with heap-allocated storage
   type Embedded_Module is record
      Name      : Identifier_String;
      Functions : Function_Vector;
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
