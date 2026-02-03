--  STUNIR x86 OS-Level Assembly Emitter - Ada SPARK Specification
--  Supports Multiboot2, ISR stubs, GDT/IDT generation
--  DO-178C Level A compliant

pragma SPARK_Mode (On);

with Emitter_Types; use Emitter_Types;

package X86_OS_Emitter is

   --  Multiboot2 Constants
   MULTIBOOT2_HEADER_MAGIC : constant := 16#E85250D6#;
   MULTIBOOT2_BOOTLOADER_MAGIC : constant := 16#36D76289#;
   MULTIBOOT2_ARCH_I386 : constant := 0;

   --  Multiboot2 Tag Types
   type MB2_Tag_Type is (
      Tag_End,
      Tag_Info_Request,
      Tag_Address,
      Tag_Entry_Address,
      Tag_Console_Flags,
      Tag_Framebuffer,
      Tag_Module_Align
   );
   
   for MB2_Tag_Type use (
      Tag_End => 0,
      Tag_Info_Request => 1,
      Tag_Address => 2,
      Tag_Entry_Address => 3,
      Tag_Console_Flags => 4,
      Tag_Framebuffer => 5,
      Tag_Module_Align => 6
   );

   --  GDT Access Flags
   GDT_ACCESS_PRESENT    : constant := 16#80#;
   GDT_ACCESS_DESCRIPTOR : constant := 16#10#;
   GDT_ACCESS_EXECUTABLE : constant := 16#08#;
   GDT_ACCESS_RW         : constant := 16#02#;
   GDT_ACCESS_DPL3       : constant := 16#60#;
   
   --  GDT Flags
   GDT_FLAG_GRANULARITY : constant := 16#08#;
   GDT_FLAG_SIZE_32     : constant := 16#04#;
   GDT_FLAG_LONG_MODE   : constant := 16#02#;

   --  IDT Gate Types
   IDT_INTERRUPT_GATE_32 : constant := 16#8E#;
   IDT_TRAP_GATE_32      : constant := 16#8F#;
   IDT_INTERRUPT_GATE_64 : constant := 16#8E#;
   IDT_TRAP_GATE_64      : constant := 16#8F#;

   --  Exceptions that push error codes
   function Exception_Has_Error_Code (Vector : Natural) return Boolean
      with Pre => Vector <= 31;

   --  Configuration types
   type X86_OS_Mode is (Mode_32, Mode_64);
   type X86_Syntax is (Intel_Syntax, AT_T_Syntax);

   type Multiboot2_Config is record
      Mode              : X86_OS_Mode;
      Syntax            : X86_Syntax;
      Request_Memmap    : Boolean;
      Request_Framebuffer : Boolean;
      FB_Width          : Positive;
      FB_Height         : Positive;
      FB_Depth          : Positive;
   end record;

   Default_MB2_Config : constant Multiboot2_Config := (
      Mode              => Mode_32,
      Syntax            => Intel_Syntax,
      Request_Memmap    => True,
      Request_Framebuffer => False,
      FB_Width          => 800,
      FB_Height         => 600,
      FB_Depth          => 32
   );

   type ISR_Config is record
      Mode           : X86_OS_Mode;
      Syntax         : X86_Syntax;
      Exception_Count : Positive;
      IRQ_Base       : Natural;
      IRQ_Count      : Positive;
      Save_All_Regs  : Boolean;
   end record;

   Default_ISR_Config : constant ISR_Config := (
      Mode           => Mode_64,
      Syntax         => Intel_Syntax,
      Exception_Count => 32,
      IRQ_Base       => 32,
      IRQ_Count      => 16,
      Save_All_Regs  => True
   );

   type GDT_Config is record
      Mode   : X86_OS_Mode;
      Syntax : X86_Syntax;
   end record;

   Default_GDT_Config : constant GDT_Config := (
      Mode   => Mode_64,
      Syntax => Intel_Syntax
   );

   --  Multiboot2 Header Generation
   procedure Emit_Multiboot2_Header (
      Content : out Content_String;
      Config  : in Multiboot2_Config;
      Status  : out Emitter_Status);

   procedure Emit_Boot_Entry (
      Content : out Content_String;
      Config  : in Multiboot2_Config;
      Status  : out Emitter_Status);

   --  ISR Stub Generation
   procedure Emit_ISR_Stub (
      Vector         : in Natural;
      Has_Error_Code : in Boolean;
      Content        : out Content_String;
      Config         : in ISR_Config;
      Status         : out Emitter_Status)
      with Pre => Vector <= 255;

   procedure Emit_ISR_Common_Stub (
      Content : out Content_String;
      Config  : in ISR_Config;
      Status  : out Emitter_Status);

   --  GDT/IDT Generation
   procedure Emit_GDT (
      Content : out Content_String;
      Config  : in GDT_Config;
      Status  : out Emitter_Status);

   procedure Emit_IDT_Setup (
      Content : out Content_String;
      Config  : in GDT_Config;
      Status  : out Emitter_Status);

   --  Helper: Generate complete OS boot assembly
   procedure Emit_Complete_Boot_Assembly (
      Content   : out Content_String;
      MB2_Cfg   : in Multiboot2_Config;
      ISR_Cfg   : in ISR_Config;
      GDT_Cfg   : in GDT_Config;
      Status    : out Emitter_Status);

end X86_OS_Emitter;
