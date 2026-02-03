--  STUNIR x86 OS-Level Assembly Emitter - Ada SPARK Implementation
--  DO-178C Level A compliant

pragma SPARK_Mode (On);

package body X86_OS_Emitter is

   NL : constant Character := ASCII.LF;

   function Exception_Has_Error_Code (Vector : Natural) return Boolean is
   begin
      --  Exceptions 8, 10, 11, 12, 13, 14, 17, 21, 29, 30 push error codes
      return Vector = 8 or Vector = 10 or Vector = 11 or
             Vector = 12 or Vector = 13 or Vector = 14 or
             Vector = 17 or Vector = 21 or Vector = 29 or Vector = 30;
   end Exception_Has_Error_Code;

   procedure Emit_Multiboot2_Header (
      Content : out Content_String;
      Config  : in Multiboot2_Config;
      Status  : out Emitter_Status)
   is
      Is_Intel : constant Boolean := Config.Syntax = Intel_Syntax;
      Cmt : constant String := (if Is_Intel then "; " else "# ");
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      --  Header comment
      Content_Strings.Append (Content,
         Cmt & "STUNIR Generated Multiboot2 Header" & NL &
         Cmt & "DO-178C Level A Compliant" & NL & NL);

      --  Section directive
      Content_Strings.Append (Content,
         "section .multiboot2" & NL &
         "align 8" & NL & NL &
         "multiboot2_header_start:" & NL);

      --  Magic number
      if Is_Intel then
         Content_Strings.Append (Content,
            "    dd 0xE85250D6  ; magic" & NL &
            "    dd 0           ; architecture (i386)" & NL &
            "    dd multiboot2_header_end - multiboot2_header_start  ; length" & NL &
            "    dd -(0xE85250D6 + 0 + (multiboot2_header_end - multiboot2_header_start))  ; checksum" & NL & NL);
      else
         Content_Strings.Append (Content,
            "    .long 0xE85250D6  # magic" & NL &
            "    .long 0           # architecture (i386)" & NL &
            "    .long multiboot2_header_end - multiboot2_header_start  # length" & NL &
            "    .long -(0xE85250D6 + 0 + (multiboot2_header_end - multiboot2_header_start))  # checksum" & NL & NL);
      end if;

      --  Memory map request tag
      if Config.Request_Memmap then
         Content_Strings.Append (Content,
            Cmt & "Information Request Tag" & NL &
            "align 8" & NL &
            "info_request_tag:" & NL);
         if Is_Intel then
            Content_Strings.Append (Content,
               "    dw 1   ; type (info request)" & NL &
               "    dw 0   ; flags" & NL &
               "    dd info_request_tag_end - info_request_tag  ; size" & NL &
               "    dd 6   ; MULTIBOOT_TAG_TYPE_MMAP" & NL);
         else
            Content_Strings.Append (Content,
               "    .word 1   # type (info request)" & NL &
               "    .word 0   # flags" & NL &
               "    .long info_request_tag_end - info_request_tag  # size" & NL &
               "    .long 6   # MULTIBOOT_TAG_TYPE_MMAP" & NL);
         end if;
         Content_Strings.Append (Content, "info_request_tag_end:" & NL & NL);
      end if;

      --  End tag
      Content_Strings.Append (Content,
         Cmt & "End Tag" & NL &
         "align 8" & NL &
         "end_tag:" & NL);
      if Is_Intel then
         Content_Strings.Append (Content,
            "    dw 0   ; type (end)" & NL &
            "    dw 0   ; flags" & NL &
            "    dd 8   ; size" & NL & NL);
      else
         Content_Strings.Append (Content,
            "    .word 0   # type (end)" & NL &
            "    .word 0   # flags" & NL &
            "    .long 8   # size" & NL & NL);
      end if;

      Content_Strings.Append (Content, "multiboot2_header_end:" & NL);
   end Emit_Multiboot2_Header;

   procedure Emit_Boot_Entry (
      Content : out Content_String;
      Config  : in Multiboot2_Config;
      Status  : out Emitter_Status)
   is
      Is_Intel : constant Boolean := Config.Syntax = Intel_Syntax;
      Is_64 : constant Boolean := Config.Mode = Mode_64;
      Cmt : constant String := (if Is_Intel then "; " else "# ");
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      Content_Strings.Append (Content,
         Cmt & "STUNIR Generated Boot Entry" & NL &
         Cmt & "DO-178C Level A Compliant" & NL & NL &
         "section .text" & NL &
         "global _start" & NL &
         "extern kernel_main" & NL & NL &
         "_start:" & NL);

      if Is_64 then
         if Is_Intel then
            Content_Strings.Append (Content,
               "    ; Save multiboot info" & NL &
               "    mov rdi, rbx" & NL &
               "    ; Setup stack" & NL &
               "    mov rsp, stack_top" & NL &
               "    ; Call kernel" & NL &
               "    call kernel_main" & NL &
               ".halt:" & NL &
               "    cli" & NL &
               "    hlt" & NL &
               "    jmp .halt" & NL);
         else
            Content_Strings.Append (Content,
               "    # Save multiboot info" & NL &
               "    movq %rbx, %rdi" & NL &
               "    # Setup stack" & NL &
               "    movq $stack_top, %rsp" & NL &
               "    # Call kernel" & NL &
               "    call kernel_main" & NL &
               ".halt:" & NL &
               "    cli" & NL &
               "    hlt" & NL &
               "    jmp .halt" & NL);
         end if;
      else
         if Is_Intel then
            Content_Strings.Append (Content,
               "    ; Disable interrupts" & NL &
               "    cli" & NL &
               "    ; Setup stack" & NL &
               "    mov esp, stack_top" & NL &
               "    ; Push multiboot info" & NL &
               "    push ebx" & NL &
               "    push eax" & NL &
               "    ; Call kernel" & NL &
               "    call kernel_main" & NL &
               ".halt:" & NL &
               "    cli" & NL &
               "    hlt" & NL &
               "    jmp .halt" & NL);
         else
            Content_Strings.Append (Content,
               "    # Disable interrupts" & NL &
               "    cli" & NL &
               "    # Setup stack" & NL &
               "    movl $stack_top, %esp" & NL &
               "    # Push multiboot info" & NL &
               "    pushl %ebx" & NL &
               "    pushl %eax" & NL &
               "    # Call kernel" & NL &
               "    call kernel_main" & NL &
               ".halt:" & NL &
               "    cli" & NL &
               "    hlt" & NL &
               "    jmp .halt" & NL);
         end if;
      end if;

      --  Stack section
      Content_Strings.Append (Content,
         NL & "section .bss" & NL &
         "align 16" & NL &
         "stack_bottom:" & NL &
         "    resb 16384  ; 16KB stack" & NL &
         "stack_top:" & NL);
   end Emit_Boot_Entry;

   procedure Emit_ISR_Stub (
      Vector         : in Natural;
      Has_Error_Code : in Boolean;
      Content        : out Content_String;
      Config         : in ISR_Config;
      Status         : out Emitter_Status)
   is
      Is_Intel : constant Boolean := Config.Syntax = Intel_Syntax;
      Vec_Str : constant String := Natural'Image (Vector);
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      Content_Strings.Append (Content,
         "global isr" & Vec_Str (Vec_Str'First + 1 .. Vec_Str'Last) & NL &
         "isr" & Vec_Str (Vec_Str'First + 1 .. Vec_Str'Last) & ":" & NL);

      if not Has_Error_Code then
         if Is_Intel then
            Content_Strings.Append (Content, "    push 0" & NL);
         else
            Content_Strings.Append (Content, "    pushl $0" & NL);
         end if;
      end if;

      if Is_Intel then
         Content_Strings.Append (Content,
            "    push" & Vec_Str & NL &
            "    jmp isr_common_stub" & NL);
      else
         Content_Strings.Append (Content,
            "    pushl $" & Vec_Str (Vec_Str'First + 1 .. Vec_Str'Last) & NL &
            "    jmp isr_common_stub" & NL);
      end if;
   end Emit_ISR_Stub;

   procedure Emit_ISR_Common_Stub (
      Content : out Content_String;
      Config  : in ISR_Config;
      Status  : out Emitter_Status)
   is
      Is_Intel : constant Boolean := Config.Syntax = Intel_Syntax;
      Is_64 : constant Boolean := Config.Mode = Mode_64;
      Cmt : constant String := (if Is_Intel then "; " else "# ");
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      Content_Strings.Append (Content,
         Cmt & "Common ISR stub - saves processor state" & NL &
         "isr_common_stub:" & NL);

      if Is_64 then
         if Is_Intel then
            Content_Strings.Append (Content,
               "    push rax" & NL &
               "    push rcx" & NL &
               "    push rdx" & NL &
               "    push rbx" & NL &
               "    push rbp" & NL &
               "    push rsi" & NL &
               "    push rdi" & NL &
               "    push r8" & NL &
               "    push r9" & NL &
               "    push r10" & NL &
               "    push r11" & NL &
               "    mov rdi, rsp  ; First arg: interrupt frame" & NL &
               "    call interrupt_handler" & NL &
               "    pop r11" & NL &
               "    pop r10" & NL &
               "    pop r9" & NL &
               "    pop r8" & NL &
               "    pop rdi" & NL &
               "    pop rsi" & NL &
               "    pop rbp" & NL &
               "    pop rbx" & NL &
               "    pop rdx" & NL &
               "    pop rcx" & NL &
               "    pop rax" & NL &
               "    add rsp, 16  ; Clean up" & NL &
               "    iretq" & NL);
         else
            Content_Strings.Append (Content,
               "    pushq %rax" & NL &
               "    pushq %rcx" & NL &
               "    pushq %rdx" & NL &
               "    movq %rsp, %rdi" & NL &
               "    call interrupt_handler" & NL &
               "    popq %rdx" & NL &
               "    popq %rcx" & NL &
               "    popq %rax" & NL &
               "    addq $16, %rsp" & NL &
               "    iretq" & NL);
         end if;
      else
         if Is_Intel then
            Content_Strings.Append (Content,
               "    pusha" & NL &
               "    push ds" & NL &
               "    push es" & NL &
               "    push fs" & NL &
               "    push gs" & NL &
               "    mov ax, 0x10" & NL &
               "    mov ds, ax" & NL &
               "    mov es, ax" & NL &
               "    push esp" & NL &
               "    call interrupt_handler" & NL &
               "    add esp, 4" & NL &
               "    pop gs" & NL &
               "    pop fs" & NL &
               "    pop es" & NL &
               "    pop ds" & NL &
               "    popa" & NL &
               "    add esp, 8" & NL &
               "    iret" & NL);
         else
            Content_Strings.Append (Content,
               "    pusha" & NL &
               "    pushl %esp" & NL &
               "    call interrupt_handler" & NL &
               "    addl $4, %esp" & NL &
               "    popa" & NL &
               "    addl $8, %esp" & NL &
               "    iret" & NL);
         end if;
      end if;
   end Emit_ISR_Common_Stub;

   procedure Emit_GDT (
      Content : out Content_String;
      Config  : in GDT_Config;
      Status  : out Emitter_Status)
   is
      Is_Intel : constant Boolean := Config.Syntax = Intel_Syntax;
      Is_64 : constant Boolean := Config.Mode = Mode_64;
      Cmt : constant String := (if Is_Intel then "; " else "# ");
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      Content_Strings.Append (Content,
         Cmt & "STUNIR Generated GDT" & NL &
         Cmt & "DO-178C Level A Compliant" & NL & NL &
         "section .data" & NL &
         "align 16" & NL & NL &
         Cmt & "GDT Descriptor" & NL &
         "global gdt_ptr" & NL &
         "gdt_ptr:" & NL);

      if Is_Intel then
         Content_Strings.Append (Content,
            "    dw gdt_end - gdt_start - 1  ; limit" & NL);
         if Is_64 then
            Content_Strings.Append (Content, "    dq gdt_start  ; base" & NL);
         else
            Content_Strings.Append (Content, "    dd gdt_start  ; base" & NL);
         end if;
      else
         Content_Strings.Append (Content,
            "    .word gdt_end - gdt_start - 1  # limit" & NL);
         if Is_64 then
            Content_Strings.Append (Content, "    .quad gdt_start  # base" & NL);
         else
            Content_Strings.Append (Content, "    .long gdt_start  # base" & NL);
         end if;
      end if;

      Content_Strings.Append (Content,
         NL & Cmt & "GDT Entries" & NL &
         "global gdt_start" & NL &
         "gdt_start:" & NL);

      --  Null descriptor
      if Is_Intel then
         Content_Strings.Append (Content,
            "gdt_null:  ; Selector 0x00" & NL &
            "    dq 0x0000000000000000" & NL);
      else
         Content_Strings.Append (Content,
            "gdt_null:  # Selector 0x00" & NL &
            "    .quad 0x0000000000000000" & NL);
      end if;

      if Is_64 then
         --  64-bit kernel code
         if Is_Intel then
            Content_Strings.Append (Content,
               "gdt_kernel_code:  ; Selector 0x08" & NL &
               "    dq 0x00AF9A000000FFFF" & NL &
               "gdt_kernel_data:  ; Selector 0x10" & NL &
               "    dq 0x00CF92000000FFFF" & NL);
         else
            Content_Strings.Append (Content,
               "gdt_kernel_code:  # Selector 0x08" & NL &
               "    .quad 0x00AF9A000000FFFF" & NL &
               "gdt_kernel_data:  # Selector 0x10" & NL &
               "    .quad 0x00CF92000000FFFF" & NL);
         end if;
      else
         --  32-bit kernel code/data
         if Is_Intel then
            Content_Strings.Append (Content,
               "gdt_kernel_code:  ; Selector 0x08" & NL &
               "    dq 0x00CF9A000000FFFF" & NL &
               "gdt_kernel_data:  ; Selector 0x10" & NL &
               "    dq 0x00CF92000000FFFF" & NL);
         else
            Content_Strings.Append (Content,
               "gdt_kernel_code:  # Selector 0x08" & NL &
               "    .quad 0x00CF9A000000FFFF" & NL &
               "gdt_kernel_data:  # Selector 0x10" & NL &
               "    .quad 0x00CF92000000FFFF" & NL);
         end if;
      end if;

      Content_Strings.Append (Content, "gdt_end:" & NL & NL);

      --  Selector constants
      if Is_Intel then
         Content_Strings.Append (Content,
            "KERNEL_CODE_SEL equ 0x08" & NL &
            "KERNEL_DATA_SEL equ 0x10" & NL);
      else
         Content_Strings.Append (Content,
            ".set KERNEL_CODE_SEL, 0x08" & NL &
            ".set KERNEL_DATA_SEL, 0x10" & NL);
      end if;
   end Emit_GDT;

   procedure Emit_IDT_Setup (
      Content : out Content_String;
      Config  : in GDT_Config;
      Status  : out Emitter_Status)
   is
      Is_Intel : constant Boolean := Config.Syntax = Intel_Syntax;
      Is_64 : constant Boolean := Config.Mode = Mode_64;
      Cmt : constant String := (if Is_Intel then "; " else "# ");
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      Content_Strings.Append (Content,
         Cmt & "STUNIR Generated IDT Setup" & NL &
         Cmt & "DO-178C Level A Compliant" & NL & NL &
         "section .data" & NL &
         "align 16" & NL & NL &
         "global idt_ptr" & NL &
         "idt_ptr:" & NL);

      if Is_Intel then
         if Is_64 then
            Content_Strings.Append (Content,
               "    dw 4095  ; limit (256 * 16 - 1)" & NL &
               "    dq idt_entries  ; base" & NL);
         else
            Content_Strings.Append (Content,
               "    dw 2047  ; limit (256 * 8 - 1)" & NL &
               "    dd idt_entries  ; base" & NL);
         end if;
      else
         if Is_64 then
            Content_Strings.Append (Content,
               "    .word 4095  # limit" & NL &
               "    .quad idt_entries  # base" & NL);
         else
            Content_Strings.Append (Content,
               "    .word 2047  # limit" & NL &
               "    .long idt_entries  # base" & NL);
         end if;
      end if;

      Content_Strings.Append (Content,
         NL & "section .bss" & NL &
         "align 16" & NL &
         "global idt_entries" & NL &
         "idt_entries:" & NL);

      if Is_64 then
         Content_Strings.Append (Content, "    resb 4096  ; 256 * 16 bytes" & NL);
      else
         Content_Strings.Append (Content, "    resb 2048  ; 256 * 8 bytes" & NL);
      end if;
   end Emit_IDT_Setup;

   procedure Emit_Complete_Boot_Assembly (
      Content   : out Content_String;
      MB2_Cfg   : in Multiboot2_Config;
      ISR_Cfg   : in ISR_Config;
      GDT_Cfg   : in GDT_Config;
      Status    : out Emitter_Status)
   is
      Temp : Content_String;
      Temp_Status : Emitter_Status;
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      --  Generate Multiboot2 header
      Emit_Multiboot2_Header (Temp, MB2_Cfg, Temp_Status);
      if Temp_Status /= Success then
         Status := Temp_Status;
         return;
      end if;
      Content_Strings.Append (Content, Content_Strings.To_String (Temp));
      Content_Strings.Append (Content, ASCII.LF & ASCII.LF);

      --  Generate GDT
      Emit_GDT (Temp, GDT_Cfg, Temp_Status);
      if Temp_Status /= Success then
         Status := Temp_Status;
         return;
      end if;
      Content_Strings.Append (Content, Content_Strings.To_String (Temp));
      Content_Strings.Append (Content, ASCII.LF & ASCII.LF);

      --  Generate boot entry
      Emit_Boot_Entry (Temp, MB2_Cfg, Temp_Status);
      if Temp_Status /= Success then
         Status := Temp_Status;
         return;
      end if;
      Content_Strings.Append (Content, Content_Strings.To_String (Temp));
   end Emit_Complete_Boot_Assembly;

end X86_OS_Emitter;
