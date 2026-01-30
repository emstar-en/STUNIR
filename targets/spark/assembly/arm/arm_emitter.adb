--  STUNIR ARM Assembly Emitter - Ada SPARK Implementation

pragma SPARK_Mode (On);

package body ARM_Emitter is

   New_Line : constant Character := ASCII.LF;

   procedure Emit_Prologue (
      Func_Name : in Identifier_String;
      Content   : out Content_String;
      Config    : in ARM_Config;
      Status    : out Emitter_Status)
   is
      Name : constant String := Identifier_Strings.To_String (Func_Name);
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      case Config.Mode is
         when ARM32 | Thumb =>
            Content_Strings.Append (Content,
               "; STUNIR Generated ARM Assembly" & New_Line &
               ".global " & Name & New_Line &
               ".type " & Name & ", %function" & New_Line &
               Name & ":" & New_Line &
               "    push {r4-r11, lr}" & New_Line &
               "    sub sp, sp, #16" & New_Line);
         when ARM64_AArch64 =>
            Content_Strings.Append (Content,
               "// STUNIR Generated ARM64 Assembly" & New_Line &
               ".global " & Name & New_Line &
               ".type " & Name & ", %function" & New_Line &
               Name & ":" & New_Line &
               "    stp x29, x30, [sp, #-16]!" & New_Line &
               "    mov x29, sp" & New_Line);
      end case;
   end Emit_Prologue;

   procedure Emit_Epilogue (
      Content   : out Content_String;
      Config    : in ARM_Config;
      Status    : out Emitter_Status)
   is
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      case Config.Mode is
         when ARM32 | Thumb =>
            Content_Strings.Append (Content,
               "    add sp, sp, #16" & New_Line &
               "    pop {r4-r11, pc}" & New_Line);
         when ARM64_AArch64 =>
            Content_Strings.Append (Content,
               "    ldp x29, x30, [sp], #16" & New_Line &
               "    ret" & New_Line);
      end case;
   end Emit_Epilogue;

   procedure Emit_Load (
      Reg       : in Natural;
      Offset    : in Integer;
      Content   : out Content_String;
      Config    : in ARM_Config;
      Status    : out Emitter_Status)
   is
      Reg_Str : constant String := Natural'Image (Reg);
      Off_Str : constant String := Integer'Image (Offset);
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      case Config.Mode is
         when ARM32 | Thumb =>
            Content_Strings.Append (Content,
               "    ldr r" & Reg_Str (2 .. Reg_Str'Last) & 
               ", [sp, #" & Off_Str (2 .. Off_Str'Last) & "]" & New_Line);
         when ARM64_AArch64 =>
            Content_Strings.Append (Content,
               "    ldr x" & Reg_Str (2 .. Reg_Str'Last) &
               ", [sp, #" & Off_Str (2 .. Off_Str'Last) & "]" & New_Line);
      end case;
   end Emit_Load;

   procedure Emit_Store (
      Reg       : in Natural;
      Offset    : in Integer;
      Content   : out Content_String;
      Config    : in ARM_Config;
      Status    : out Emitter_Status)
   is
      Reg_Str : constant String := Natural'Image (Reg);
      Off_Str : constant String := Integer'Image (Offset);
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      case Config.Mode is
         when ARM32 | Thumb =>
            Content_Strings.Append (Content,
               "    str r" & Reg_Str (2 .. Reg_Str'Last) &
               ", [sp, #" & Off_Str (2 .. Off_Str'Last) & "]" & New_Line);
         when ARM64_AArch64 =>
            Content_Strings.Append (Content,
               "    str x" & Reg_Str (2 .. Reg_Str'Last) &
               ", [sp, #" & Off_Str (2 .. Off_Str'Last) & "]" & New_Line);
      end case;
   end Emit_Store;

end ARM_Emitter;
