--  STUNIR x86 Assembly Emitter - Ada SPARK Implementation

pragma SPARK_Mode (On);

package body X86_Emitter is

   New_Line : constant Character := ASCII.LF;

   procedure Emit_Prologue (
      Func_Name : in Identifier_String;
      Content   : out Content_String;
      Config    : in X86_Config;
      Status    : out Emitter_Status)
   is
      Name : constant String := Identifier_Strings.To_String (Func_Name);
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      case Config.Mode is
         when X86_32 =>
            if Config.Syntax = AT_T_Syntax then
               Content_Strings.Append (Content,
                  "# STUNIR Generated x86 Assembly" & New_Line &
                  ".global " & Name & New_Line &
                  ".type " & Name & ", @function" & New_Line &
                  Name & ":" & New_Line &
                  "    pushl %ebp" & New_Line &
                  "    movl %esp, %ebp" & New_Line &
                  "    subl $16, %esp" & New_Line);
            else
               Content_Strings.Append (Content,
                  "; STUNIR Generated x86 Assembly" & New_Line &
                  "global " & Name & New_Line &
                  Name & ":" & New_Line &
                  "    push ebp" & New_Line &
                  "    mov ebp, esp" & New_Line &
                  "    sub esp, 16" & New_Line);
            end if;

         when X86_64 =>
            if Config.Syntax = AT_T_Syntax then
               Content_Strings.Append (Content,
                  "# STUNIR Generated x86_64 Assembly" & New_Line &
                  ".global " & Name & New_Line &
                  ".type " & Name & ", @function" & New_Line &
                  Name & ":" & New_Line &
                  "    pushq %rbp" & New_Line &
                  "    movq %rsp, %rbp" & New_Line &
                  "    subq $32, %rsp" & New_Line);
            else
               Content_Strings.Append (Content,
                  "; STUNIR Generated x86_64 Assembly" & New_Line &
                  "global " & Name & New_Line &
                  Name & ":" & New_Line &
                  "    push rbp" & New_Line &
                  "    mov rbp, rsp" & New_Line &
                  "    sub rsp, 32" & New_Line);
            end if;
      end case;
   end Emit_Prologue;

   procedure Emit_Epilogue (
      Content   : out Content_String;
      Config    : in X86_Config;
      Status    : out Emitter_Status)
   is
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      case Config.Mode is
         when X86_32 =>
            if Config.Syntax = AT_T_Syntax then
               Content_Strings.Append (Content,
                  "    leave" & New_Line &
                  "    ret" & New_Line);
            else
               Content_Strings.Append (Content,
                  "    leave" & New_Line &
                  "    ret" & New_Line);
            end if;
         when X86_64 =>
            if Config.Syntax = AT_T_Syntax then
               Content_Strings.Append (Content,
                  "    leave" & New_Line &
                  "    ret" & New_Line);
            else
               Content_Strings.Append (Content,
                  "    leave" & New_Line &
                  "    ret" & New_Line);
            end if;
      end case;
   end Emit_Epilogue;

end X86_Emitter;
