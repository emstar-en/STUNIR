--  Emit Target C Family - C/C++ Code Emission
--  Phase: 3 (Emit)
--  SPARK_Mode: On
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

with Identifier_Strings;
use Identifier_Strings;

package body Emit_Target.C_Family is

   function Is_In_Nested_Block (Func : IR_Function; Idx : Step_Index) return Boolean is
   begin
      for J in Step_Index range 1 .. Func.Steps.Count loop
         declare
            S : constant IR_Step := Func.Steps.Steps (J);
         begin
            --  Check if step is in then block
            if S.Step_Type = Step_If then
               for B in Step_Index range S.Then_Start .. S.Then_Start + S.Then_Count - 1 loop
                  if B = Idx then
                     return True;
                  end if;
               end loop;
               --  Check if step is in else block
               for B in Step_Index range S.Else_Start .. S.Else_Start + S.Else_Count - 1 loop
                  if B = Idx then
                     return True;
                  end if;
               end loop;
            end if;
            --  Check if step is in while/for body
            if S.Step_Type = Step_While or S.Step_Type = Step_For then
               for B in Step_Index range S.Body_Start .. S.Body_Start + S.Body_Count - 1 loop
                  if B = Idx then
                     return True;
                  end if;
               end loop;
            end if;
         end;
      end loop;
      return False;
   end Is_In_Nested_Block;

   procedure Emit_Steps_C (Func : IR_Function; Append_Line : not null access procedure (Text : String)) is
      Step : IR_Step;
   begin
      if Func.Steps.Count = 0 then
         Append_Line ("    // TODO: implement");
         return;
      end if;

      for J in Step_Index range 1 .. Func.Steps.Count loop
         --  Skip steps that are part of nested blocks
         if Is_In_Nested_Block (Func, J) then
            goto Continue;
         end if;
         
         Step := Func.Steps.Steps (J);
         declare
            Val : constant String := To_String (Step.Value);
            Tgt : constant String := To_String (Step.Target);
            Cond : constant String := To_String (Step.Condition);
            Args : constant String := To_String (Step.Args);
            Init : constant String := To_String (Step.Init);
            Incr : constant String := To_String (Step.Increment);
         begin
            case Step.Step_Type is
               when Step_Assign =>
                  if Tgt'Length > 0 then
                     Append_Line ("    " & Tgt & " = " & Val & ";");
                  else
                     Append_Line ("    // assign: " & Val);
                  end if;
               when Step_Call =>
                  if Tgt'Length > 0 then
                     Append_Line ("    " & Tgt & " = " & Val & "(" & Args & ");");
                  else
                     Append_Line ("    " & Val & "(" & Args & ");");
                  end if;
               when Step_Return =>
                  if Val'Length > 0 then
                     Append_Line ("    return " & Val & ";");
                  else
                     Append_Line ("    return;");
                  end if;
               when Step_If =>
                  Append_Line ("    if (" & Cond & ") {");
                  --  Emit then block steps
                  for B in Step_Index range Step.Then_Start .. Step.Then_Start + Step.Then_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := To_String (Body_Step.Value);
                           B_Tgt : constant String := To_String (Body_Step.Target);
                           B_Args : constant String := To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        " & B_Tgt & " = " & B_Val & ";");
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        " & B_Tgt & " = " & B_Val & "(" & B_Args & ");");
                                 else
                                    Append_Line ("        " & B_Val & "(" & B_Args & ");");
                                 end if;
                              when Step_Return =>
                                 if B_Val'Length > 0 then
                                    Append_Line ("        return " & B_Val & ";");
                                 else
                                    Append_Line ("        return;");
                                 end if;
                              when others =>
                                 Append_Line ("        // step");
                           end case;
                        end;
                     end if;
                  end loop;
                  if Step.Else_Count > 0 then
                     Append_Line ("    } else {");
                     for B in Step_Index range Step.Else_Start .. Step.Else_Start + Step.Else_Count - 1 loop
                        if B <= Func.Steps.Count then
                           declare
                              Body_Step : constant IR_Step := Func.Steps.Steps (B);
                              B_Val : constant String := To_String (Body_Step.Value);
                              B_Tgt : constant String := To_String (Body_Step.Target);
                              B_Args : constant String := To_String (Body_Step.Args);
                           begin
                              case Body_Step.Step_Type is
                                 when Step_Assign =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("        " & B_Tgt & " = " & B_Val & ";");
                                    end if;
                                 when Step_Call =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("        " & B_Tgt & " = " & B_Val & "(" & B_Args & ");");
                                    else
                                       Append_Line ("        " & B_Val & "(" & B_Args & ");");
                                    end if;
                                 when Step_Return =>
                                    if B_Val'Length > 0 then
                                       Append_Line ("        return " & B_Val & ";");
                                    else
                                       Append_Line ("        return;");
                                    end if;
                                 when others =>
                                    Append_Line ("        // step");
                              end case;
                           end;
                        end if;
                     end loop;
                  end if;
                  Append_Line ("    }");
               when Step_While =>
                  Append_Line ("    while (" & Cond & ") {");
                  for B in Step_Index range Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := To_String (Body_Step.Value);
                           B_Tgt : constant String := To_String (Body_Step.Target);
                           B_Args : constant String := To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        " & B_Tgt & " = " & B_Val & ";");
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        " & B_Tgt & " = " & B_Val & "(" & B_Args & ");");
                                 else
                                    Append_Line ("        " & B_Val & "(" & B_Args & ");");
                                 end if;
                              when Step_Return =>
                                 if B_Val'Length > 0 then
                                    Append_Line ("        return " & B_Val & ";");
                                 else
                                    Append_Line ("        return;");
                                 end if;
                              when others =>
                                 Append_Line ("        // step");
                           end case;
                        end;
                     end if;
                  end loop;
                  Append_Line ("    }");
               when Step_For =>
                  Append_Line ("    for (" & Init & "; " & Cond & "; " & Incr & ") {");
                  for B in Step_Index range Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := To_String (Body_Step.Value);
                           B_Tgt : constant String := To_String (Body_Step.Target);
                           B_Args : constant String := To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        " & B_Tgt & " = " & B_Val & ";");
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        " & B_Tgt & " = " & B_Val & "(" & B_Args & ");");
                                 else
                                    Append_Line ("        " & B_Val & "(" & B_Args & ");");
                                 end if;
                              when Step_Return =>
                                 if B_Val'Length > 0 then
                                    Append_Line ("        return " & B_Val & ";");
                                 else
                                    Append_Line ("        return;");
                                 end if;
                              when others =>
                                 Append_Line ("        // step");
                           end case;
                        end;
                     end if;
                  end loop;
                  Append_Line ("    }");
               when Step_Nop =>
                  Append_Line ("    ;");
               when others =>
                  Append_Line ("    // TODO: unsupported step");
            end case;
         end;
         <<Continue>>
      end loop;
   end Emit_Steps_C;

end Emit_Target.C_Family;
