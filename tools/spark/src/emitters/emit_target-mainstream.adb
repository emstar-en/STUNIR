--  Emit Target Mainstream - Mainstream Language Code Emission
--  Phase: 3 (Emit)
--  SPARK_Mode: On
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

with STUNIR_Types;
use STUNIR_Types;

package body Emit_Target.Mainstream is

   --  Use clauses for nested packages
   package IStr renames Identifier_Strings;
   package TStr renames Type_Name_Strings;

   function Is_In_Nested_Block (Func : IR_Function; Idx : Step_Index) return Boolean is
   begin
      for J in Step_Index range 1 .. Func.Steps.Count loop
         declare
            S : constant IR_Step := Func.Steps.Steps (J);
         begin
            if S.Step_Type = Step_If then
               for B in Step_Index range S.Then_Start .. S.Then_Start + S.Then_Count - 1 loop
                  if B = Idx then return True; end if;
               end loop;
               for B in Step_Index range S.Else_Start .. S.Else_Start + S.Else_Count - 1 loop
                  if B = Idx then return True; end if;
               end loop;
            end if;
            if S.Step_Type = Step_While or S.Step_Type = Step_For then
               for B in Step_Index range S.Body_Start .. S.Body_Start + S.Body_Count - 1 loop
                  if B = Idx then return True; end if;
               end loop;
            end if;
         end;
      end loop;
      return False;
   end Is_In_Nested_Block;

   procedure Emit_Steps_Rust (Func : IR_Function; Append_Line : not null access procedure (Text : String)) is
   begin
      if Func.Steps.Count = 0 then
         Append_Line ("    todo!()");
         return;
      end if;

      for J in Step_Index range 1 .. Func.Steps.Count loop
         if Is_In_Nested_Block (Func, J) then
            goto Continue_Rust;
         end if;
         
         declare
            Step : constant IR_Step := Func.Steps.Steps (J);
            Val  : constant String := IStr.To_String (Step.Value);
            Tgt  : constant String := IStr.To_String (Step.Target);
            Cond : constant String := IStr.To_String (Step.Condition);
            Args : constant String := IStr.To_String (Step.Args);
            Init : constant String := IStr.To_String (Step.Init);
            Incr : constant String := IStr.To_String (Step.Increment);
         begin
            case Step.Step_Type is
               when Step_Assign =>
                  if Tgt'Length > 0 then
                     Append_Line ("    let " & Tgt & " = " & Val & ";");
                  else
                     Append_Line ("    // assign: " & Val);
                  end if;
               when Step_Call =>
                  if Tgt'Length > 0 then
                     Append_Line ("    let " & Tgt & " = " & Val & "(" & Args & ");");
                  else
                     Append_Line ("    " & Val & "(" & Args & ");");
                  end if;
               when Step_Return =>
                  if Val'Length > 0 then
                     Append_Line ("    " & Val);
                  else
                     Append_Line ("    ()");
                  end if;
               when Step_If =>
                  Append_Line ("    if " & Cond & " {");
                  for B in Step_Index range Step.Then_Start .. Step.Then_Start + Step.Then_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        let " & B_Tgt & " = " & B_Val & ";");
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        let " & B_Tgt & " = " & B_Val & "(" & B_Args & ");");
                                 else
                                    Append_Line ("        " & B_Val & "(" & B_Args & ");");
                                 end if;
                              when Step_Return =>
                                 Append_Line ("        " & B_Val);
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
                              B_Val : constant String := IStr.To_String (Body_Step.Value);
                              B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                              B_Args : constant String := IStr.To_String (Body_Step.Args);
                           begin
                              case Body_Step.Step_Type is
                                 when Step_Assign =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("        let " & B_Tgt & " = " & B_Val & ";");
                                    end if;
                                 when Step_Call =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("        let " & B_Tgt & " = " & B_Val & "(" & B_Args & ");");
                                    else
                                       Append_Line ("        " & B_Val & "(" & B_Args & ");");
                                    end if;
                                 when Step_Return =>
                                    Append_Line ("        " & B_Val);
                                 when others =>
                                    Append_Line ("        // step");
                              end case;
                           end;
                        end if;
                     end loop;
                  end if;
                  Append_Line ("    }");
               when Step_While =>
                  Append_Line ("    while " & Cond & " {");
                  for B in Step_Index range Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        let " & B_Tgt & " = " & B_Val & ";");
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        let " & B_Tgt & " = " & B_Val & "(" & B_Args & ");");
                                 else
                                    Append_Line ("        " & B_Val & "(" & B_Args & ");");
                                 end if;
                              when others =>
                                 Append_Line ("        // step");
                           end case;
                        end;
                     end if;
                  end loop;
                  Append_Line ("    }");
               when Step_For =>
                  Append_Line ("    for i in " & Init & ".." & Cond & " {");
                  for B in Step_Index range Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        let " & B_Tgt & " = " & B_Val & ";");
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        let " & B_Tgt & " = " & B_Val & "(" & B_Args & ");");
                                 else
                                    Append_Line ("        " & B_Val & "(" & B_Args & ");");
                                 end if;
                              when others =>
                                 Append_Line ("        // step");
                           end case;
                        end;
                     end if;
                  end loop;
                  Append_Line ("    }");
               when others =>
                  Append_Line ("    // TODO: unsupported step");
            end case;
         end;
         <<Continue_Rust>>
      end loop;
   end Emit_Steps_Rust;

   procedure Emit_Steps_Python (Func : IR_Function; Append_Line : not null access procedure (Text : String)) is
   begin
      if Func.Steps.Count = 0 then
         Append_Line ("    pass  # TODO: implement");
         return;
      end if;

      for J in Step_Index range 1 .. Func.Steps.Count loop
         if Is_In_Nested_Block (Func, J) then
            goto Continue_Python;
         end if;
         
         declare
            Step : constant IR_Step := Func.Steps.Steps (J);
            Val  : constant String := IStr.To_String (Step.Value);
            Tgt  : constant String := IStr.To_String (Step.Target);
            Cond : constant String := IStr.To_String (Step.Condition);
            Args : constant String := IStr.To_String (Step.Args);
            Init : constant String := IStr.To_String (Step.Init);
            Incr : constant String := IStr.To_String (Step.Increment);
         begin
            case Step.Step_Type is
               when Step_Assign =>
                  if Tgt'Length > 0 then
                     Append_Line ("    " & Tgt & " = " & Val);
                  else
                     Append_Line ("    # assign: " & Val);
                  end if;
               when Step_Call =>
                  if Tgt'Length > 0 then
                     Append_Line ("    " & Tgt & " = " & Val & "(" & Args & ")");
                  else
                     Append_Line ("    " & Val & "(" & Args & ")");
                  end if;
               when Step_Return =>
                  if Val'Length > 0 then
                     Append_Line ("    return " & Val);
                  else
                     Append_Line ("    return None");
                  end if;
               when Step_If =>
                  Append_Line ("    if " & Cond & ":");
                  for B in Step_Index range Step.Then_Start .. Step.Then_Start + Step.Then_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        " & B_Tgt & " = " & B_Val);
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        " & B_Tgt & " = " & B_Val & "(" & B_Args & ")");
                                 else
                                    Append_Line ("        " & B_Val & "(" & B_Args & ")");
                                 end if;
                              when Step_Return =>
                                 Append_Line ("        return " & B_Val);
                              when others =>
                                 Append_Line ("        pass");
                           end case;
                        end;
                     end if;
                  end loop;
                  if Step.Else_Count > 0 then
                     Append_Line ("    else:");
                     for B in Step_Index range Step.Else_Start .. Step.Else_Start + Step.Else_Count - 1 loop
                        if B <= Func.Steps.Count then
                           declare
                              Body_Step : constant IR_Step := Func.Steps.Steps (B);
                              B_Val : constant String := IStr.To_String (Body_Step.Value);
                              B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                              B_Args : constant String := IStr.To_String (Body_Step.Args);
                           begin
                              case Body_Step.Step_Type is
                                 when Step_Assign =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("        " & B_Tgt & " = " & B_Val);
                                    end if;
                                 when Step_Call =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("        " & B_Tgt & " = " & B_Val & "(" & B_Args & ")");
                                    else
                                       Append_Line ("        " & B_Val & "(" & B_Args & ")");
                                    end if;
                                 when Step_Return =>
                                    Append_Line ("        return " & B_Val);
                                 when others =>
                                    Append_Line ("        pass");
                              end case;
                           end;
                        end if;
                     end loop;
                  end if;
               when Step_While =>
                  Append_Line ("    while " & Cond & ":");
                  for B in Step_Index range Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        " & B_Tgt & " = " & B_Val);
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        " & B_Tgt & " = " & B_Val & "(" & B_Args & ")");
                                 else
                                    Append_Line ("        " & B_Val & "(" & B_Args & ")");
                                 end if;
                              when others =>
                                 Append_Line ("        pass");
                           end case;
                        end;
                     end if;
                  end loop;
               when Step_For =>
                  Append_Line ("    for i in range(" & Init & ", " & Cond & "):");
                  for B in Step_Index range Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        " & B_Tgt & " = " & B_Val);
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        " & B_Tgt & " = " & B_Val & "(" & B_Args & ")");
                                 else
                                    Append_Line ("        " & B_Val & "(" & B_Args & ")");
                                 end if;
                              when others =>
                                 Append_Line ("        pass");
                           end case;
                        end;
                     end if;
                  end loop;
               when others =>
                  Append_Line ("    # TODO: unsupported step");
            end case;
         end;
         <<Continue_Python>>
      end loop;
   end Emit_Steps_Python;

   procedure Emit_Steps_Go (Func : IR_Function; Append_Line : not null access procedure (Text : String)) is
   begin
      if Func.Steps.Count = 0 then
         Append_Line ("    // TODO: implement");
         return;
      end if;

      for J in Step_Index range 1 .. Func.Steps.Count loop
         if Is_In_Nested_Block (Func, J) then
            goto Continue_Go;
         end if;
         
         declare
            Step : constant IR_Step := Func.Steps.Steps (J);
            Val  : constant String := IStr.To_String (Step.Value);
            Tgt  : constant String := IStr.To_String (Step.Target);
            Cond : constant String := IStr.To_String (Step.Condition);
            Args : constant String := IStr.To_String (Step.Args);
            Init : constant String := IStr.To_String (Step.Init);
            Incr : constant String := IStr.To_String (Step.Increment);
         begin
            case Step.Step_Type is
               when Step_Assign =>
                  if Tgt'Length > 0 then
                     Append_Line ("    " & Tgt & " := " & Val);
                  else
                     Append_Line ("    // assign: " & Val);
                  end if;
               when Step_Call =>
                  if Tgt'Length > 0 then
                     Append_Line ("    " & Tgt & " := " & Val & "(" & Args & ")");
                  else
                     Append_Line ("    " & Val & "(" & Args & ")");
                  end if;
               when Step_Return =>
                  if Val'Length > 0 then
                     Append_Line ("    return " & Val);
                  else
                     Append_Line ("    return");
                  end if;
               when Step_If =>
                  Append_Line ("    if " & Cond & " {");
                  for B in Step_Index range Step.Then_Start .. Step.Then_Start + Step.Then_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        " & B_Tgt & " := " & B_Val);
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        " & B_Tgt & " := " & B_Val & "(" & B_Args & ")");
                                 else
                                    Append_Line ("        " & B_Val & "(" & B_Args & ")");
                                 end if;
                              when Step_Return =>
                                 Append_Line ("        return " & B_Val);
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
                              B_Val : constant String := IStr.To_String (Body_Step.Value);
                              B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                              B_Args : constant String := IStr.To_String (Body_Step.Args);
                           begin
                              case Body_Step.Step_Type is
                                 when Step_Assign =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("        " & B_Tgt & " := " & B_Val);
                                    end if;
                                 when Step_Call =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("        " & B_Tgt & " := " & B_Val & "(" & B_Args & ")");
                                    else
                                       Append_Line ("        " & B_Val & "(" & B_Args & ")");
                                    end if;
                                 when Step_Return =>
                                    Append_Line ("        return " & B_Val);
                                 when others =>
                                    Append_Line ("        // step");
                              end case;
                           end;
                        end if;
                     end loop;
                  end if;
                  Append_Line ("    }");
               when Step_While =>
                  Append_Line ("    for " & Cond & " {");
                  for B in Step_Index range Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        " & B_Tgt & " := " & B_Val);
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        " & B_Tgt & " := " & B_Val & "(" & B_Args & ")");
                                 else
                                    Append_Line ("        " & B_Val & "(" & B_Args & ")");
                                 end if;
                              when others =>
                                 Append_Line ("        // step");
                           end case;
                        end;
                     end if;
                  end loop;
                  Append_Line ("    }");
               when Step_For =>
                  Append_Line ("    for i := " & Init & "; i < " & Cond & "; i++ {");
                  for B in Step_Index range Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        " & B_Tgt & " := " & B_Val);
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        " & B_Tgt & " := " & B_Val & "(" & B_Args & ")");
                                 else
                                    Append_Line ("        " & B_Val & "(" & B_Args & ")");
                                 end if;
                              when others =>
                                 Append_Line ("        // step");
                           end case;
                        end;
                     end if;
                  end loop;
                  Append_Line ("    }");
               when others =>
                  Append_Line ("    // TODO: unsupported step");
            end case;
         end;
         <<Continue_Go>>
      end loop;
   end Emit_Steps_Go;

   procedure Emit_Steps_Java (Func : IR_Function; Append_Line : not null access procedure (Text : String)) is
   begin
      if Func.Steps.Count = 0 then
         Append_Line ("        // TODO: implement");
         return;
      end if;

      for J in Step_Index range 1 .. Func.Steps.Count loop
         if Is_In_Nested_Block (Func, J) then
            goto Continue_Java;
         end if;
         
         declare
            Step : constant IR_Step := Func.Steps.Steps (J);
            Val  : constant String := IStr.To_String (Step.Value);
            Tgt  : constant String := IStr.To_String (Step.Target);
            Cond : constant String := IStr.To_String (Step.Condition);
            Args : constant String := IStr.To_String (Step.Args);
            Init : constant String := IStr.To_String (Step.Init);
            Incr : constant String := IStr.To_String (Step.Increment);
         begin
            case Step.Step_Type is
               when Step_Assign =>
                  if Tgt'Length > 0 then
                     Append_Line ("        " & Tgt & " = " & Val & ";");
                  else
                     Append_Line ("        // assign: " & Val);
                  end if;
               when Step_Call =>
                  if Tgt'Length > 0 then
                     Append_Line ("        " & Tgt & " = " & Val & "(" & Args & ");");
                  else
                     Append_Line ("        " & Val & "(" & Args & ");");
                  end if;
               when Step_Return =>
                  if Val'Length > 0 then
                     Append_Line ("        return " & Val & ";");
                  else
                     Append_Line ("        return;");
                  end if;
               when Step_If =>
                  Append_Line ("        if (" & Cond & ") {");
                  for B in Step_Index range Step.Then_Start .. Step.Then_Start + Step.Then_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("            " & B_Tgt & " = " & B_Val & ";");
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("            " & B_Tgt & " = " & B_Val & "(" & B_Args & ");");
                                 else
                                    Append_Line ("            " & B_Val & "(" & B_Args & ");");
                                 end if;
                              when Step_Return =>
                                 Append_Line ("            return " & B_Val & ";");
                              when others =>
                                 Append_Line ("            // step");
                           end case;
                        end;
                     end if;
                  end loop;
                  if Step.Else_Count > 0 then
                     Append_Line ("        } else {");
                     for B in Step_Index range Step.Else_Start .. Step.Else_Start + Step.Else_Count - 1 loop
                        if B <= Func.Steps.Count then
                           declare
                              Body_Step : constant IR_Step := Func.Steps.Steps (B);
                              B_Val : constant String := IStr.To_String (Body_Step.Value);
                              B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                              B_Args : constant String := IStr.To_String (Body_Step.Args);
                           begin
                              case Body_Step.Step_Type is
                                 when Step_Assign =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("            " & B_Tgt & " = " & B_Val & ";");
                                    end if;
                                 when Step_Call =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("            " & B_Tgt & " = " & B_Val & "(" & B_Args & ");");
                                    else
                                       Append_Line ("            " & B_Val & "(" & B_Args & ");");
                                    end if;
                                 when Step_Return =>
                                    Append_Line ("            return " & B_Val & ";");
                                 when others =>
                                    Append_Line ("            // step");
                              end case;
                           end;
                        end if;
                     end loop;
                  end if;
                  Append_Line ("        }");
               when Step_While =>
                  Append_Line ("        while (" & Cond & ") {");
                  for B in Step_Index range Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("            " & B_Tgt & " = " & B_Val & ";");
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("            " & B_Tgt & " = " & B_Val & "(" & B_Args & ");");
                                 else
                                    Append_Line ("            " & B_Val & "(" & B_Args & ");");
                                 end if;
                              when others =>
                                 Append_Line ("            // step");
                           end case;
                        end;
                     end if;
                  end loop;
                  Append_Line ("        }");
               when Step_For =>
                  Append_Line ("        for (int i = " & Init & "; i < " & Cond & "; i++) {");
                  for B in Step_Index range Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("            " & B_Tgt & " = " & B_Val & ";");
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("            " & B_Tgt & " = " & B_Val & "(" & B_Args & ");");
                                 else
                                    Append_Line ("            " & B_Val & "(" & B_Args & ");");
                                 end if;
                              when others =>
                                 Append_Line ("            // step");
                           end case;
                        end;
                     end if;
                  end loop;
                  Append_Line ("        }");
               when others =>
                  Append_Line ("        // TODO: unsupported step");
            end case;
         end;
         <<Continue_Java>>
      end loop;
   end Emit_Steps_Java;

   procedure Emit_Steps_JavaScript (Func : IR_Function; Append_Line : not null access procedure (Text : String)) is
   begin
      if Func.Steps.Count = 0 then
         Append_Line ("    // TODO: implement");
         return;
      end if;

      for J in Step_Index range 1 .. Func.Steps.Count loop
         if Is_In_Nested_Block (Func, J) then
            goto Continue_JS;
         end if;
         
         declare
            Step : constant IR_Step := Func.Steps.Steps (J);
            Val  : constant String := IStr.To_String (Step.Value);
            Tgt  : constant String := IStr.To_String (Step.Target);
            Cond : constant String := IStr.To_String (Step.Condition);
            Args : constant String := IStr.To_String (Step.Args);
            Init : constant String := IStr.To_String (Step.Init);
            Incr : constant String := IStr.To_String (Step.Increment);
         begin
            case Step.Step_Type is
               when Step_Assign =>
                  if Tgt'Length > 0 then
                     Append_Line ("    let " & Tgt & " = " & Val & ";");
                  else
                     Append_Line ("    // assign: " & Val);
                  end if;
               when Step_Call =>
                  if Tgt'Length > 0 then
                     Append_Line ("    let " & Tgt & " = " & Val & "(" & Args & ");");
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
                  for B in Step_Index range Step.Then_Start .. Step.Then_Start + Step.Then_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        let " & B_Tgt & " = " & B_Val & ";");
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        let " & B_Tgt & " = " & B_Val & "(" & B_Args & ");");
                                 else
                                    Append_Line ("        " & B_Val & "(" & B_Args & ");");
                                 end if;
                              when Step_Return =>
                                 Append_Line ("        return " & B_Val & ";");
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
                              B_Val : constant String := IStr.To_String (Body_Step.Value);
                              B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                              B_Args : constant String := IStr.To_String (Body_Step.Args);
                           begin
                              case Body_Step.Step_Type is
                                 when Step_Assign =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("        let " & B_Tgt & " = " & B_Val & ";");
                                    end if;
                                 when Step_Call =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("        let " & B_Tgt & " = " & B_Val & "(" & B_Args & ");");
                                    else
                                       Append_Line ("        " & B_Val & "(" & B_Args & ");");
                                    end if;
                                 when Step_Return =>
                                    Append_Line ("        return " & B_Val & ";");
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
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        let " & B_Tgt & " = " & B_Val & ";");
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        let " & B_Tgt & " = " & B_Val & "(" & B_Args & ");");
                                 else
                                    Append_Line ("        " & B_Val & "(" & B_Args & ");");
                                 end if;
                              when others =>
                                 Append_Line ("        // step");
                           end case;
                        end;
                     end if;
                  end loop;
                  Append_Line ("    }");
               when Step_For =>
                  Append_Line ("    for (let i = " & Init & "; i < " & Cond & "; i++) {");
                  for B in Step_Index range Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        let " & B_Tgt & " = " & B_Val & ";");
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        let " & B_Tgt & " = " & B_Val & "(" & B_Args & ");");
                                 else
                                    Append_Line ("        " & B_Val & "(" & B_Args & ");");
                                 end if;
                              when others =>
                                 Append_Line ("        // step");
                           end case;
                        end;
                     end if;
                  end loop;
                  Append_Line ("    }");
               when others =>
                  Append_Line ("    // TODO: unsupported step");
            end case;
         end;
         <<Continue_JS>>
      end loop;
   end Emit_Steps_JavaScript;

   procedure Emit_Steps_CSharp (Func : IR_Function; Append_Line : not null access procedure (Text : String)) is
   begin
      if Func.Steps.Count = 0 then
         Append_Line ("        // TODO: implement");
         return;
      end if;

      for J in Step_Index range 1 .. Func.Steps.Count loop
         if Is_In_Nested_Block (Func, J) then
            goto Continue_CSharp;
         end if;
         
         declare
            Step : constant IR_Step := Func.Steps.Steps (J);
            Val  : constant String := IStr.To_String (Step.Value);
            Tgt  : constant String := IStr.To_String (Step.Target);
            Cond : constant String := IStr.To_String (Step.Condition);
            Args : constant String := IStr.To_String (Step.Args);
            Init : constant String := IStr.To_String (Step.Init);
            Incr : constant String := IStr.To_String (Step.Increment);
         begin
            case Step.Step_Type is
               when Step_Assign =>
                  if Tgt'Length > 0 then
                     Append_Line ("        " & Tgt & " = " & Val & ";");
                  else
                     Append_Line ("        // assign: " & Val);
                  end if;
               when Step_Call =>
                  if Tgt'Length > 0 then
                     Append_Line ("        " & Tgt & " = " & Val & "(" & Args & ");");
                  else
                     Append_Line ("        " & Val & "(" & Args & ");");
                  end if;
               when Step_Return =>
                  if Val'Length > 0 then
                     Append_Line ("        return " & Val & ";");
                  else
                     Append_Line ("        return;");
                  end if;
               when Step_If =>
                  Append_Line ("        if (" & Cond & ") {");
                  for B in Step_Index range Step.Then_Start .. Step.Then_Start + Step.Then_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("            " & B_Tgt & " = " & B_Val & ";");
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("            " & B_Tgt & " = " & B_Val & "(" & B_Args & ");");
                                 else
                                    Append_Line ("            " & B_Val & "(" & B_Args & ");");
                                 end if;
                              when Step_Return =>
                                 Append_Line ("            return " & B_Val & ";");
                              when others =>
                                 Append_Line ("            // step");
                           end case;
                        end;
                     end if;
                  end loop;
                  if Step.Else_Count > 0 then
                     Append_Line ("        } else {");
                     for B in Step_Index range Step.Else_Start .. Step.Else_Start + Step.Else_Count - 1 loop
                        if B <= Func.Steps.Count then
                           declare
                              Body_Step : constant IR_Step := Func.Steps.Steps (B);
                              B_Val : constant String := IStr.To_String (Body_Step.Value);
                              B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                              B_Args : constant String := IStr.To_String (Body_Step.Args);
                           begin
                              case Body_Step.Step_Type is
                                 when Step_Assign =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("            " & B_Tgt & " = " & B_Val & ";");
                                    end if;
                                 when Step_Call =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("            " & B_Tgt & " = " & B_Val & "(" & B_Args & ");");
                                    else
                                       Append_Line ("            " & B_Val & "(" & B_Args & ");");
                                    end if;
                                 when Step_Return =>
                                    Append_Line ("            return " & B_Val & ";");
                                 when others =>
                                    Append_Line ("            // step");
                              end case;
                           end;
                        end if;
                     end loop;
                  end if;
                  Append_Line ("        }");
               when Step_While =>
                  Append_Line ("        while (" & Cond & ") {");
                  for B in Step_Index range Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("            " & B_Tgt & " = " & B_Val & ";");
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("            " & B_Tgt & " = " & B_Val & "(" & B_Args & ");");
                                 else
                                    Append_Line ("            " & B_Val & "(" & B_Args & ");");
                                 end if;
                              when others =>
                                 Append_Line ("            // step");
                           end case;
                        end;
                     end if;
                  end loop;
                  Append_Line ("        }");
               when Step_For =>
                  Append_Line ("        for (int i = " & Init & "; i < " & Cond & "; i++) {");
                  for B in Step_Index range Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("            " & B_Tgt & " = " & B_Val & ";");
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("            " & B_Tgt & " = " & B_Val & "(" & B_Args & ");");
                                 else
                                    Append_Line ("            " & B_Val & "(" & B_Args & ");");
                                 end if;
                              when others =>
                                 Append_Line ("            // step");
                           end case;
                        end;
                     end if;
                  end loop;
                  Append_Line ("        }");
               when others =>
                  Append_Line ("        // TODO: unsupported step");
            end case;
         end;
         <<Continue_CSharp>>
      end loop;
   end Emit_Steps_CSharp;

   procedure Emit_Steps_Swift (Func : IR_Function; Append_Line : not null access procedure (Text : String)) is
   begin
      if Func.Steps.Count = 0 then
         Append_Line ("    // TODO: implement");
         return;
      end if;

      for J in Step_Index range 1 .. Func.Steps.Count loop
         if Is_In_Nested_Block (Func, J) then
            goto Continue_Swift;
         end if;
         
         declare
            Step : constant IR_Step := Func.Steps.Steps (J);
            Val  : constant String := IStr.To_String (Step.Value);
            Tgt  : constant String := IStr.To_String (Step.Target);
            Cond : constant String := IStr.To_String (Step.Condition);
            Args : constant String := IStr.To_String (Step.Args);
            Init : constant String := IStr.To_String (Step.Init);
            Incr : constant String := IStr.To_String (Step.Increment);
         begin
            case Step.Step_Type is
               when Step_Assign =>
                  if Tgt'Length > 0 then
                     Append_Line ("    let " & Tgt & " = " & Val);
                  else
                     Append_Line ("    // assign: " & Val);
                  end if;
               when Step_Call =>
                  if Tgt'Length > 0 then
                     Append_Line ("    let " & Tgt & " = " & Val & "(" & Args & ")");
                  else
                     Append_Line ("    " & Val & "(" & Args & ")");
                  end if;
               when Step_Return =>
                  if Val'Length > 0 then
                     Append_Line ("    return " & Val);
                  else
                     Append_Line ("    return");
                  end if;
               when Step_If =>
                  Append_Line ("    if " & Cond & " {");
                  for B in Step_Index range Step.Then_Start .. Step.Then_Start + Step.Then_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        let " & B_Tgt & " = " & B_Val);
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        let " & B_Tgt & " = " & B_Val & "(" & B_Args & ")");
                                 else
                                    Append_Line ("        " & B_Val & "(" & B_Args & ")");
                                 end if;
                              when Step_Return =>
                                 Append_Line ("        return " & B_Val);
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
                              B_Val : constant String := IStr.To_String (Body_Step.Value);
                              B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                              B_Args : constant String := IStr.To_String (Body_Step.Args);
                           begin
                              case Body_Step.Step_Type is
                                 when Step_Assign =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("        let " & B_Tgt & " = " & B_Val);
                                    end if;
                                 when Step_Call =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("        let " & B_Tgt & " = " & B_Val & "(" & B_Args & ")");
                                    else
                                       Append_Line ("        " & B_Val & "(" & B_Args & ")");
                                    end if;
                                 when Step_Return =>
                                    Append_Line ("        return " & B_Val);
                                 when others =>
                                    Append_Line ("        // step");
                              end case;
                           end;
                        end if;
                     end loop;
                  end if;
                  Append_Line ("    }");
               when Step_While =>
                  Append_Line ("    while " & Cond & " {");
                  for B in Step_Index range Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        let " & B_Tgt & " = " & B_Val);
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        let " & B_Tgt & " = " & B_Val & "(" & B_Args & ")");
                                 else
                                    Append_Line ("        " & B_Val & "(" & B_Args & ")");
                                 end if;
                              when others =>
                                 Append_Line ("        // step");
                           end case;
                        end;
                     end if;
                  end loop;
                  Append_Line ("    }");
               when Step_For =>
                  Append_Line ("    for i in " & Init & "..<" & Cond & " {");
                  for B in Step_Index range Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        let " & B_Tgt & " = " & B_Val);
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        let " & B_Tgt & " = " & B_Val & "(" & B_Args & ")");
                                 else
                                    Append_Line ("        " & B_Val & "(" & B_Args & ")");
                                 end if;
                              when others =>
                                 Append_Line ("        // step");
                           end case;
                        end;
                     end if;
                  end loop;
                  Append_Line ("    }");
               when others =>
                  Append_Line ("    // TODO: unsupported step");
            end case;
         end;
         <<Continue_Swift>>
      end loop;
   end Emit_Steps_Swift;

   procedure Emit_Steps_Kotlin (Func : IR_Function; Append_Line : not null access procedure (Text : String)) is
   begin
      if Func.Steps.Count = 0 then
         Append_Line ("    // TODO: implement");
         return;
      end if;

      for J in Step_Index range 1 .. Func.Steps.Count loop
         if Is_In_Nested_Block (Func, J) then
            goto Continue_Kotlin;
         end if;
         
         declare
            Step : constant IR_Step := Func.Steps.Steps (J);
            Val  : constant String := IStr.To_String (Step.Value);
            Tgt  : constant String := IStr.To_String (Step.Target);
            Cond : constant String := IStr.To_String (Step.Condition);
            Args : constant String := IStr.To_String (Step.Args);
            Init : constant String := IStr.To_String (Step.Init);
            Incr : constant String := IStr.To_String (Step.Increment);
         begin
            case Step.Step_Type is
               when Step_Assign =>
                  if Tgt'Length > 0 then
                     Append_Line ("    val " & Tgt & " = " & Val);
                  else
                     Append_Line ("    // assign: " & Val);
                  end if;
               when Step_Call =>
                  if Tgt'Length > 0 then
                     Append_Line ("    val " & Tgt & " = " & Val & "(" & Args & ")");
                  else
                     Append_Line ("    " & Val & "(" & Args & ")");
                  end if;
               when Step_Return =>
                  if Val'Length > 0 then
                     Append_Line ("    return " & Val);
                  else
                     Append_Line ("    return");
                  end if;
               when Step_If =>
                  Append_Line ("    if (" & Cond & ") {");
                  for B in Step_Index range Step.Then_Start .. Step.Then_Start + Step.Then_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        val " & B_Tgt & " = " & B_Val);
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        val " & B_Tgt & " = " & B_Val & "(" & B_Args & ")");
                                 else
                                    Append_Line ("        " & B_Val & "(" & B_Args & ")");
                                 end if;
                              when Step_Return =>
                                 Append_Line ("        return " & B_Val);
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
                              B_Val : constant String := IStr.To_String (Body_Step.Value);
                              B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                              B_Args : constant String := IStr.To_String (Body_Step.Args);
                           begin
                              case Body_Step.Step_Type is
                                 when Step_Assign =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("        val " & B_Tgt & " = " & B_Val);
                                    end if;
                                 when Step_Call =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("        val " & B_Tgt & " = " & B_Val & "(" & B_Args & ")");
                                    else
                                       Append_Line ("        " & B_Val & "(" & B_Args & ")");
                                    end if;
                                 when Step_Return =>
                                    Append_Line ("        return " & B_Val);
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
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        val " & B_Tgt & " = " & B_Val);
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        val " & B_Tgt & " = " & B_Val & "(" & B_Args & ")");
                                 else
                                    Append_Line ("        " & B_Val & "(" & B_Args & ")");
                                 end if;
                              when others =>
                                 Append_Line ("        // step");
                           end case;
                        end;
                     end if;
                  end loop;
                  Append_Line ("    }");
               when Step_For =>
                  Append_Line ("    for (i in " & Init & " until " & Cond & ") {");
                  for B in Step_Index range Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        val " & B_Tgt & " = " & B_Val);
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        val " & B_Tgt & " = " & B_Val & "(" & B_Args & ")");
                                 else
                                    Append_Line ("        " & B_Val & "(" & B_Args & ")");
                                 end if;
                              when others =>
                                 Append_Line ("        // step");
                           end case;
                        end;
                     end if;
                  end loop;
                  Append_Line ("    }");
               when others =>
                  Append_Line ("    // TODO: unsupported step");
            end case;
         end;
         <<Continue_Kotlin>>
      end loop;
   end Emit_Steps_Kotlin;

   procedure Emit_Steps_SPARK (Func : IR_Function; Append_Line : not null access procedure (Text : String)) is
   begin
      if Func.Steps.Count = 0 then
         Append_Line ("   --  TODO: implement");
         Append_Line ("   return Integer'First;");
         return;
      end if;

      for J in Step_Index range 1 .. Func.Steps.Count loop
         if Is_In_Nested_Block (Func, J) then
            goto Continue_SPARK;
         end if;
         
         declare
            Step : constant IR_Step := Func.Steps.Steps (J);
            Val  : constant String := IStr.To_String (Step.Value);
            Tgt  : constant String := IStr.To_String (Step.Target);
            Cond : constant String := IStr.To_String (Step.Condition);
            Args : constant String := IStr.To_String (Step.Args);
            Init : constant String := IStr.To_String (Step.Init);
            Incr : constant String := IStr.To_String (Step.Increment);
         begin
            case Step.Step_Type is
               when Step_Assign =>
                  if Tgt'Length > 0 then
                     Append_Line ("   " & Tgt & " := " & Val & ";");
                  else
                     Append_Line ("   --  assign: " & Val);
                  end if;
               when Step_Call =>
                  if Tgt'Length > 0 then
                     Append_Line ("   " & Tgt & " := " & Val & " (" & Args & ");");
                  else
                     Append_Line ("   " & Val & " (" & Args & ");");
                  end if;
               when Step_Return =>
                  if Val'Length > 0 then
                     Append_Line ("   return " & Val & ";");
                  else
                     Append_Line ("   return;");
                  end if;
               when Step_If =>
                  Append_Line ("   if " & Cond & " then");
                  for B in Step_Index range Step.Then_Start .. Step.Then_Start + Step.Then_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("      " & B_Tgt & " := " & B_Val & ";");
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("      " & B_Tgt & " := " & B_Val & " (" & B_Args & ");");
                                 else
                                    Append_Line ("      " & B_Val & " (" & B_Args & ");");
                                 end if;
                              when Step_Return =>
                                 Append_Line ("      return " & B_Val & ";");
                              when others =>
                                 Append_Line ("      --  step");
                           end case;
                        end;
                     end if;
                  end loop;
                  if Step.Else_Count > 0 then
                     Append_Line ("   else");
                     for B in Step_Index range Step.Else_Start .. Step.Else_Start + Step.Else_Count - 1 loop
                        if B <= Func.Steps.Count then
                           declare
                              Body_Step : constant IR_Step := Func.Steps.Steps (B);
                              B_Val : constant String := IStr.To_String (Body_Step.Value);
                              B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                              B_Args : constant String := IStr.To_String (Body_Step.Args);
                           begin
                              case Body_Step.Step_Type is
                                 when Step_Assign =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("      " & B_Tgt & " := " & B_Val & ";");
                                    end if;
                                 when Step_Call =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("      " & B_Tgt & " := " & B_Val & " (" & B_Args & ");");
                                    else
                                       Append_Line ("      " & B_Val & " (" & B_Args & ");");
                                    end if;
                                 when Step_Return =>
                                    Append_Line ("      return " & B_Val & ";");
                                 when others =>
                                    Append_Line ("      --  step");
                              end case;
                           end;
                        end if;
                     end loop;
                  end if;
                  Append_Line ("   end if;");
               when Step_While =>
                  Append_Line ("   while " & Cond & " loop");
                  for B in Step_Index range Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("      " & B_Tgt & " := " & B_Val & ";");
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("      " & B_Tgt & " := " & B_Val & " (" & B_Args & ");");
                                 else
                                    Append_Line ("      " & B_Val & " (" & B_Args & ");");
                                 end if;
                              when others =>
                                 Append_Line ("      --  step");
                           end case;
                        end;
                     end if;
                  end loop;
                  Append_Line ("   end loop;");
               when Step_For =>
                  Append_Line ("   for I in " & Init & " .. " & Cond & " loop");
                  for B in Step_Index range Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("      " & B_Tgt & " := " & B_Val & ";");
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("      " & B_Tgt & " := " & B_Val & " (" & B_Args & ");");
                                 else
                                    Append_Line ("      " & B_Val & " (" & B_Args & ");");
                                 end if;
                              when others =>
                                 Append_Line ("      --  step");
                           end case;
                        end;
                     end if;
                  end loop;
                  Append_Line ("   end loop;");
               when others =>
                  Append_Line ("   --  TODO: unsupported step");
            end case;
         end;
         <<Continue_SPARK>>
      end loop;
   end Emit_Steps_SPARK;

   procedure Emit_Steps_Ada (Func : IR_Function; Append_Line : not null access procedure (Text : String)) is
   begin
      --  Ada emitter is identical to SPARK for now
      --  Future: SPARK may add pragma SPARK_Mode, contracts, etc.
      if Func.Steps.Count = 0 then
         Append_Line ("   --  TODO: implement");
         Append_Line ("   return Integer'First;");
         return;
      end if;

      for J in Step_Index range 1 .. Func.Steps.Count loop
         if Is_In_Nested_Block (Func, J) then
            goto Continue_Ada;
         end if;
         
         declare
            Step : constant IR_Step := Func.Steps.Steps (J);
            Val  : constant String := IStr.To_String (Step.Value);
            Tgt  : constant String := IStr.To_String (Step.Target);
            Cond : constant String := IStr.To_String (Step.Condition);
            Args : constant String := IStr.To_String (Step.Args);
            Init : constant String := IStr.To_String (Step.Init);
            Incr : constant String := IStr.To_String (Step.Increment);
         begin
            case Step.Step_Type is
               when Step_Assign =>
                  if Tgt'Length > 0 then
                     Append_Line ("   " & Tgt & " := " & Val & ";");
                  else
                     Append_Line ("   --  assign: " & Val);
                  end if;
               when Step_Call =>
                  if Tgt'Length > 0 then
                     Append_Line ("   " & Tgt & " := " & Val & " (" & Args & ");");
                  else
                     Append_Line ("   " & Val & " (" & Args & ");");
                  end if;
               when Step_Return =>
                  if Val'Length > 0 then
                     Append_Line ("   return " & Val & ";");
                  else
                     Append_Line ("   return;");
                  end if;
               when Step_If =>
                  Append_Line ("   if " & Cond & " then");
                  for B in Step_Index range Step.Then_Start .. Step.Then_Start + Step.Then_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("      " & B_Tgt & " := " & B_Val & ";");
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("      " & B_Tgt & " := " & B_Val & " (" & B_Args & ");");
                                 else
                                    Append_Line ("      " & B_Val & " (" & B_Args & ");");
                                 end if;
                              when Step_Return =>
                                 Append_Line ("      return " & B_Val & ";");
                              when others =>
                                 Append_Line ("      --  step");
                           end case;
                        end;
                     end if;
                  end loop;
                  if Step.Else_Count > 0 then
                     Append_Line ("   else");
                     for B in Step_Index range Step.Else_Start .. Step.Else_Start + Step.Else_Count - 1 loop
                        if B <= Func.Steps.Count then
                           declare
                              Body_Step : constant IR_Step := Func.Steps.Steps (B);
                              B_Val : constant String := IStr.To_String (Body_Step.Value);
                              B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                              B_Args : constant String := IStr.To_String (Body_Step.Args);
                           begin
                              case Body_Step.Step_Type is
                                 when Step_Assign =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("      " & B_Tgt & " := " & B_Val & ";");
                                    end if;
                                 when Step_Call =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("      " & B_Tgt & " := " & B_Val & " (" & B_Args & ");");
                                    else
                                       Append_Line ("      " & B_Val & " (" & B_Args & ");");
                                    end if;
                                 when Step_Return =>
                                    Append_Line ("      return " & B_Val & ";");
                                 when others =>
                                    Append_Line ("      --  step");
                              end case;
                           end;
                        end if;
                     end loop;
                  end if;
                  Append_Line ("   end if;");
               when Step_While =>
                  Append_Line ("   while " & Cond & " loop");
                  for B in Step_Index range Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("      " & B_Tgt & " := " & B_Val & ";");
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("      " & B_Tgt & " := " & B_Val & " (" & B_Args & ");");
                                 else
                                    Append_Line ("      " & B_Val & " (" & B_Args & ");");
                                 end if;
                              when others =>
                                 Append_Line ("      --  step");
                           end case;
                        end;
                     end if;
                  end loop;
                  Append_Line ("   end loop;");
               when Step_For =>
                  Append_Line ("   for I in " & Init & " .. " & Cond & " loop");
                  for B in Step_Index range Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("      " & B_Tgt & " := " & B_Val & ";");
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("      " & B_Tgt & " := " & B_Val & " (" & B_Args & ");");
                                 else
                                    Append_Line ("      " & B_Val & " (" & B_Args & ");");
                                 end if;
                              when others =>
                                 Append_Line ("      --  step");
                           end case;
                        end;
                     end if;
                  end loop;
                  Append_Line ("   end loop;");
               when others =>
                  Append_Line ("   --  TODO: unsupported step");
            end case;
         end;
         <<Continue_Ada>>
      end loop;
   end Emit_Steps_Ada;

end Emit_Target.Mainstream;
