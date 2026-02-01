-- v0.8.2: Recursive multi-level nesting support
-- This snippet shows the key recursive flattening logic

-- Declare the recursive procedure inside the function parsing block
procedure Flatten_Block (Block_JSON : String; Array_Pos : Natural; Depth : Natural := 0) is
   Stmt_Pos : Natural := Array_Pos + 1;
   Stmt_Start, Stmt_End : Natural;
begin
   -- Safety: Limit nesting depth to 5 levels
   if Depth > 5 then
      Put_Line ("[ERROR] Maximum nesting depth (5) exceeded");
      return;
   end if;
   
   while Module.Functions (Func_Idx).Stmt_Cnt < Max_Statements loop
      Get_Next_Object (Block_JSON, Stmt_Pos, Stmt_Start, Stmt_End);
      exit when Stmt_Start = 0 or Stmt_End = 0;
      
      declare
         Stmt_JSON : constant String := Block_JSON (Stmt_Start .. Stmt_End);
         Stmt_Type : constant String := Extract_String_Value (Stmt_JSON, "type");
      begin
         -- Reserve slot for this statement
         Module.Functions (Func_Idx).Stmt_Cnt := Module.Functions (Func_Idx).Stmt_Cnt + 1;
         declare
            Current_Idx : constant Positive := Module.Functions (Func_Idx).Stmt_Cnt;
         begin
            -- Initialize statement with defaults
            Module.Functions (Func_Idx).Statements (Current_Idx).Kind := Stmt_Nop;
            Module.Functions (Func_Idx).Statements (Current_Idx).Data := Code_Buffers.Null_Bounded_String;
            Module.Functions (Func_Idx).Statements (Current_Idx).Target := Name_Strings.Null_Bounded_String;
            Module.Functions (Func_Idx).Statements (Current_Idx).Value := Code_Buffers.Null_Bounded_String;
            Module.Functions (Func_Idx).Statements (Current_Idx).Condition := Code_Buffers.Null_Bounded_String;
            Module.Functions (Func_Idx).Statements (Current_Idx).Init_Expr := Code_Buffers.Null_Bounded_String;
            Module.Functions (Func_Idx).Statements (Current_Idx).Incr_Expr := Code_Buffers.Null_Bounded_String;
            Module.Functions (Func_Idx).Statements (Current_Idx).Block_Start := 0;
            Module.Functions (Func_Idx).Statements (Current_Idx).Block_Count := 0;
            Module.Functions (Func_Idx).Statements (Current_Idx).Else_Start := 0;
            Module.Functions (Func_Idx).Statements (Current_Idx).Else_Count := 0;
            
            -- Parse based on statement type
            if Stmt_Type = "assign" or Stmt_Type = "var_decl" then
               Module.Functions (Func_Idx).Statements (Current_Idx).Kind := Stmt_Assign;
               declare
                  Target_Str : constant String := Extract_String_Value (Stmt_JSON, "target");
                  Var_Name : constant String := Extract_String_Value (Stmt_JSON, "var_name");
                  Value_Str : constant String := Extract_String_Value (Stmt_JSON, "value");
                  Init_Str : constant String := Extract_String_Value (Stmt_JSON, "init");
               begin
                  if Target_Str'Length > 0 and Target_Str'Length <= Max_Name_Length then
                     Module.Functions (Func_Idx).Statements (Current_Idx).Target :=
                       Name_Strings.To_Bounded_String (Target_Str);
                  elsif Var_Name'Length > 0 and Var_Name'Length <= Max_Name_Length then
                     Module.Functions (Func_Idx).Statements (Current_Idx).Target :=
                       Name_Strings.To_Bounded_String (Var_Name);
                  end if;
                  
                  if Value_Str'Length > 0 and Value_Str'Length <= Max_Code_Length then
                     Module.Functions (Func_Idx).Statements (Current_Idx).Value :=
                       Code_Buffers.To_Bounded_String (Value_Str);
                  elsif Init_Str'Length > 0 and Init_Str'Length <= Max_Code_Length then
                     Module.Functions (Func_Idx).Statements (Current_Idx).Value :=
                       Code_Buffers.To_Bounded_String (Init_Str);
                  end if;
               end;
            
            elsif Stmt_Type = "return" then
               Module.Functions (Func_Idx).Statements (Current_Idx).Kind := Stmt_Return;
               declare
                  Value_Str : constant String := Extract_String_Value (Stmt_JSON, "value");
               begin
                  if Value_Str'Length > 0 and Value_Str'Length <= Max_Code_Length then
                     Module.Functions (Func_Idx).Statements (Current_Idx).Value :=
                       Code_Buffers.To_Bounded_String (Value_Str);
                  end if;
               end;
            
            elsif Stmt_Type = "call" then
               Module.Functions (Func_Idx).Statements (Current_Idx).Kind := Stmt_Call;
               declare
                  Func_Name : constant String := Extract_String_Value (Stmt_JSON, "func");
                  Args_Str : constant String := Extract_String_Value (Stmt_JSON, "args");
                  Assign_To : constant String := Extract_String_Value (Stmt_JSON, "assign_to");
               begin
                  if Func_Name'Length > 0 then
                     declare
                        Call_Expr : constant String := Func_Name & "(" & Args_Str & ")";
                     begin
                        if Call_Expr'Length <= Max_Code_Length then
                           Module.Functions (Func_Idx).Statements (Current_Idx).Value :=
                             Code_Buffers.To_Bounded_String (Call_Expr);
                        end if;
                     end;
                  end if;
                  if Assign_To'Length > 0 and Assign_To'Length <= Max_Name_Length then
                     Module.Functions (Func_Idx).Statements (Current_Idx).Target :=
                       Name_Strings.To_Bounded_String (Assign_To);
                  end if;
               end;
            
            elsif Stmt_Type = "if" then
               -- v0.8.2: Recursive handling of nested if statements
               Module.Functions (Func_Idx).Statements (Current_Idx).Kind := Stmt_If;
               declare
                  Cond_Str : constant String := Extract_String_Value (Stmt_JSON, "condition");
                  Then_Array_Pos : constant Natural := Find_Array (Stmt_JSON, "then_block");
                  Else_Array_Pos : constant Natural := Find_Array (Stmt_JSON, "else_block");
                  Then_Start_Idx : Natural := 0;
                  Then_Count_Val : Natural := 0;
                  Else_Start_Idx : Natural := 0;
                  Else_Count_Val : Natural := 0;
                  Count_Before : Natural;
               begin
                  -- Extract condition
                  if Cond_Str'Length > 0 and Cond_Str'Length <= Max_Code_Length then
                     Module.Functions (Func_Idx).Statements (Current_Idx).Condition :=
                       Code_Buffers.To_Bounded_String (Cond_Str);
                  end if;
                  
                  -- Recursively flatten then_block
                  if Then_Array_Pos > 0 then
                     Then_Start_Idx := Module.Functions (Func_Idx).Stmt_Cnt + 1;  -- 1-based
                     Count_Before := Module.Functions (Func_Idx).Stmt_Cnt;
                     Flatten_Block (Stmt_JSON, Then_Array_Pos, Depth + 1);  -- RECURSIVE CALL
                     Then_Count_Val := Module.Functions (Func_Idx).Stmt_Cnt - Count_Before;
                  end if;
                  
                  -- Recursively flatten else_block
                  if Else_Array_Pos > 0 then
                     Else_Start_Idx := Module.Functions (Func_Idx).Stmt_Cnt + 1;  -- 1-based
                     Count_Before := Module.Functions (Func_Idx).Stmt_Cnt;
                     Flatten_Block (Stmt_JSON, Else_Array_Pos, Depth + 1);  -- RECURSIVE CALL
                     Else_Count_Val := Module.Functions (Func_Idx).Stmt_Cnt - Count_Before;
                  end if;
                  
                  -- Fill in block indices
                  Module.Functions (Func_Idx).Statements (Current_Idx).Block_Start := Then_Start_Idx;
                  Module.Functions (Func_Idx).Statements (Current_Idx).Block_Count := Then_Count_Val;
                  Module.Functions (Func_Idx).Statements (Current_Idx).Else_Start := Else_Start_Idx;
                  Module.Functions (Func_Idx).Statements (Current_Idx).Else_Count := Else_Count_Val;
                  
                  Put_Line ("[INFO] Flattened if: then_block[" & Natural'Image(Then_Start_Idx) & ".." & 
                            Natural'Image(Then_Start_Idx + Then_Count_Val - 1) & "] else_block[" & 
                            Natural'Image(Else_Start_Idx) & ".." & Natural'Image(Else_Start_Idx + Else_Count_Val - 1) & "]");
               end;
            
            elsif Stmt_Type = "while" then
               -- v0.8.2: Recursive handling of nested while statements
               Module.Functions (Func_Idx).Statements (Current_Idx).Kind := Stmt_While;
               declare
                  Cond_Str : constant String := Extract_String_Value (Stmt_JSON, "condition");
                  Body_Array_Pos : constant Natural := Find_Array (Stmt_JSON, "body");
                  Body_Start_Idx : Natural := 0;
                  Body_Count_Val : Natural := 0;
                  Count_Before : Natural;
               begin
                  -- Extract condition
                  if Cond_Str'Length > 0 and Cond_Str'Length <= Max_Code_Length then
                     Module.Functions (Func_Idx).Statements (Current_Idx).Condition :=
                       Code_Buffers.To_Bounded_String (Cond_Str);
                  end if;
                  
                  -- Recursively flatten body
                  if Body_Array_Pos > 0 then
                     Body_Start_Idx := Module.Functions (Func_Idx).Stmt_Cnt + 1;
                     Count_Before := Module.Functions (Func_Idx).Stmt_Cnt;
                     Flatten_Block (Stmt_JSON, Body_Array_Pos, Depth + 1);  -- RECURSIVE CALL
                     Body_Count_Val := Module.Functions (Func_Idx).Stmt_Cnt - Count_Before;
                  end if;
                  
                  -- Fill in block indices
                  Module.Functions (Func_Idx).Statements (Current_Idx).Block_Start := Body_Start_Idx;
                  Module.Functions (Func_Idx).Statements (Current_Idx).Block_Count := Body_Count_Val;
                  
                  Put_Line ("[INFO] Flattened while: body[" & Natural'Image(Body_Start_Idx) & ".." & 
                            Natural'Image(Body_Start_Idx + Body_Count_Val - 1) & "]");
               end;
            
            elsif Stmt_Type = "for" then
               -- v0.8.2: Recursive handling of nested for statements
               Module.Functions (Func_Idx).Statements (Current_Idx).Kind := Stmt_For;
               declare
                  Init_Str : constant String := Extract_String_Value (Stmt_JSON, "init");
                  Cond_Str : constant String := Extract_String_Value (Stmt_JSON, "condition");
                  Incr_Str : constant String := Extract_String_Value (Stmt_JSON, "increment");
                  Body_Array_Pos : constant Natural := Find_Array (Stmt_JSON, "body");
                  Body_Start_Idx : Natural := 0;
                  Body_Count_Val : Natural := 0;
                  Count_Before : Natural;
               begin
                  -- Extract for loop components
                  if Init_Str'Length > 0 and Init_Str'Length <= Max_Code_Length then
                     Module.Functions (Func_Idx).Statements (Current_Idx).Init_Expr :=
                       Code_Buffers.To_Bounded_String (Init_Str);
                  end if;
                  if Cond_Str'Length > 0 and Cond_Str'Length <= Max_Code_Length then
                     Module.Functions (Func_Idx).Statements (Current_Idx).Condition :=
                       Code_Buffers.To_Bounded_String (Cond_Str);
                  end if;
                  if Incr_Str'Length > 0 and Incr_Str'Length <= Max_Code_Length then
                     Module.Functions (Func_Idx).Statements (Current_Idx).Incr_Expr :=
                       Code_Buffers.To_Bounded_String (Incr_Str);
                  end if;
                  
                  -- Recursively flatten body
                  if Body_Array_Pos > 0 then
                     Body_Start_Idx := Module.Functions (Func_Idx).Stmt_Cnt + 1;
                     Count_Before := Module.Functions (Func_Idx).Stmt_Cnt;
                     Flatten_Block (Stmt_JSON, Body_Array_Pos, Depth + 1);  -- RECURSIVE CALL
                     Body_Count_Val := Module.Functions (Func_Idx).Stmt_Cnt - Count_Before;
                  end if;
                  
                  -- Fill in block indices
                  Module.Functions (Func_Idx).Statements (Current_Idx).Block_Start := Body_Start_Idx;
                  Module.Functions (Func_Idx).Statements (Current_Idx).Block_Count := Body_Count_Val;
                  
                  Put_Line ("[INFO] Flattened for: body[" & Natural'Image(Body_Start_Idx) & ".." & 
                            Natural'Image(Body_Start_Idx + Body_Count_Val - 1) & "]");
               end;
            
            else
               -- Unknown statement type - keep as Stmt_Nop
               null;
            end if;
         end;
      end;
      
      Stmt_Pos := Stmt_End + 1;
   end loop;
end Flatten_Block;

-- Then call it for the function body:
-- Flatten_Block (Func_JSON, Body_Pos);
