--  STUNIR DO-330 Template Engine Implementation
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

package body Template_Engine is

   --  ============================================================
   --  Initialize_Context
   --  ============================================================

   procedure Initialize_Context
     (Context : out Template_Context;
      Kind    : Template_Kind)
   is
   begin
      Context := (
         Content     => (others => ' '),
         Content_Len => 0,
         Variables   => (others => Null_Template_Variable),
         Var_Count   => 0,
         Kind        => Kind,
         Is_Loaded   => False
      );
   end Initialize_Context;

   --  ============================================================
   --  Load_Template_Content
   --  ============================================================

   procedure Load_Template_Content
     (Context : in out Template_Context;
      Content : String;
      Status  : out Process_Status)
   is
   begin
      if Content'Length > Max_Template_Size then
         Status := Output_Buffer_Overflow;
         Context.Is_Loaded := False;
         return;
      end if;

      --  Copy content to template buffer
      Context.Content_Len := Content'Length;
      for I in 1 .. Content'Length loop
         Context.Content (I) := Content (Content'First + I - 1);
      end loop;

      --  Clear remaining buffer
      for I in Content'Length + 1 .. Max_Template_Size loop
         Context.Content (I) := ' ';
      end loop;

      Context.Is_Loaded := True;
      Status := Success;
   end Load_Template_Content;

   --  ============================================================
   --  Set_Variable
   --  ============================================================

   procedure Set_Variable
     (Context : in out Template_Context;
      Name    : String;
      Value   : String;
      Status  : out Process_Status)
   is
      Idx : Variable_Count;
   begin
      --  Check if variable already exists
      Idx := Find_Variable (Context, Name);

      if Idx > 0 then
         --  Update existing variable
         for I in 1 .. Value'Length loop
            Context.Variables (Idx).Value (I) := Value (Value'First + I - 1);
         end loop;
         Context.Variables (Idx).Value_Len := Value'Length;
         Context.Variables (Idx).Is_Set := True;
         Status := Success;
      elsif Context.Var_Count < Max_Variables then
         --  Add new variable
         Context.Var_Count := Context.Var_Count + 1;
         Idx := Context.Var_Count;

         --  Initialize the variable record
         Context.Variables (Idx).Name := (others => ' ');
         Context.Variables (Idx).Value := (others => ' ');

         --  Copy name
         for I in 1 .. Name'Length loop
            Context.Variables (Idx).Name (I) := Name (Name'First + I - 1);
         end loop;
         Context.Variables (Idx).Name_Len := Name'Length;

         --  Copy value
         for I in 1 .. Value'Length loop
            Context.Variables (Idx).Value (I) := Value (Value'First + I - 1);
         end loop;
         Context.Variables (Idx).Value_Len := Value'Length;

         Context.Variables (Idx).Is_Set := True;
         Status := Success;
      else
         Status := Output_Buffer_Overflow;
      end if;
   end Set_Variable;

   --  ============================================================
   --  Clear_Variables
   --  ============================================================

   procedure Clear_Variables
     (Context : in out Template_Context)
   is
   begin
      Context.Var_Count := 0;
      Context.Variables := (others => Null_Template_Variable);
   end Clear_Variables;

   --  ============================================================
   --  Find_Variable
   --  ============================================================

   function Find_Variable
     (Context : Template_Context;
      Name    : String) return Variable_Count
   is
      Match : Boolean;
   begin
      for I in 1 .. Context.Var_Count loop
         if Context.Variables (I).Name_Len = Name'Length then
            Match := True;
            for J in 1 .. Name'Length loop
               if Context.Variables (I).Name (J) /= Name (Name'First + J - 1) then
                  Match := False;
                  exit;
               end if;
            end loop;
            if Match then
               return I;
            end if;
         end if;
      end loop;
      return 0;
   end Find_Variable;

   --  ============================================================
   --  Variable_Exists
   --  ============================================================

   function Variable_Exists
     (Context : Template_Context;
      Name    : String) return Boolean
   is
   begin
      return Find_Variable (Context, Name) > 0;
   end Variable_Exists;

   --  ============================================================
   --  Process_Template
   --  Replaces {{VARIABLE}} patterns with values
   --  ============================================================

   procedure Process_Template
     (Context : Template_Context;
      Output  : out Output_Content;
      Out_Len : out Output_Length_Type;
      Status  : out Process_Status)
   is
      In_Pos   : Template_Length_Type := 1;
      Out_Pos  : Output_Length_Type := 1;
      Var_Name : Name_String;
      Var_Len  : Name_Length_Type;
      Var_Idx  : Variable_Count;

      --  Helper to check for delimiter at position
      function Match_Delim
        (Content : Template_Content;
         Pos     : Template_Length_Type;
         Delim   : String;
         Len     : Template_Length_Type) return Boolean
      is
      begin
         if Pos + Delim'Length - 1 > Len then
            return False;
         end if;
         for I in 1 .. Delim'Length loop
            if Content (Pos + I - 1) /= Delim (Delim'First + I - 1) then
               return False;
            end if;
         end loop;
         return True;
      end Match_Delim;

   begin
      Output := (others => ' ');
      Out_Len := 0;
      Status := Success;

      while In_Pos <= Context.Content_Len loop
         --  Check for variable start delimiter "{{"
         if Match_Delim (Context.Content, In_Pos, Variable_Start_Delim, Context.Content_Len) then
            --  Found "{{", extract variable name
            In_Pos := In_Pos + 2;  --  Skip "{{"
            Var_Name := (others => ' ');
            Var_Len := 0;

            --  Read variable name until "}}"
            while In_Pos <= Context.Content_Len and then
                  not Match_Delim (Context.Content, In_Pos, Variable_End_Delim, Context.Content_Len)
            loop
               if Var_Len < Max_Name_Length then
                  Var_Len := Var_Len + 1;
                  Var_Name (Var_Len) := Context.Content (In_Pos);
               end if;
               In_Pos := In_Pos + 1;
            end loop;

            --  Skip "}}"
            if In_Pos + 1 <= Context.Content_Len then
               In_Pos := In_Pos + 2;
            else
               In_Pos := In_Pos + 1;
            end if;

            --  Find and substitute variable
            Var_Idx := Find_Variable (Context, Var_Name (1 .. Var_Len));
            if Var_Idx > 0 and then Context.Variables (Var_Idx).Is_Set then
               --  Copy variable value to output
               for I in 1 .. Context.Variables (Var_Idx).Value_Len loop
                  if Out_Pos <= Max_Output_Size then
                     Output (Out_Pos) := Context.Variables (Var_Idx).Value (I);
                     Out_Pos := Out_Pos + 1;
                  else
                     Status := Output_Buffer_Overflow;
                     Out_Len := Max_Output_Size;
                     return;
                  end if;
               end loop;
            else
               --  Variable not found, keep original placeholder
               if Out_Pos + Var_Len + 4 <= Max_Output_Size then
                  Output (Out_Pos) := '{';
                  Output (Out_Pos + 1) := '{';
                  Out_Pos := Out_Pos + 2;
                  for I in 1 .. Var_Len loop
                     Output (Out_Pos) := Var_Name (I);
                     Out_Pos := Out_Pos + 1;
                  end loop;
                  Output (Out_Pos) := '}';
                  Output (Out_Pos + 1) := '}';
                  Out_Pos := Out_Pos + 2;
               end if;
            end if;
         else
            --  Copy regular character
            if Out_Pos <= Max_Output_Size then
               Output (Out_Pos) := Context.Content (In_Pos);
               Out_Pos := Out_Pos + 1;
            else
               Status := Output_Buffer_Overflow;
               Out_Len := Max_Output_Size;
               return;
            end if;
            In_Pos := In_Pos + 1;
         end if;
      end loop;

      Out_Len := Out_Pos - 1;
   end Process_Template;

   --  ============================================================
   --  Set_DO330_Standard_Variables
   --  ============================================================

   procedure Set_DO330_Standard_Variables
     (Context    : in out Template_Context;
      Tool_Name  : String;
      Version    : String;
      TQL        : TQL_Level;
      DAL        : DAL_Level;
      Author     : String;
      Date       : String;
      Status     : out Process_Status)
   is
      Temp_Status : Process_Status;
   begin
      Status := Success;

      Set_Variable (Context, "TOOL_NAME", Tool_Name, Temp_Status);
      if Temp_Status /= Success then
         Status := Temp_Status;
         return;
      end if;

      Set_Variable (Context, "TOOL_VERSION", Version, Temp_Status);
      if Temp_Status /= Success then
         Status := Temp_Status;
         return;
      end if;

      Set_Variable (Context, "TQL_LEVEL", TQL_To_String (TQL), Temp_Status);
      if Temp_Status /= Success then
         Status := Temp_Status;
         return;
      end if;

      Set_Variable (Context, "DAL_LEVEL", DAL_To_String (DAL), Temp_Status);
      if Temp_Status /= Success then
         Status := Temp_Status;
         return;
      end if;

      Set_Variable (Context, "AUTHOR", Author, Temp_Status);
      if Temp_Status /= Success then
         Status := Temp_Status;
         return;
      end if;

      Set_Variable (Context, "QUALIFICATION_DATE", Date, Temp_Status);
      if Temp_Status /= Success then
         Status := Temp_Status;
         return;
      end if;

      --  Set schema version
      Set_Variable (Context, "SCHEMA_VERSION", "stunir.do330.v1", Temp_Status);
      if Temp_Status /= Success then
         Status := Temp_Status;
      end if;
   end Set_DO330_Standard_Variables;

   --  ============================================================
   --  Status_Message
   --  ============================================================

   function Status_Message (Status : Process_Status) return String is
   begin
      case Status is
         when Success =>
            return "Operation completed successfully";
         when Template_Not_Loaded =>
            return "Template has not been loaded";
         when Variable_Not_Found =>
            return "Variable not found in context";
         when Output_Buffer_Overflow =>
            return "Output buffer overflow";
         when Invalid_Template_Format =>
            return "Invalid template format";
         when File_Read_Error =>
            return "Error reading file";
         when File_Write_Error =>
            return "Error writing file";
      end case;
   end Status_Message;

end Template_Engine;
