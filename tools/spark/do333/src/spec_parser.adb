--  STUNIR DO-333 Specification Parser
--  Implementation
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

package body Spec_Parser is

   --  ============================================================
   --  Helper: Check String Contains Substring
   --  ============================================================

   function Contains (Source : String; Pattern : String) return Boolean
   with
      Pre => Source'Length > 0 and then Pattern'Length > 0
   is
   begin
      if Pattern'Length > Source'Length then
         return False;
      end if;

      for I in Source'First .. Source'Last - Pattern'Length + 1 loop
         declare
            Match : Boolean := True;
         begin
            for J in Pattern'Range loop
               if Source (I + J - Pattern'First) /= Pattern (J) then
                  Match := False;
                  exit;
               end if;
            end loop;
            if Match then
               return True;
            end if;
         end;
      end loop;
      return False;
   end Contains;

   --  ============================================================
   --  Helper: Extract Expression After Keyword
   --  ============================================================

   procedure Extract_After_Arrow
     (Line    : String;
      Expr    : out String;
      Length  : out Natural;
      Found   : out Boolean)
   with
      Pre => Line'Length > 0
   is
      Arrow_Pos : Natural := 0;
      Start_Pos : Natural := 0;
   begin
      Expr := (others => ' ');
      Length := 0;
      Found := False;

      --  Find "=>" in line
      for I in Line'First .. Line'Last - 1 loop
         if Line (I) = '=' and then Line (I + 1) = '>' then
            Arrow_Pos := I + 2;
            exit;
         end if;
      end loop;

      if Arrow_Pos = 0 or else Arrow_Pos > Line'Last then
         return;
      end if;

      --  Skip whitespace
      Start_Pos := Arrow_Pos;
      while Start_Pos <= Line'Last and then Line (Start_Pos) = ' ' loop
         Start_Pos := Start_Pos + 1;
      end loop;

      if Start_Pos > Line'Last then
         return;
      end if;

      --  Copy expression (up to comma, semicolon, or end)
      declare
         Out_Idx : Natural := Expr'First;
      begin
         for I in Start_Pos .. Line'Last loop
            exit when Line (I) = ',' or else Line (I) = ';';
            if Out_Idx <= Expr'Last then
               Expr (Out_Idx) := Line (I);
               Out_Idx := Out_Idx + 1;
            end if;
         end loop;
         Length := Out_Idx - Expr'First;
      end;

      Found := Length > 0;
   end Extract_After_Arrow;

   --  ============================================================
   --  Contains_Pre
   --  ============================================================

   function Contains_Pre (Line : String) return Boolean is
   begin
      return Contains (Line, "Pre ") or else
             Contains (Line, "Pre=>") or else
             Contains (Line, "Pre =>") or else
             Contains (Line, "Precondition");
   end Contains_Pre;

   --  ============================================================
   --  Contains_Post
   --  ============================================================

   function Contains_Post (Line : String) return Boolean is
   begin
      return Contains (Line, "Post ") or else
             Contains (Line, "Post=>") or else
             Contains (Line, "Post =>") or else
             Contains (Line, "Postcondition");
   end Contains_Post;

   --  ============================================================
   --  Contains_Invariant
   --  ============================================================

   function Contains_Invariant (Line : String) return Boolean is
   begin
      return Contains (Line, "Loop_Invariant") or else
             Contains (Line, "Type_Invariant") or else
             Contains (Line, "Invariant");
   end Contains_Invariant;

   --  ============================================================
   --  Is_Ghost_Line
   --  ============================================================

   function Is_Ghost_Line (Line : String) return Boolean is
   begin
      return Contains (Line, "Ghost") or else
             Contains (Line, "pragma Ghost");
   end Is_Ghost_Line;

   --  ============================================================
   --  Parse Source Content
   --  ============================================================

   procedure Parse_Source_Content
     (Source    : String;
      Contracts : out Contract_Spec;
      Stats     : out Parse_Statistics;
      Result    : out Parse_Result)
   is
      Line_Start   : Natural := Source'First;
      Line_End     : Natural;
      Line_Num     : Natural := 1;
      Success      : Boolean;
      Expr_Buffer  : String (1 .. Max_Expression_Length) := (others => ' ');
      Expr_Len     : Natural;
      Found        : Boolean;
   begin
      Contracts := Empty_Contract;
      Stats := Empty_Statistics;

      if Source'Length = 0 then
         Result := Parse_Error_Empty;
         return;
      end if;

      --  Process line by line
      while Line_Start <= Source'Last loop
         --  Find end of line
         Line_End := Line_Start;
         while Line_End <= Source'Last and then
               Source (Line_End) /= ASCII.LF loop
            Line_End := Line_End + 1;
         end loop;

         --  Process this line
         if Line_End > Line_Start then
            declare
               Line : constant String := Source (Line_Start .. Line_End - 1);
            begin
               Stats.Total_Lines := Stats.Total_Lines + 1;

               --  Check for preconditions
               if Contains_Pre (Line) then
                  Extract_After_Arrow (Line, Expr_Buffer, Expr_Len, Found);
                  if Found and then Expr_Len > 0 then
                     Add_Precondition
                       (Contracts, Expr_Buffer (1 .. Expr_Len),
                        Line_Num, 1, Success);
                     if Success then
                        Stats.Pre_Count := Stats.Pre_Count + 1;
                        Stats.Contract_Lines := Stats.Contract_Lines + 1;
                     end if;
                  end if;
               end if;

               --  Check for postconditions
               if Contains_Post (Line) then
                  Extract_After_Arrow (Line, Expr_Buffer, Expr_Len, Found);
                  if Found and then Expr_Len > 0 then
                     Add_Postcondition
                       (Contracts, Expr_Buffer (1 .. Expr_Len),
                        Line_Num, 1, Success);
                     if Success then
                        Stats.Post_Count := Stats.Post_Count + 1;
                        Stats.Contract_Lines := Stats.Contract_Lines + 1;
                     end if;
                  end if;
               end if;

               --  Check for invariants
               if Contains_Invariant (Line) then
                  Extract_After_Arrow (Line, Expr_Buffer, Expr_Len, Found);
                  if Found and then Expr_Len > 0 then
                     Add_Invariant
                       (Contracts, Expr_Buffer (1 .. Expr_Len),
                        Invariant, Line_Num, 1, Success);
                     if Success then
                        Stats.Invariant_Count := Stats.Invariant_Count + 1;
                        Stats.Contract_Lines := Stats.Contract_Lines + 1;
                     end if;
                  end if;
               end if;

               --  Check for ghost code
               if Is_Ghost_Line (Line) then
                  Stats.Ghost_Count := Stats.Ghost_Count + 1;
               end if;
            end;
         end if;

         --  Move to next line
         Line_Start := Line_End + 1;
         Line_Num := Line_Num + 1;
      end loop;

      Result := Parse_Success;
   end Parse_Source_Content;

   --  ============================================================
   --  Parse Single Expression
   --  ============================================================

   procedure Parse_Single_Expression
     (Expr_Text : String;
      Kind      : Spec_Kind;
      Expr      : out Formal_Expression;
      Result    : out Parse_Result)
   is
   begin
      Make_Expression (Expr_Text, Kind, 0, 0, Expr);
      Result := Parse_Success;
   end Parse_Single_Expression;

end Spec_Parser;
