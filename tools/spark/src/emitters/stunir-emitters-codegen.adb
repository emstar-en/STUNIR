-- STUNIR Code Generation Utilities (Body)
-- DO-178C Level A
-- Phase 3a: Core Category Emitters

package body STUNIR.Emitters.CodeGen is
   pragma SPARK_Mode (On);

   procedure Initialize
     (Gen          : out Code_Generator;
      Indent_Width : Positive := 4)
   is
   begin
      Gen.Buffer := Code_Buffers.To_Bounded_String ("");
      Gen.Indent := 0;
      Gen.Indent_Width := Indent_Width;
   end Initialize;

   procedure Append_Line
     (Gen     : in out Code_Generator;
      Line    : in     String;
      Success :    out Boolean)
   is
      Indent_Str : constant String := Get_Indent_String (Gen);
      New_Line   : constant String := Indent_Str & Line & ASCII.LF;
      Current_Len : constant Natural := Code_Buffers.Length (Gen.Buffer);
   begin
      Success := False;

      if Current_Len + New_Line'Length <= Max_Code_Length then
         Code_Buffers.Append (Gen.Buffer, New_Line);
         Success := True;
      end if;
   end Append_Line;

   procedure Append_Raw
     (Gen     : in out Code_Generator;
      Text    : in     String;
      Success :    out Boolean)
   is
      Current_Len : constant Natural := Code_Buffers.Length (Gen.Buffer);
   begin
      Success := False;

      if Current_Len + Text'Length <= Max_Code_Length then
         Code_Buffers.Append (Gen.Buffer, Text);
         Success := True;
      end if;
   end Append_Raw;

   procedure Increase_Indent (Gen : in out Code_Generator) is
   begin
      if Gen.Indent < Indent_Level'Last then
         Gen.Indent := Gen.Indent + 1;
      end if;
   end Increase_Indent;

   procedure Decrease_Indent (Gen : in out Code_Generator) is
   begin
      if Gen.Indent > 0 then
         Gen.Indent := Gen.Indent - 1;
      end if;
   end Decrease_Indent;

   procedure Get_Output
     (Gen    : in  Code_Generator;
      Output : out IR_Code_Buffer)
   is
   begin
      Output := Gen.Buffer;
   end Get_Output;

   function Get_Indent_String (Gen : Code_Generator) return String is
      Spaces_Needed : constant Natural :=
        Natural (Gen.Indent) * Gen.Indent_Width;
      Max_Spaces : constant := 80;
      Actual_Spaces : constant Natural :=
        (if Spaces_Needed > Max_Spaces then Max_Spaces else Spaces_Needed);
      Spaces : String (1 .. Actual_Spaces) := (others => ' ');
   begin
      return Spaces;
   end Get_Indent_String;

end STUNIR.Emitters.CodeGen;
