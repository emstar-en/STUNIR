-- STUNIR Code Generation Utilities
-- DO-178C Level A
-- Phase 3a: Core Category Emitters

with STUNIR.Semantic_IR; use STUNIR.Semantic_IR;

package STUNIR.Emitters.CodeGen is
   pragma SPARK_Mode (On);

   type Indent_Level is range 0 .. 10;

   type Code_Generator is record
      Buffer       : IR_Code_Buffer;
      Indent       : Indent_Level := 0;
      Indent_Width : Positive range 2 .. 8 := 4;
   end record;

   procedure Initialize
     (Gen          : out Code_Generator;
      Indent_Width : Positive := 4)
   with
     Pre  => Indent_Width in 2 .. 8,
     Post => Gen.Indent = 0 and
             Gen.Indent_Width = Indent_Width and
             Code_Buffers.Length (Gen.Buffer) = 0;

   procedure Append_Line
     (Gen     : in out Code_Generator;
      Line    : in     String;
      Success :    out Boolean)
   with
     Pre  => Line'Length < Max_Code_Length,
     Post => (if Success then Code_Buffers.Length (Gen.Buffer) <= Max_Code_Length);

   procedure Append_Raw
     (Gen     : in out Code_Generator;
      Text    : in     String;
      Success :    out Boolean)
   with
     Pre  => Text'Length < Max_Code_Length,
     Post => (if Success then Code_Buffers.Length (Gen.Buffer) <= Max_Code_Length);

   procedure Increase_Indent (Gen : in out Code_Generator)
   with
     Pre  => Gen.Indent < Indent_Level'Last,
     Post => Gen.Indent = Gen.Indent'Old + 1;

   procedure Decrease_Indent (Gen : in out Code_Generator)
   with
     Pre  => Gen.Indent > 0,
     Post => Gen.Indent = Gen.Indent'Old - 1;

   procedure Get_Output
     (Gen    : in  Code_Generator;
      Output : out IR_Code_Buffer)
   with
     Post => Code_Buffers.Length (Output) = Code_Buffers.Length (Gen.Buffer);

   function Get_Indent_String (Gen : Code_Generator) return String
   with
     Global => null,
     Post => Get_Indent_String'Result'Length <= 80;

end STUNIR.Emitters.CodeGen;
