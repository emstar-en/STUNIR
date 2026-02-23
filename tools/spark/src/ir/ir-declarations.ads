-- STUNIR IR Declarations Package Specification
-- DO-178C Level A Compliant
-- SPARK 2014 Mode

pragma SPARK_Mode (On);

with IR.Types; use IR.Types;
with IR.Nodes; use IR.Nodes;

package IR.Declarations with
   SPARK_Mode => On
is
   --  Declaration node: wraps an IR_Node with declaration-specific fields.
   --  IR_Node is a discriminated record (not tagged), so we use composition.
   type Declaration_Node (Kind : IR_Node_Kind) is record
      Base       : IR_Node (Kind);
      Decl_Name  : IR_Name;
      Visibility : Visibility_Kind := Vis_Public;
   end record;
   
   -- Function declaration
   Max_Parameters : constant := 8;
   type Parameter_List is array (1 .. Max_Parameters) of Node_ID;
   
   type Function_Declaration is record
      Base           : Declaration_Node (Kind_Function_Decl);
      Return_Type    : Type_Reference;
      Param_Count    : Natural range 0 .. Max_Parameters := 0;
      Parameters     : Parameter_List;
      Body_ID        : Node_ID; -- Statement node ID
      Inline         : Inline_Hint := Inline_None;
      Is_Pure        : Boolean := False;
      Stack_Usage    : Natural := 0;
      Priority       : Integer := 0;
      Interrupt_Vec  : Natural := 0; -- 0 means not an interrupt handler
      Entry_Point    : Boolean := False;
   end record;
   
   -- Type declaration
   type Type_Declaration is record
      Base           : Declaration_Node (Kind_Type_Decl);
      Type_Def       : Type_Reference;
   end record;
   
   -- Constant declaration
   type Const_Declaration is record
      Base           : Declaration_Node (Kind_Const_Decl);
      Const_Type     : Type_Reference;
      Value_ID       : Node_ID; -- Expression node ID
      Compile_Time   : Boolean := True;
   end record;
   
   -- Variable declaration
   type Variable_Declaration is record
      Base           : Declaration_Node (Kind_Var_Decl);
      Var_Type       : Type_Reference;
      Init_ID        : Node_ID; -- Initializer expression
      Storage        : Storage_Class := Storage_Auto;
      Mutability     : Mutability_Kind := Mut_Mutable;
      Alignment      : Positive := 1;
      Is_Volatile    : Boolean := False;
   end record;
   
   -- Declaration validation
   function Is_Declaration_Kind (Kind : IR_Node_Kind) return Boolean is
      (Kind in Kind_Function_Decl .. Kind_Var_Decl);
   
   function Is_Valid_Declaration (Decl : Declaration_Node) return Boolean
      with Post => (if Is_Valid_Declaration'Result then
                       Is_Valid_Node_ID (Decl.Base.ID) and then
                       Name_Strings.Length (Decl.Decl_Name) > 0);
   
end IR.Declarations;
