-- STUNIR IR Modules Package Specification
-- DO-178C Level A Compliant
-- SPARK 2014 Mode

pragma SPARK_Mode (On);

with IR.Types; use IR.Types;
with IR.Nodes; use IR.Nodes;

package IR.Modules with
   SPARK_Mode => On
is
   -- Import statement
   --  v0.8.6: Reduced from 64 to 8 to lower stack usage
   Max_Symbols : constant := 8;
   type Symbol_List is array (1 .. Max_Symbols) of IR_Name;
   
   type Import_Statement is record
      Module_Name  : IR_Name;
      Symbol_Count : Natural range 0 .. Max_Symbols := 0;
      Symbols      : Symbol_List;
      Import_All   : Boolean := False;
      Alias        : IR_Name;
   end record;
   
   -- Module metadata
   Max_Target_Categories : constant := 4;
   type Target_Category_List is array (1 .. Max_Target_Categories) of Target_Category;
   
   type Module_Metadata is record
      Target_Count       : Natural range 0 .. Max_Target_Categories := 0;
      Target_Categories  : Target_Category_List;
      Module_Safety      : Safety_Level := Level_None;  --  renamed: Safety_Level field shadowed type
      Optimization_Level : Natural range 0 .. 3 := 0;  --  O0..O3
   end record;
   
   -- Module structure
   --  v0.8.6: Reduced limits to lower stack usage
   Max_Imports       : constant := 2;
   Max_Exports       : constant := 4;
   Max_Declarations  : constant := 8;
   
   type Import_List is array (1 .. Max_Imports) of Import_Statement;
   type Export_List is array (1 .. Max_Exports) of IR_Name;
   type Declaration_List is array (1 .. Max_Declarations) of Node_ID;
   
   type IR_Module is record
      Base            : IR_Node (Kind_Module);
      Module_Name     : IR_Name;
      Import_Count    : Natural range 0 .. Max_Imports := 0;
      Imports         : Import_List;
      Export_Count    : Natural range 0 .. Max_Exports := 0;
      Exports         : Export_List;
      Decl_Count      : Natural range 0 .. Max_Declarations := 0;
      Declarations    : Declaration_List;
      Metadata        : Module_Metadata;
   end record;
   
   -- Module validation
   function Is_Valid_Module (M : IR_Module) return Boolean
      with Post => (if Is_Valid_Module'Result then
                       Is_Valid_Node_ID (M.Base.ID) and then
                       Name_Strings.Length (M.Module_Name) > 0);
   
   -- Module operations
   procedure Add_Import (
      M      : in out IR_Module;
      Import : Import_Statement;
      Success : out Boolean
   ) with
      Pre  => Is_Valid_Module (M),
      Post => Is_Valid_Module (M);
   
   procedure Add_Export (
      M      : in out IR_Module;
      Name   : IR_Name;
      Success : out Boolean
   ) with
      Pre  => Is_Valid_Module (M) and then Name_Strings.Length (Name) > 0,
      Post => Is_Valid_Module (M);
   
   procedure Add_Declaration (
      M       : in out IR_Module;
      Decl_ID : Node_ID;
      Success : out Boolean
   ) with
      Pre  => Is_Valid_Module (M) and then Is_Valid_Node_ID (Decl_ID),
      Post => Is_Valid_Module (M);
   
end IR.Modules;
