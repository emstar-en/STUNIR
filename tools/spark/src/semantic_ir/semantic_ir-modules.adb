-- STUNIR Semantic IR Modules Package Body
-- DO-178C Level A Compliant
-- SPARK 2014 Mode

pragma SPARK_Mode (On);

package body Semantic_IR.Modules is

   function Is_Valid_Module (M : IR_Module) return Boolean is
      use Name_Strings;
   begin
      return Is_Valid_Node_ID (M.Base.Node_ID) and then
             Is_Valid_Hash (M.Base.Hash) and then
             Length (M.Module_Name) > 0 and then
             M.Import_Count <= Max_Imports and then
             M.Export_Count <= Max_Exports and then
             M.Decl_Count <= Max_Declarations;
   end Is_Valid_Module;
   
   procedure Add_Import (
      M      : in out IR_Module;
      Import : Import_Statement;
      Success : out Boolean
   ) is
   begin
      if M.Import_Count < Max_Imports then
         M.Import_Count := M.Import_Count + 1;
         M.Imports (M.Import_Count) := Import;
         Success := True;
      else
         Success := False;
      end if;
   end Add_Import;
   
   procedure Add_Export (
      M      : in out IR_Module;
      Name   : IR_Name;
      Success : out Boolean
   ) is
   begin
      if M.Export_Count < Max_Exports then
         M.Export_Count := M.Export_Count + 1;
         M.Exports (M.Export_Count) := Name;
         Success := True;
      else
         Success := False;
      end if;
   end Add_Export;
   
   procedure Add_Declaration (
      M       : in out IR_Module;
      Decl_ID : Node_ID;
      Success : out Boolean
   ) is
   begin
      if M.Decl_Count < Max_Declarations then
         M.Decl_Count := M.Decl_Count + 1;
         M.Declarations (M.Decl_Count) := Decl_ID;
         Success := True;
      else
         Success := False;
      end if;
   end Add_Declaration;
   
end Semantic_IR.Modules;
