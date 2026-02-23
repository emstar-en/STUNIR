-------------------------------------------------------------------------------
--  STUNIR IR JSON Parser
--  DO-178C Level A Compliant
--  SPARK 2014 Mode
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

with STUNIR_Types;
with IR.Modules;
with STUNIR.Emitters.Node_Table;

package IR.JSON with
   SPARK_Mode => On
is
   use STUNIR_Types;

   -- Parse IR JSON into AST module and node table.
   procedure Parse_IR_JSON
     (JSON_Content : in     JSON_String;
      Module       :    out IR.Modules.IR_Module;
      Nodes        :    out STUNIR.Emitters.Node_Table.Node_Table;
      Status       :    out Status_Code)
   with
      Pre  => JSON_Strings.Length (JSON_Content) > 0,
      Post => Status in Status_Code'Range;

end IR.JSON;
