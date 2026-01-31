-- STUNIR Base Emitter Implementation
-- DO-178C Level A
-- Phase 3a: Core Category Emitters

package body STUNIR.Emitters is
   pragma SPARK_Mode (On);

   function Get_Category_Name (Cat : Target_Category) return String is
   begin
      case Cat is
         when Category_Embedded => return "Embedded";
         when Category_GPU      => return "GPU";
         when Category_WASM     => return "WASM";
         when Category_Assembly => return "Assembly";
         when Category_Polyglot => return "Polyglot";
      end case;
   end Get_Category_Name;

   function Get_Status_Name (Status : Emitter_Status) return String is
   begin
      case Status is
         when Status_Success         => return "Success";
         when Status_Error_Parse     => return "Parse Error";
         when Status_Error_Generate  => return "Generation Error";
         when Status_Error_IO        => return "I/O Error";
      end case;
   end Get_Status_Name;

end STUNIR.Emitters;
