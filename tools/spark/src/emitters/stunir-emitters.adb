-- STUNIR Base Emitter Implementation
-- DO-178C Level A
-- Phase 3a: Core Category Emitters

package body STUNIR.Emitters is
   pragma SPARK_Mode (On);

   function Get_Category_Name (Cat : Target_Category) return String is
   begin
      case Cat is
         when Semantic_IR.Types.Target_Embedded        => return "Embedded";
         when Semantic_IR.Types.Target_Realtime        => return "Realtime";
         when Semantic_IR.Types.Target_Safety_Critical => return "SafetyCritical";
         when Semantic_IR.Types.Target_GPU             => return "GPU";
         when Semantic_IR.Types.Target_WASM            => return "WASM";
         when Semantic_IR.Types.Target_Native          => return "Native";
         when Semantic_IR.Types.Target_JIT             => return "JIT";
         when Semantic_IR.Types.Target_Interpreter     => return "Interpreter";
         when Semantic_IR.Types.Target_Functional      => return "Functional";
         when Semantic_IR.Types.Target_Logic           => return "Logic";
         when Semantic_IR.Types.Target_Constraint      => return "Constraint";
         when Semantic_IR.Types.Target_Dataflow        => return "Dataflow";
         when Semantic_IR.Types.Target_Reactive        => return "Reactive";
         when Semantic_IR.Types.Target_Quantum         => return "Quantum";
         when Semantic_IR.Types.Target_Neuromorphic    => return "Neuromorphic";
         when Semantic_IR.Types.Target_Biocomputing    => return "Biocomputing";
         when Semantic_IR.Types.Target_Molecular       => return "Molecular";
         when Semantic_IR.Types.Target_Optical         => return "Optical";
         when Semantic_IR.Types.Target_Reversible      => return "Reversible";
         when Semantic_IR.Types.Target_Analog          => return "Analog";
         when Semantic_IR.Types.Target_Stochastic      => return "Stochastic";
         when Semantic_IR.Types.Target_Fuzzy           => return "Fuzzy";
         when Semantic_IR.Types.Target_Approximate     => return "Approximate";
         when Semantic_IR.Types.Target_Probabilistic   => return "Probabilistic";
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
